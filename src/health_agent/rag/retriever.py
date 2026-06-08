import copy
import math

import voyageai
from langchain_core.documents import Document
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from health_agent.config import Settings
from health_agent.db import AgentResource, AgentResourceChunk, get_session_factory
from health_agent.db.models import EMBEDDING_DIMENSIONS
from health_agent.models import get_embeddings_model
from health_agent.rag.resources import filesystem_resource_manifest

_voyage_client_cache: voyageai.Client | None = None


def _database_resource_manifest(settings: Settings) -> dict[str, str]:
    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(select(AgentResource.source_path, AgentResource.content_hash)).all()
    return {source_path: content_hash for source_path, content_hash in rows}


def _chunk_to_document(chunk: AgentResourceChunk) -> Document:
    metadata = {
        "source": chunk.source,
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
        "title": chunk.title,
        "author": chunk.author,
        "header_path": chunk.header_path,
    }
    for key in ("h1", "h2", "h3"):
        value = getattr(chunk, key)
        if value:
            metadata[key] = value
    return Document(page_content=chunk.content, metadata=metadata)


def chunk_to_document(chunk: AgentResourceChunk) -> Document:
    return _chunk_to_document(chunk)


def _content_id(doc: Document) -> str:
    source = doc.metadata.get("source_path") or doc.metadata.get("source") or ""
    return f"{source}:{doc.page_content}"


def _copy_document(doc: Document) -> Document:
    return Document(page_content=doc.page_content, metadata=copy.deepcopy(doc.metadata))


def _ensure_retrieval_metadata(doc: Document) -> dict:
    retrieval = doc.metadata.get("retrieval")
    if not isinstance(retrieval, dict):
        retrieval = {}
    queries = retrieval.get("queries")
    if not isinstance(queries, list):
        retrieval["queries"] = []
    doc.metadata["retrieval"] = retrieval
    return retrieval


def _add_retrieval_signal(
    doc: Document,
    *,
    query: str,
    channel: str,
    rank: int,
    score: float | None = None,
    distance: float | None = None,
) -> None:
    retrieval = _ensure_retrieval_metadata(doc)
    if query not in retrieval["queries"]:
        retrieval["queries"].append(query)

    if channel == "vector":
        existing_rank = retrieval.get("vector_rank")
        if existing_rank is None or rank < existing_rank:
            retrieval["vector_rank"] = rank
        if distance is not None:
            existing_distance = retrieval.get("vector_distance")
            if existing_distance is None or distance < existing_distance:
                retrieval["vector_distance"] = distance
        return

    if channel == "keyword":
        existing_rank = retrieval.get("keyword_rank")
        if existing_rank is None or rank < existing_rank:
            retrieval["keyword_rank"] = rank
        if score is not None:
            existing_score = retrieval.get("keyword_score")
            if existing_score is None or score > existing_score:
                retrieval["keyword_score"] = score


def _merge_retrieval_metadata(target: Document, incoming: Document) -> None:
    target_retrieval = _ensure_retrieval_metadata(target)
    incoming_retrieval = incoming.metadata.get("retrieval")
    if not isinstance(incoming_retrieval, dict):
        return

    for query in incoming_retrieval.get("queries", []):
        if query not in target_retrieval["queries"]:
            target_retrieval["queries"].append(query)

    for key in ("vector_rank", "keyword_rank"):
        incoming_value = incoming_retrieval.get(key)
        if incoming_value is None:
            continue
        existing_value = target_retrieval.get(key)
        if existing_value is None or incoming_value < existing_value:
            target_retrieval[key] = incoming_value

    incoming_distance = incoming_retrieval.get("vector_distance")
    if incoming_distance is not None:
        existing_distance = target_retrieval.get("vector_distance")
        if existing_distance is None or incoming_distance < existing_distance:
            target_retrieval["vector_distance"] = incoming_distance

    incoming_score = incoming_retrieval.get("keyword_score")
    if incoming_score is not None:
        existing_score = target_retrieval.get("keyword_score")
        if existing_score is None or incoming_score > existing_score:
            target_retrieval["keyword_score"] = incoming_score


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    k: int = 60,
) -> list[Document]:
    """Fuse ranked lists and carry retrieval provenance into metadata."""
    doc_scores: dict[str, tuple[float, Document]] = {}
    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results, start=1):
            doc_id = _content_id(doc)
            score = weight / (k + rank)
            if doc_id in doc_scores:
                existing_score, existing_doc = doc_scores[doc_id]
                _merge_retrieval_metadata(existing_doc, doc)
                doc_scores[doc_id] = (existing_score + score, existing_doc)
            else:
                doc_scores[doc_id] = (score, _copy_document(doc))

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    fused_docs = [doc for _, doc in sorted_docs]
    for score, doc in sorted_docs:
        _ensure_retrieval_metadata(doc)["rrf_score"] = score
    return fused_docs


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _maximal_marginal_relevance(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    lambda_mult: float,
    k: int,
) -> list[int]:
    if not candidate_embeddings or k <= 0:
        return []

    query_similarities = [
        _cosine_similarity(query_embedding, candidate_embedding)
        for candidate_embedding in candidate_embeddings
    ]
    selected = [max(range(len(candidate_embeddings)), key=query_similarities.__getitem__)]
    remaining = set(range(len(candidate_embeddings))) - set(selected)

    while remaining and len(selected) < min(k, len(candidate_embeddings)):
        best_index = None
        best_score = float("-inf")
        for candidate_index in remaining:
            diversity_penalty = max(
                _cosine_similarity(
                    candidate_embeddings[candidate_index],
                    candidate_embeddings[selected_index],
                )
                for selected_index in selected
            )
            mmr_score = (
                lambda_mult * query_similarities[candidate_index]
                - (1 - lambda_mult) * diversity_penalty
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = candidate_index

        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)

    return selected


def _query_vector_rows(query: str, settings: Settings, limit: int):
    if not settings.database_url.strip():
        raise RuntimeError("DATABASE_URL must be set for vector retrieval.")
    if settings.embedding_dimensions != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "EMBEDDING_DIMENSIONS does not match the current database schema. "
            f"Expected {EMBEDDING_DIMENSIONS}."
        )

    query_embedding = get_embeddings_model(settings).embed_query(query)
    distance = AgentResourceChunk.embedding.cosine_distance(query_embedding).label("distance")
    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(
            select(AgentResourceChunk, distance)
            .order_by(distance.asc(), AgentResourceChunk.chunk_index.asc())
            .limit(limit)
        ).all()

    return query_embedding, rows


def _query_vector_candidate_chunks(
    query: str,
    settings: Settings,
    *,
    limit: int,
) -> list[Document]:
    _, rows = _query_vector_rows(query, settings, limit)
    docs: list[Document] = []
    for rank, (chunk, distance) in enumerate(rows, start=1):
        doc = _chunk_to_document(chunk)
        _add_retrieval_signal(
            doc,
            query=query,
            channel="vector",
            rank=rank,
            distance=float(distance),
        )
        docs.append(doc)
    return docs


def query_vector_candidate_chunks(
    query: str,
    settings: Settings,
    *,
    limit: int,
) -> list[Document]:
    return _query_vector_candidate_chunks(query, settings, limit=limit)


def query_vector_chunks(query: str, settings: Settings) -> list[Document]:
    query_embedding, rows = _query_vector_rows(query, settings, settings.retrieval_fetch_k)
    if not rows:
        return []

    candidate_chunks = [row[0] for row in rows]
    candidate_embeddings = [list(chunk.embedding) for chunk in candidate_chunks]
    selected_indices = _maximal_marginal_relevance(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        lambda_mult=0.7,
        k=settings.retrieval_k,
    )
    docs: list[Document] = []
    for index in selected_indices:
        chunk = candidate_chunks[index]
        doc = _chunk_to_document(chunk)
        _add_retrieval_signal(
            doc,
            query=query,
            channel="vector",
            rank=index + 1,
            distance=float(rows[index][1]),
        )
        docs.append(doc)
    return docs


def query_keyword_chunks(
    query: str,
    settings: Settings,
    *,
    limit: int | None = None,
) -> list[Document]:
    if not settings.database_url.strip():
        raise RuntimeError("DATABASE_URL must be set for keyword retrieval.")

    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(AgentResourceChunk.search_vector, tsquery).label("rank")

    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(
            select(AgentResourceChunk, rank)
            .where(AgentResourceChunk.search_vector.op("@@")(tsquery))
            .order_by(rank.desc(), AgentResourceChunk.chunk_index.asc())
            .limit(limit or settings.keyword_k)
        ).all()

    docs: list[Document] = []
    for rank_index, (chunk, keyword_score) in enumerate(rows, start=1):
        doc = _chunk_to_document(chunk)
        _add_retrieval_signal(
            doc,
            query=query,
            channel="keyword",
            rank=rank_index,
            score=float(keyword_score),
        )
        docs.append(doc)
    return docs


def _normalize_retrieval_queries(queries: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = query.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _retrieve_legacy(queries: list[str], settings: Settings) -> list[Document]:
    result_lists: list[list[Document]] = []
    weights: list[float] = []
    query_weight_divisor = max(len(queries), 1)

    for query in queries:
        result_lists.extend([
            query_vector_chunks(query, settings),
            query_keyword_chunks(query, settings),
        ])
        weights.extend([
            settings.vector_weight / query_weight_divisor,
            settings.keyword_weight / query_weight_divisor,
        ])

    fused = reciprocal_rank_fusion(result_lists, weights, k=settings.rrf_k)
    return fused[: settings.retrieval_fetch_k]


def _retrieve_hybrid_v2(queries: list[str], settings: Settings) -> list[Document]:
    result_lists: list[list[Document]] = []
    weights: list[float] = []
    query_weight_divisor = max(len(queries), 1)

    for query in queries:
        result_lists.extend([
            _query_vector_candidate_chunks(
                query,
                settings,
                limit=settings.retrieval_fetch_k,
            ),
            query_keyword_chunks(
                query,
                settings,
                limit=settings.keyword_fetch_k,
            ),
        ])
        weights.extend([
            settings.vector_weight / query_weight_divisor,
            settings.keyword_weight / query_weight_divisor,
        ])

    fused = reciprocal_rank_fusion(result_lists, weights, k=settings.rrf_k)
    return fused[: settings.retrieval_fetch_k]


def retrieve_documents(queries: list[str], settings: Settings) -> list[Document]:
    normalized_queries = _normalize_retrieval_queries(queries)
    if not normalized_queries:
        return []

    if settings.retrieval_strategy == "legacy":
        return _retrieve_legacy(normalized_queries, settings)
    if settings.retrieval_strategy == "hybrid_v2":
        return _retrieve_hybrid_v2(normalized_queries, settings)
    raise RuntimeError(f"Unsupported retrieval strategy: {settings.retrieval_strategy}")


def rerank_documents(
    query: str, docs: list[Document], settings: Settings
) -> list[Document]:
    if not docs:
        return docs

    global _voyage_client_cache
    if _voyage_client_cache is None:
        _voyage_client_cache = voyageai.Client(api_key=settings.voyage_api_key)

    result = _voyage_client_cache.rerank(
        query=query,
        documents=[doc.page_content for doc in docs],
        model=settings.reranker_model,
        top_k=settings.reranker_top_k,
    )

    reranked: list[Document] = []
    for item in result.results:
        if item.relevance_score >= settings.reranker_score_threshold:
            doc = docs[item.index]
            doc.metadata["relevance_score"] = item.relevance_score
            reranked.append(doc)
    return reranked


def needs_reindex(settings: Settings) -> bool:
    if not settings.database_url.strip():
        return True

    current_manifest = filesystem_resource_manifest(settings.resources_dir)
    try:
        stored_manifest = _database_resource_manifest(settings)
    except SQLAlchemyError:
        return True

    return current_manifest != stored_manifest
