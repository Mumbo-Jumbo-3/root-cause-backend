from unittest.mock import patch

from langchain_core.documents import Document

from health_agent.config import Settings
from health_agent.rag.retriever import reciprocal_rank_fusion, retrieve_documents


def _doc(content: str, source: str, retrieval: dict | None = None) -> Document:
    metadata = {"source": source, "source_path": source}
    if retrieval is not None:
        metadata["retrieval"] = retrieval
    return Document(page_content=content, metadata=metadata)


def test_reciprocal_rank_fusion_merges_duplicate_metadata():
    vector_doc = _doc(
        "same chunk",
        "a.md",
        {"queries": ["magnesium"], "vector_rank": 1, "vector_distance": 0.12},
    )
    keyword_doc = _doc(
        "same chunk",
        "a.md",
        {"queries": ["magnesium sleep"], "keyword_rank": 1, "keyword_score": 0.9},
    )

    fused = reciprocal_rank_fusion([[vector_doc], [keyword_doc]], [0.6, 0.4], k=60)

    assert len(fused) == 1
    retrieval = fused[0].metadata["retrieval"]
    assert retrieval["queries"] == ["magnesium", "magnesium sleep"]
    assert retrieval["vector_rank"] == 1
    assert retrieval["vector_distance"] == 0.12
    assert retrieval["keyword_rank"] == 1
    assert retrieval["keyword_score"] == 0.9
    assert retrieval["rrf_score"] == (0.6 / 61) + (0.4 / 61)


def test_hybrid_v2_retrieves_wide_candidates_before_fusion():
    settings = Settings(
        retrieval_strategy="hybrid_v2",
        retrieval_fetch_k=80,
        keyword_fetch_k=70,
    )
    vector_doc = _doc("vector", "vector.md")
    keyword_doc = _doc("keyword", "keyword.md")

    with (
        patch(
            "health_agent.rag.retriever._query_vector_candidate_chunks",
            return_value=[vector_doc],
        ) as vector_search,
        patch(
            "health_agent.rag.retriever.query_keyword_chunks",
            return_value=[keyword_doc],
        ) as keyword_search,
    ):
        docs = retrieve_documents(["magnesium"], settings)

    assert [doc.page_content for doc in docs] == ["vector", "keyword"]
    assert vector_search.call_args.kwargs["limit"] == 80
    assert keyword_search.call_args.kwargs["limit"] == 70


def test_legacy_strategy_uses_existing_vector_mmr_path():
    settings = Settings(retrieval_strategy="legacy")
    vector_doc = _doc("vector", "vector.md")
    keyword_doc = _doc("keyword", "keyword.md")

    with (
        patch(
            "health_agent.rag.retriever.query_vector_chunks",
            return_value=[vector_doc],
        ) as vector_search,
        patch(
            "health_agent.rag.retriever.query_keyword_chunks",
            return_value=[keyword_doc],
        ),
    ):
        retrieve_documents(["magnesium"], settings)

    vector_search.assert_called_once()


def test_retrieve_documents_normalizes_duplicate_queries():
    settings = Settings(retrieval_strategy="hybrid_v2")

    with (
        patch(
            "health_agent.rag.retriever._query_vector_candidate_chunks",
            return_value=[],
        ) as vector_search,
        patch(
            "health_agent.rag.retriever.query_keyword_chunks",
            return_value=[],
        ),
    ):
        retrieve_documents([" magnesium ", "Magnesium", ""], settings)

    vector_search.assert_called_once()
