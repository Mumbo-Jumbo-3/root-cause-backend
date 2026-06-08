from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import or_, select

from health_agent.config import Settings
from health_agent.db import AgentResourceChunk, get_session_factory
from health_agent.models import get_claude_judge_model
from health_agent.rag.resources import filesystem_resource_manifest
from health_agent.rag.retriever import (
    chunk_to_document,
    needs_reindex,
    query_keyword_chunks,
    query_vector_candidate_chunks,
    reciprocal_rank_fusion,
    rerank_documents,
    retrieve_documents,
)


DEFAULT_CONTEXT_CASES_PATH = Path("evals/rag_context_cases.json")
DEFAULT_CONTEXT_ORACLE_PATH = Path("evals/rag_context_oracle.json")
ORACLE_VERSION = "rag-context-oracle-v1"

RetrieveFunc = Callable[[list[str], Settings], list[Document]]
RerankFunc = Callable[[str, list[Document], Settings], list[Document]]


class RagContextEvalCase(BaseModel):
    id: str
    question: str
    refined_queries: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("refined_queries", "tags", mode="before")
    @classmethod
    def _coerce_string_list(cls, value):
        if value is None:
            return []
        return value


class RagContextCaseFile(BaseModel):
    version: str = "rag-context-cases-v1"
    cases: list[RagContextEvalCase]


class OracleChunkSupport(BaseModel):
    source_path: str
    chunk_index: int
    content_hash: str
    support: Literal["direct", "partial"]
    rationale: str = ""


class OracleClaim(BaseModel):
    id: str
    claim: str
    required: bool = True
    reviewed: bool = False
    min_partial_supports: int = Field(default=2, ge=1)
    supporting_chunks: list[OracleChunkSupport] = Field(default_factory=list)


class OracleCase(BaseModel):
    case_id: str
    claims: list[OracleClaim] = Field(default_factory=list)
    generated_queries: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    generated_at: str = ""


class RagContextOracle(BaseModel):
    version: str = ORACLE_VERSION
    corpus_hash: str = ""
    cases: list[OracleCase] = Field(default_factory=list)


class OracleCandidate(BaseModel):
    candidate_id: str
    source_path: str
    chunk_index: int
    content_hash: str
    title: str = ""
    header_path: str | None = None
    content: str


class OracleJudgeSupport(BaseModel):
    candidate_id: str
    support: Literal["direct", "partial", "none"] = "none"
    rationale: str = ""


class OracleJudgeClaim(BaseModel):
    id: str
    claim: str
    required: bool = True
    min_partial_supports: int = Field(default=2, ge=1)
    supporting_chunks: list[OracleJudgeSupport] = Field(default_factory=list)


class OracleJudgeOutput(BaseModel):
    claims: list[OracleJudgeClaim] = Field(default_factory=list)


def load_context_eval_cases(
    path: Path = DEFAULT_CONTEXT_CASES_PATH,
) -> list[RagContextEvalCase]:
    return RagContextCaseFile.model_validate(_read_json(path)).cases


def load_context_oracle(
    path: Path = DEFAULT_CONTEXT_ORACLE_PATH,
) -> RagContextOracle:
    return RagContextOracle.model_validate(_read_json(path))


def write_context_oracle(oracle: RagContextOracle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(oracle.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def current_corpus_hash(settings: Settings) -> str:
    manifest = filesystem_resource_manifest(settings.resources_dir)
    payload = "\n".join(
        f"{source_path}:{content_hash}"
        for source_path, content_hash in sorted(manifest.items())
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def evaluate_oracle_context_strategy(
    settings: Settings,
    strategy: str,
    *,
    cases: Sequence[RagContextEvalCase] | None = None,
    oracle: RagContextOracle | None = None,
    cases_path: Path = DEFAULT_CONTEXT_CASES_PATH,
    oracle_path: Path = DEFAULT_CONTEXT_ORACLE_PATH,
    include_unreviewed: bool = False,
    validate_corpus: bool = True,
    retrieve: RetrieveFunc = retrieve_documents,
    rerank: RerankFunc = rerank_documents,
) -> dict[str, object]:
    eval_cases = list(cases) if cases is not None else load_context_eval_cases(cases_path)
    eval_oracle = oracle if oracle is not None else load_context_oracle(oracle_path)

    if validate_corpus:
        _ensure_index_current(settings)
        _ensure_oracle_matches_corpus(settings, eval_oracle)

    oracle_cases = {case.case_id: case for case in eval_oracle.cases}
    strategy_settings = settings.model_copy(update={"retrieval_strategy": strategy})
    case_results: list[dict[str, object]] = []

    for case in eval_cases:
        start = perf_counter()
        docs = _retrieve_production_context(case, strategy_settings, retrieve, rerank)
        latency_ms = (perf_counter() - start) * 1000
        case_results.append(
            score_oracle_context(
                case,
                oracle_cases.get(case.id),
                docs,
                latency_ms=latency_ms,
                include_unreviewed=include_unreviewed,
            )
        )

    summary = _summarize_oracle_results(case_results)
    if summary["evaluable_cases"] == 0:
        raise ValueError(
            "Oracle fixture has no evaluable reviewed claims. Run build-rag-oracle, "
            "review the generated labels, or pass --include-unreviewed for draft scoring."
        )

    return {
        "mode": "oracle-context",
        "strategy": strategy,
        "summary": summary,
        "cases": case_results,
    }


def evaluate_oracle_context_strategies(
    settings: Settings,
    strategies: Sequence[str],
    *,
    cases_path: Path = DEFAULT_CONTEXT_CASES_PATH,
    oracle_path: Path = DEFAULT_CONTEXT_ORACLE_PATH,
    include_unreviewed: bool = False,
) -> list[dict[str, object]]:
    cases = load_context_eval_cases(cases_path)
    oracle = load_context_oracle(oracle_path)
    return [
        evaluate_oracle_context_strategy(
            settings,
            strategy,
            cases=cases,
            oracle=oracle,
            include_unreviewed=include_unreviewed,
        )
        for strategy in strategies
    ]


def score_oracle_context(
    case: RagContextEvalCase,
    oracle_case: OracleCase | None,
    docs: Sequence[Document],
    *,
    latency_ms: float,
    include_unreviewed: bool = False,
) -> dict[str, object]:
    claims = _scorable_claims(oracle_case, include_unreviewed=include_unreviewed)
    required_claims = [claim for claim in claims if claim.required]
    returned_keys_by_rank = _returned_chunk_keys_by_rank(docs)
    returned_keys = set(returned_keys_by_rank)
    supporting_keys = {
        _support_key(support)
        for claim in claims
        for support in claim.supporting_chunks
    }
    returned_supporting_keys = returned_keys.intersection(supporting_keys)

    claim_results = [
        _score_claim(claim, returned_keys, returned_keys_by_rank)
        for claim in required_claims
    ]
    supported_claims = sum(1 for result in claim_results if result["supported"])
    required_count = len(required_claims)
    first_support_rr_sum = sum(float(result["first_support_rr"]) for result in claim_results)
    noise_count = sum(
        1 for doc in docs if _document_chunk_key(doc) not in supporting_keys
    )

    return {
        "case_id": case.id,
        "question": case.question,
        "evaluable": required_count > 0,
        "required_claims": required_count,
        "supported_claims": supported_claims,
        "sufficient_context": required_count > 0 and supported_claims == required_count,
        "required_claim_coverage": (
            0.0 if required_count == 0 else supported_claims / required_count
        ),
        "first_support_mrr": (
            0.0 if required_count == 0 else first_support_rr_sum / required_count
        ),
        "gold_context_recall": (
            0.0 if not supporting_keys else len(returned_supporting_keys) / len(supporting_keys)
        ),
        "context_noise_rate": 0.0 if not docs else noise_count / len(docs),
        "latency_ms": latency_ms,
        "claim_results": claim_results,
        "supporting_sources": _supporting_sources(claims, returned_keys),
        "top_sources": _top_sources(docs),
    }


def build_rag_context_oracle(
    settings: Settings,
    *,
    cases: Sequence[RagContextEvalCase] | None = None,
    cases_path: Path = DEFAULT_CONTEXT_CASES_PATH,
    existing_oracle_path: Path = DEFAULT_CONTEXT_ORACLE_PATH,
    case_ids: Sequence[str] | None = None,
    max_candidates: int = 80,
    max_candidate_chars: int = 1000,
    preserve_reviewed: bool = True,
) -> RagContextOracle:
    _ensure_index_current(settings)
    corpus_hash = current_corpus_hash(settings)
    eval_cases = list(cases) if cases is not None else load_context_eval_cases(cases_path)
    if case_ids:
        allowed_case_ids = set(case_ids)
        eval_cases = [case for case in eval_cases if case.id in allowed_case_ids]

    existing_oracle = (
        load_context_oracle(existing_oracle_path)
        if existing_oracle_path.exists()
        else RagContextOracle(corpus_hash=corpus_hash)
    )
    existing_cases = {case.case_id: case for case in existing_oracle.cases}
    oracle_cases: list[OracleCase] = []

    for case in eval_cases:
        existing_case = existing_cases.get(case.id)
        if preserve_reviewed and existing_case and _has_reviewed_claims(existing_case):
            oracle_cases.append(existing_case)
            continue

        generated_queries = generate_oracle_search_queries(case, settings)
        candidates = collect_oracle_candidates(
            case,
            settings,
            generated_queries=generated_queries,
            max_candidates=max_candidates,
        )
        oracle_cases.append(
            label_oracle_case(
                case,
                candidates,
                settings,
                generated_queries=generated_queries,
                max_candidate_chars=max_candidate_chars,
            )
        )

    return RagContextOracle(corpus_hash=corpus_hash, cases=oracle_cases)


def collect_oracle_candidates(
    case: RagContextEvalCase,
    settings: Settings,
    *,
    generated_queries: Sequence[str] = (),
    max_candidates: int = 80,
) -> list[OracleCandidate]:
    queries = _dedupe_strings([case.question, *case.refined_queries, *generated_queries])
    result_lists: list[list[Document]] = []
    weights: list[float] = []

    for query in queries:
        result_lists.extend(
            [
                query_vector_candidate_chunks(query, settings, limit=max_candidates),
                query_keyword_chunks(query, settings, limit=max_candidates),
                query_metadata_chunks(query, settings, limit=max(10, max_candidates // 2)),
            ]
        )
        weights.extend([1.0, 1.0, 0.5])

    for strategy in ("legacy", "hybrid_v2"):
        strategy_settings = settings.model_copy(update={"retrieval_strategy": strategy})
        result_lists.append(retrieve_documents(queries, strategy_settings))
        weights.append(1.0)

    fused_docs = reciprocal_rank_fusion(result_lists, weights, k=settings.rrf_k)
    candidates: list[OracleCandidate] = []
    seen_keys: set[str] = set()
    for doc in fused_docs:
        candidate = _document_to_candidate(doc, len(candidates) + 1)
        key = _candidate_key(candidate)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(candidate)
        if len(candidates) == max_candidates:
            break
    return candidates


def query_metadata_chunks(query: str, settings: Settings, *, limit: int) -> list[Document]:
    terms = _query_terms(query)
    if not terms:
        return []

    clauses = []
    for term in terms:
        pattern = f"%{term}%"
        clauses.extend(
            [
                AgentResourceChunk.source_path.ilike(pattern),
                AgentResourceChunk.title.ilike(pattern),
                AgentResourceChunk.header_path.ilike(pattern),
            ]
        )

    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(
            select(AgentResourceChunk)
            .where(or_(*clauses))
            .order_by(AgentResourceChunk.source_path.asc(), AgentResourceChunk.chunk_index.asc())
            .limit(limit)
        ).scalars()
        return [chunk_to_document(chunk) for chunk in rows]


def generate_oracle_search_queries(
    case: RagContextEvalCase,
    settings: Settings,
    *,
    max_queries: int = 6,
) -> list[str]:
    model = get_claude_judge_model(settings)
    system = """You are preparing a retrieval benchmark over a curated health archive.
Return only JSON: {"queries": ["short query", ...]}.
Generate distinct archive-search queries that would find evidence needed to answer
the user's question. Keep each query under 10 words."""
    user = (
        f"Question: {case.question}\n"
        f"Existing refined queries: {json.dumps(case.refined_queries)}\n"
        f"Return at most {max_queries} additional queries."
    )
    raw = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    parsed = _parse_json_text(_extract_message_text(raw))
    queries = parsed.get("queries", [])
    if not isinstance(queries, list):
        return []
    return _dedupe_strings([str(query) for query in queries])[:max_queries]


def label_oracle_case(
    case: RagContextEvalCase,
    candidates: Sequence[OracleCandidate],
    settings: Settings,
    *,
    generated_queries: Sequence[str] = (),
    max_candidate_chars: int = 1000,
) -> OracleCase:
    model = get_claude_judge_model(settings)
    system = """You build gold labels for a retrieval benchmark.
Given a health question and candidate archive chunks, identify atomic answer claims
that the final RAG context must support. Then label which candidate chunks support
each claim.

Return ONLY JSON with this shape:
{
  "claims": [
    {
      "id": "c1",
      "claim": "specific required answer claim",
      "required": true,
      "min_partial_supports": 2,
      "supporting_chunks": [
        {
          "candidate_id": "C001",
          "support": "direct",
          "rationale": "short reason"
        }
      ]
    }
  ]
}

Use "direct" when the chunk independently supports the claim.
Use "partial" only when multiple chunks are needed together.
Do not include chunks with support "none".
Prefer 3-8 required claims. Do not invent evidence outside the candidates."""
    user = "\n\n".join(
        [
            f"## Question\n{case.question}",
            "## Candidate Chunks\n" + _format_candidates(candidates, max_candidate_chars),
        ]
    )
    raw = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    parsed = OracleJudgeOutput.model_validate(_parse_json_text(_extract_message_text(raw)))
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    claims: list[OracleClaim] = []

    for index, claim in enumerate(parsed.claims, start=1):
        supports: list[OracleChunkSupport] = []
        for support in claim.supporting_chunks:
            if support.support == "none":
                continue
            candidate = candidate_map.get(support.candidate_id)
            if candidate is None:
                continue
            supports.append(
                OracleChunkSupport(
                    source_path=candidate.source_path,
                    chunk_index=candidate.chunk_index,
                    content_hash=candidate.content_hash,
                    support=support.support,
                    rationale=support.rationale,
                )
            )
        claims.append(
            OracleClaim(
                id=claim.id or f"c{index}",
                claim=claim.claim,
                required=claim.required,
                reviewed=False,
                min_partial_supports=claim.min_partial_supports,
                supporting_chunks=supports,
            )
        )

    return OracleCase(
        case_id=case.id,
        claims=claims,
        generated_queries=list(generated_queries),
        candidate_count=len(candidates),
        generated_at=datetime.now(UTC).isoformat(),
    )


def _retrieve_production_context(
    case: RagContextEvalCase,
    settings: Settings,
    retrieve: RetrieveFunc,
    rerank: RerankFunc,
) -> list[Document]:
    base_docs = retrieve([case.question], settings)
    enrich_docs = retrieve(case.refined_queries, settings) if case.refined_queries else []
    merged_docs = reciprocal_rank_fusion(
        [base_docs, enrich_docs],
        [1.0, 1.0],
        k=settings.rrf_k,
    )
    return rerank(case.question, merged_docs, settings)


def _score_claim(
    claim: OracleClaim,
    returned_keys: set[str],
    returned_keys_by_rank: dict[str, int],
) -> dict[str, object]:
    direct_keys = {
        _support_key(support)
        for support in claim.supporting_chunks
        if support.support == "direct"
    }
    partial_keys = {
        _support_key(support)
        for support in claim.supporting_chunks
        if support.support == "partial"
    }
    returned_direct = sorted(direct_keys.intersection(returned_keys))
    returned_partial = sorted(partial_keys.intersection(returned_keys))
    support_keys = direct_keys | partial_keys
    first_rank = min(
        (returned_keys_by_rank[key] for key in support_keys if key in returned_keys_by_rank),
        default=None,
    )
    supported = bool(returned_direct) or len(returned_partial) >= claim.min_partial_supports

    return {
        "claim_id": claim.id,
        "claim": claim.claim,
        "supported": supported,
        "first_support_rank": first_rank,
        "first_support_rr": 0.0 if first_rank is None else 1.0 / first_rank,
        "direct_supports_found": returned_direct,
        "partial_supports_found": returned_partial,
    }


def _summarize_oracle_results(case_results: Sequence[dict[str, object]]) -> dict[str, object]:
    evaluable = [result for result in case_results if result["evaluable"]]
    total = len(evaluable)
    if total == 0:
        return {
            "cases": len(case_results),
            "evaluable_cases": 0,
            "sufficient_context_rate": 0.0,
            "mean_required_claim_coverage": 0.0,
            "mean_first_support_mrr": 0.0,
            "mean_gold_context_recall": 0.0,
            "mean_context_noise_rate": 0.0,
            "mean_latency_ms": 0.0,
        }

    return {
        "cases": len(case_results),
        "evaluable_cases": total,
        "sufficient_context_rate": (
            sum(1 for result in evaluable if result["sufficient_context"]) / total
        ),
        "mean_required_claim_coverage": (
            sum(float(result["required_claim_coverage"]) for result in evaluable) / total
        ),
        "mean_first_support_mrr": (
            sum(float(result["first_support_mrr"]) for result in evaluable) / total
        ),
        "mean_gold_context_recall": (
            sum(float(result["gold_context_recall"]) for result in evaluable) / total
        ),
        "mean_context_noise_rate": (
            sum(float(result["context_noise_rate"]) for result in evaluable) / total
        ),
        "mean_latency_ms": sum(float(result["latency_ms"]) for result in evaluable) / total,
    }


def _scorable_claims(
    oracle_case: OracleCase | None,
    *,
    include_unreviewed: bool,
) -> list[OracleClaim]:
    if oracle_case is None:
        return []
    return [
        claim
        for claim in oracle_case.claims
        if include_unreviewed or claim.reviewed
    ]


def _ensure_index_current(settings: Settings) -> None:
    if needs_reindex(settings):
        raise RuntimeError(
            "RAG index is missing or stale. Run `health-agent ingest` before building "
            "or evaluating the oracle."
        )


def _ensure_oracle_matches_corpus(settings: Settings, oracle: RagContextOracle) -> None:
    corpus_hash = current_corpus_hash(settings)
    if oracle.corpus_hash and oracle.corpus_hash != corpus_hash:
        raise RuntimeError(
            "Oracle corpus_hash does not match the current resources. Rebuild or review "
            "the oracle before scoring retrieval strategies."
        )


def _has_reviewed_claims(oracle_case: OracleCase) -> bool:
    return any(claim.reviewed for claim in oracle_case.claims)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_text(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            return json.loads(match.group())
        raise


def _extract_message_text(raw) -> str:
    if isinstance(raw.content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw.content
            if not (isinstance(block, dict) and block.get("type") == "tool_use")
        ).strip()
    return str(raw.content).strip()


def _format_candidates(
    candidates: Sequence[OracleCandidate],
    max_candidate_chars: int,
) -> str:
    blocks: list[str] = []
    for candidate in candidates:
        content = candidate.content.strip()
        if len(content) > max_candidate_chars:
            content = content[:max_candidate_chars].rstrip() + "\n[truncated]"
        blocks.append(
            "\n".join(
                [
                    f"[{candidate.candidate_id}]",
                    f"source_path: {candidate.source_path}",
                    f"chunk_index: {candidate.chunk_index}",
                    f"title: {candidate.title}",
                    f"header_path: {candidate.header_path or ''}",
                    "content:",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks)


def _document_to_candidate(doc: Document, index: int) -> OracleCandidate:
    source_path = _source_name(doc)
    chunk_index = int(doc.metadata.get("chunk_index", -1))
    content_hash = str(doc.metadata.get("content_hash") or _hash_text(doc.page_content))
    return OracleCandidate(
        candidate_id=f"C{index:03d}",
        source_path=source_path,
        chunk_index=chunk_index,
        content_hash=content_hash,
        title=str(doc.metadata.get("title") or ""),
        header_path=doc.metadata.get("header_path"),
        content=doc.page_content,
    )


def _candidate_key(candidate: OracleCandidate) -> str:
    return _make_chunk_key(
        candidate.source_path,
        candidate.chunk_index,
        candidate.content_hash,
    )


def _document_chunk_key(doc: Document) -> str:
    return _make_chunk_key(
        _source_name(doc),
        int(doc.metadata.get("chunk_index", -1)),
        str(doc.metadata.get("content_hash") or _hash_text(doc.page_content)),
    )


def _support_key(support: OracleChunkSupport) -> str:
    return _make_chunk_key(
        support.source_path,
        support.chunk_index,
        support.content_hash,
    )


def _make_chunk_key(source_path: str, chunk_index: int, content_hash: str) -> str:
    return f"{source_path}:{chunk_index}:{content_hash}"


def _returned_chunk_keys_by_rank(docs: Sequence[Document]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, doc in enumerate(docs, start=1):
        key = _document_chunk_key(doc)
        ranks.setdefault(key, rank)
    return ranks


def _supporting_sources(claims: Sequence[OracleClaim], returned_keys: set[str]) -> list[str]:
    sources: list[str] = []
    for claim in claims:
        for support in claim.supporting_chunks:
            if _support_key(support) not in returned_keys:
                continue
            if support.source_path not in sources:
                sources.append(support.source_path)
    return sources


def _top_sources(docs: Sequence[Document], limit: int = 5) -> list[str]:
    sources: list[str] = []
    for doc in docs:
        source = _source_name(doc)
        if source and source not in sources:
            sources.append(source)
        if len(sources) == limit:
            break
    return sources


def _source_name(doc: Document) -> str:
    return str(doc.metadata.get("source_path") or doc.metadata.get("source") or "")


def _query_terms(query: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[A-Za-z0-9_]{3,}", query.lower())
        if term not in {"the", "and", "for", "with", "what", "does", "are", "how"}
    ][:8]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _hash_text(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()
