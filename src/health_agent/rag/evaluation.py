from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from langchain_core.documents import Document

from health_agent.config import Settings
from health_agent.rag.retriever import (
    reciprocal_rank_fusion,
    rerank_documents,
    retrieve_documents,
)


EvalMode = Literal["retrieval", "production-context", "production-packed", "oracle-context"]
RetrieveFunc = Callable[[list[str], Settings], list[Document]]
RerankFunc = Callable[[str, list[Document], Settings], list[Document]]


@dataclass(frozen=True)
class RagEvalCase:
    question: str
    expected_sources: tuple[str, ...]
    refined_queries: tuple[str, ...] = ()


DEFAULT_RAG_EVAL_CASES: tuple[RagEvalCase, ...] = (
    RagEvalCase(
        question="How does magnesium relate to sleep and stress?",
        expected_sources=("grimhood_magnesium.md", "grimhood_sleep.md"),
        refined_queries=(
            "magnesium sleep stress",
            "magnesium GABA stress response",
            "magnesium recovery sleep quality",
        ),
    ),
    RagEvalCase(
        question="What does the archive say about progesterone and thyroid function?",
        expected_sources=(
            "peat_progesterone_in_orthomolecular_medicine_ray_peat.md",
            "peat_progesterone_thyroid_cancer.md",
            "peat_thyroid.md",
        ),
        refined_queries=(
            "progesterone thyroid function",
            "thyroid estrogen progesterone liver",
            "thyroid therapy progesterone synthesis",
        ),
    ),
    RagEvalCase(
        question="What are the arguments for coconut oil over unsaturated fats?",
        expected_sources=(
            "peat_coconutoil.md",
            "peat_unsaturated.md",
            "peat_unsaturated2.md",
        ),
        refined_queries=(
            "coconut oil unsaturated fats",
            "PUFA saturated fat coconut oil",
            "unsaturated oils thyroid metabolism",
        ),
    ),
    RagEvalCase(
        question="What links serotonin with stress, depression, and endotoxin?",
        expected_sources=("peat_serotonin.md", "peat_endotoxin_stress_depression.md"),
        refined_queries=(
            "serotonin stress depression endotoxin",
            "endotoxin serotonin inflammation",
            "serotonin learned helplessness stress",
        ),
    ),
    RagEvalCase(
        question="What nutrition considerations matter for pregnancy and iron?",
        expected_sources=("peat_nutrition_for_women_by_ray_peat.md",),
        refined_queries=(
            "pregnancy iron nutrition",
            "pregnancy vitamin E iron",
            "nutrition for women pregnancy minerals",
        ),
    ),
    RagEvalCase(
        question="How are TSH and thyroid misconceptions discussed?",
        expected_sources=("peat_tsh.md", "peat_thyroid_misconceptions.md"),
        refined_queries=(
            "TSH thyroid misconceptions",
            "thyroid blood tests TSH",
            "thyroid therapy confusion TSH",
        ),
    ),
    RagEvalCase(
        question="What does Grimhood recommend for glyphosate detox?",
        expected_sources=("grimhood_glyphosate_detox.md",),
        refined_queries=(
            "glyphosate detox Grimhood",
            "glyphosate gut minerals detox",
            "glyphosate exposure mitigation",
        ),
    ),
    RagEvalCase(
        question="What does the archive say about calcium, skin, and osteoporosis?",
        expected_sources=("peat_osteoporosis_and_the_skin.md", "peat_calcium.md"),
        refined_queries=(
            "calcium skin osteoporosis",
            "osteoporosis skin calcium vitamin D",
            "calcium phosphate osteoporosis",
        ),
    ),
)


def _source_name(doc: Document) -> str:
    return str(doc.metadata.get("source_path") or doc.metadata.get("source") or "")


def _source_rank(docs: Sequence[Document], expected_sources: set[str]) -> int | None:
    for index, doc in enumerate(docs, start=1):
        if _source_name(doc) in expected_sources:
            return index
    return None


def _unique_source_rank(docs: Sequence[Document], expected_sources: set[str]) -> int | None:
    seen: set[str] = set()
    unique_rank = 0
    for doc in docs:
        source = _source_name(doc)
        if not source or source in seen:
            continue
        seen.add(source)
        unique_rank += 1
        if source in expected_sources:
            return unique_rank
    return None


def _unique_sources(docs: Sequence[Document], limit: int | None = None) -> list[str]:
    sources: list[str] = []
    for doc in docs:
        source = _source_name(doc)
        if not source or source in sources:
            continue
        sources.append(source)
        if limit is not None and len(sources) == limit:
            break
    return sources


def _top_sources(docs: Sequence[Document], limit: int = 5) -> list[str]:
    return _unique_sources(docs, limit=limit)


def _score_docs(
    case: RagEvalCase,
    docs: Sequence[Document],
    *,
    latency_ms: float,
) -> dict[str, object]:
    expected_sources = set(case.expected_sources)
    rank = _source_rank(docs, expected_sources)
    unique_source_rank = _unique_source_rank(docs, expected_sources)
    found_sources = sorted(expected_sources.intersection(_unique_sources(docs)))
    reciprocal_rank = 0.0 if rank is None else 1.0 / rank
    unique_source_reciprocal_rank = (
        0.0 if unique_source_rank is None else 1.0 / unique_source_rank
    )

    return {
        "question": case.question,
        "refined_queries": list(case.refined_queries),
        "expected_sources": list(case.expected_sources),
        "expected_sources_found": found_sources,
        "expected_source_coverage": (
            0.0 if not expected_sources else len(found_sources) / len(expected_sources)
        ),
        "hit": rank is not None,
        "rank": rank,
        "reciprocal_rank": reciprocal_rank,
        "unique_source_rank": unique_source_rank,
        "unique_source_reciprocal_rank": unique_source_reciprocal_rank,
        "latency_ms": latency_ms,
        "top_sources": _top_sources(docs),
    }


def _summarize(case_results: Sequence[dict[str, object]]) -> dict[str, object]:
    total = len(case_results)
    hits = sum(1 for result in case_results if result["hit"])
    reciprocal_rank_sum = sum(float(result["reciprocal_rank"]) for result in case_results)
    unique_source_rr_sum = sum(
        float(result["unique_source_reciprocal_rank"]) for result in case_results
    )
    coverage_sum = sum(
        float(result["expected_source_coverage"]) for result in case_results
    )
    latency_sum = sum(float(result["latency_ms"]) for result in case_results)

    return {
        "cases": total,
        "hits": hits,
        "hit_rate": 0.0 if total == 0 else hits / total,
        "mrr": 0.0 if total == 0 else reciprocal_rank_sum / total,
        "unique_source_mrr": 0.0 if total == 0 else unique_source_rr_sum / total,
        "mean_expected_source_coverage": 0.0 if total == 0 else coverage_sum / total,
        "mean_latency_ms": 0.0 if total == 0 else latency_sum / total,
    }


def _retrieve_production_context(
    case: RagEvalCase,
    settings: Settings,
    *,
    retrieve: RetrieveFunc,
    rerank: RerankFunc,
) -> list[Document]:
    base_docs = retrieve([case.question], settings)
    enrich_docs = retrieve(list(case.refined_queries), settings) if case.refined_queries else []
    merged_docs = reciprocal_rank_fusion(
        [base_docs, enrich_docs],
        [1.0, 1.0],
        k=settings.rrf_k,
    )
    return rerank(case.question, merged_docs, settings)


def pack_final_context(
    docs: Sequence[Document],
    *,
    target_k: int,
    max_chunks_per_source: int = 3,
) -> list[Document]:
    if target_k <= 0:
        return []

    packed: list[Document] = []
    overflow: list[Document] = []
    source_counts: dict[str, int] = {}

    for doc in docs:
        source = _source_name(doc)
        count = source_counts.get(source, 0)
        if source and count >= max_chunks_per_source:
            overflow.append(doc)
            continue

        packed.append(doc)
        if source:
            source_counts[source] = count + 1
        if len(packed) == target_k:
            return packed

    for doc in overflow:
        packed.append(doc)
        if len(packed) == target_k:
            break

    return packed


def evaluate_retrieval_strategy(
    settings: Settings,
    strategy: str,
    *,
    cases: Sequence[RagEvalCase] = DEFAULT_RAG_EVAL_CASES,
    retrieve: RetrieveFunc = retrieve_documents,
) -> dict[str, object]:
    strategy_settings = settings.model_copy(update={"retrieval_strategy": strategy})
    case_results: list[dict[str, object]] = []

    for case in cases:
        start = perf_counter()
        docs = retrieve([case.question], strategy_settings)
        latency_ms = (perf_counter() - start) * 1000
        case_results.append(_score_docs(case, docs, latency_ms=latency_ms))

    return {
        "mode": "retrieval",
        "strategy": strategy,
        "summary": _summarize(case_results),
        "cases": case_results,
    }


def evaluate_production_context_strategy(
    settings: Settings,
    strategy: str,
    *,
    mode: EvalMode = "production-context",
    pack_context: bool = False,
    max_chunks_per_source: int = 3,
    cases: Sequence[RagEvalCase] = DEFAULT_RAG_EVAL_CASES,
    retrieve: RetrieveFunc = retrieve_documents,
    rerank: RerankFunc = rerank_documents,
) -> dict[str, object]:
    strategy_settings = settings.model_copy(update={"retrieval_strategy": strategy})
    case_results: list[dict[str, object]] = []

    for case in cases:
        start = perf_counter()
        docs = _retrieve_production_context(
            case,
            strategy_settings,
            retrieve=retrieve,
            rerank=rerank,
        )
        if pack_context:
            docs = pack_final_context(
                docs,
                target_k=settings.reranker_top_k,
                max_chunks_per_source=max_chunks_per_source,
            )
        latency_ms = (perf_counter() - start) * 1000
        case_results.append(_score_docs(case, docs, latency_ms=latency_ms))

    return {
        "mode": mode,
        "strategy": strategy,
        "summary": _summarize(case_results),
        "cases": case_results,
    }


def evaluate_rag_strategy(
    settings: Settings,
    strategy: str,
    *,
    mode: EvalMode = "retrieval",
    cases: Sequence[RagEvalCase] = DEFAULT_RAG_EVAL_CASES,
) -> dict[str, object]:
    if mode == "retrieval":
        return evaluate_retrieval_strategy(settings, strategy, cases=cases)
    if mode == "production-context":
        return evaluate_production_context_strategy(settings, strategy, cases=cases)
    if mode == "production-packed":
        return evaluate_production_context_strategy(
            settings,
            strategy,
            mode="production-packed",
            pack_context=True,
            cases=cases,
        )
    if mode == "oracle-context":
        from health_agent.rag.oracle_eval import evaluate_oracle_context_strategy

        return evaluate_oracle_context_strategy(settings, strategy)
    raise ValueError(f"Unsupported eval mode: {mode}")


def evaluate_rag_strategies(
    settings: Settings,
    strategies: Sequence[str],
    *,
    mode: EvalMode = "retrieval",
    cases: Sequence[RagEvalCase] = DEFAULT_RAG_EVAL_CASES,
) -> list[dict[str, object]]:
    return [
        evaluate_rag_strategy(settings, strategy, mode=mode, cases=cases)
        for strategy in strategies
    ]


def evaluate_retrieval_strategies(
    settings: Settings,
    strategies: Sequence[str],
    *,
    cases: Sequence[RagEvalCase] = DEFAULT_RAG_EVAL_CASES,
) -> list[dict[str, object]]:
    return evaluate_rag_strategies(
        settings,
        strategies,
        mode="retrieval",
        cases=cases,
    )
