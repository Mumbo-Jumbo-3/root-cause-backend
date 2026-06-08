from langchain_core.documents import Document

from health_agent.config import Settings
from health_agent.rag.evaluation import (
    RagEvalCase,
    evaluate_production_context_strategy,
    evaluate_retrieval_strategy,
    pack_final_context,
)


def test_evaluate_retrieval_strategy_reports_hits_and_mrr():
    cases = [
        RagEvalCase(question="magnesium", expected_sources=("magnesium.md",)),
        RagEvalCase(question="thyroid", expected_sources=("thyroid.md",)),
    ]

    def fake_retrieve(queries, settings):
        assert settings.retrieval_strategy == "hybrid_v2"
        if queries == ["magnesium"]:
            return [
                Document(page_content="x", metadata={"source_path": "other.md"}),
                Document(page_content="y", metadata={"source_path": "magnesium.md"}),
            ]
        return [Document(page_content="z", metadata={"source_path": "other.md"})]

    result = evaluate_retrieval_strategy(
        Settings(),
        "hybrid_v2",
        cases=cases,
        retrieve=fake_retrieve,
    )

    assert result["summary"]["cases"] == 2
    assert result["summary"]["hits"] == 1
    assert result["summary"]["hit_rate"] == 0.5
    assert result["summary"]["mrr"] == 0.25
    assert result["summary"]["unique_source_mrr"] == 0.25
    assert result["summary"]["mean_expected_source_coverage"] == 0.5
    assert result["cases"][0]["rank"] == 2
    assert result["cases"][0]["unique_source_rank"] == 2
    assert result["cases"][0]["expected_sources_found"] == ["magnesium.md"]
    assert result["cases"][1]["rank"] is None


def test_evaluate_production_context_uses_refined_queries_and_reranks():
    cases = [
        RagEvalCase(
            question="progesterone thyroid",
            expected_sources=("expected.md",),
            refined_queries=("progesterone thyroid function", "thyroid estrogen"),
        )
    ]
    retrieve_calls = []

    def fake_retrieve(queries, settings):
        assert settings.retrieval_strategy == "hybrid_v2"
        retrieve_calls.append(queries)
        if queries == ["progesterone thyroid"]:
            return [Document(page_content="base", metadata={"source_path": "other.md"})]
        return [
            Document(
                page_content="enriched",
                metadata={"source_path": "expected.md"},
            )
        ]

    def fake_rerank(query, docs, settings):
        assert query == "progesterone thyroid"
        assert [doc.metadata["source_path"] for doc in docs] == [
            "other.md",
            "expected.md",
        ]
        return docs

    result = evaluate_production_context_strategy(
        Settings(),
        "hybrid_v2",
        cases=cases,
        retrieve=fake_retrieve,
        rerank=fake_rerank,
    )

    assert retrieve_calls == [
        ["progesterone thyroid"],
        ["progesterone thyroid function", "thyroid estrogen"],
    ]
    assert result["mode"] == "production-context"
    assert result["summary"]["hit_rate"] == 1.0
    assert result["summary"]["mrr"] == 0.5
    assert result["cases"][0]["rank"] == 2
    assert result["cases"][0]["unique_source_rank"] == 2


def test_pack_final_context_limits_repeated_sources_before_overflow():
    docs = [
        Document(page_content="a1", metadata={"source_path": "a.md"}),
        Document(page_content="a2", metadata={"source_path": "a.md"}),
        Document(page_content="a3", metadata={"source_path": "a.md"}),
        Document(page_content="b1", metadata={"source_path": "b.md"}),
        Document(page_content="c1", metadata={"source_path": "c.md"}),
        Document(page_content="a4", metadata={"source_path": "a.md"}),
    ]

    packed = pack_final_context(docs, target_k=5, max_chunks_per_source=2)

    assert [doc.page_content for doc in packed] == ["a1", "a2", "b1", "c1", "a3"]


def test_evaluate_production_context_can_pack_final_docs():
    cases = [
        RagEvalCase(
            question="magnesium",
            expected_sources=("expected.md",),
            refined_queries=("magnesium sleep",),
        )
    ]

    def fake_retrieve(queries, settings):
        return [Document(page_content=queries[0], metadata={"source_path": "seed.md"})]

    def fake_rerank(query, docs, settings):
        return [
            Document(page_content="a1", metadata={"source_path": "a.md"}),
            Document(page_content="a2", metadata={"source_path": "a.md"}),
            Document(page_content="a3", metadata={"source_path": "a.md"}),
            Document(page_content="expected", metadata={"source_path": "expected.md"}),
        ]

    result = evaluate_production_context_strategy(
        Settings(reranker_top_k=4),
        "hybrid_v2",
        mode="production-packed",
        pack_context=True,
        max_chunks_per_source=2,
        cases=cases,
        retrieve=fake_retrieve,
        rerank=fake_rerank,
    )

    assert result["mode"] == "production-packed"
    assert result["cases"][0]["rank"] == 3
    assert result["cases"][0]["unique_source_rank"] == 2
