import pytest
from langchain_core.documents import Document

from health_agent.config import Settings
from health_agent.rag.oracle_eval import (
    OracleCase,
    OracleChunkSupport,
    OracleClaim,
    RagContextEvalCase,
    RagContextOracle,
    evaluate_oracle_context_strategy,
    score_oracle_context,
)


def _doc(content: str, source: str, chunk_index: int, content_hash: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_path": source,
            "chunk_index": chunk_index,
            "content_hash": content_hash,
        },
    )


def _support(source: str, chunk_index: int, content_hash: str, support="direct"):
    return OracleChunkSupport(
        source_path=source,
        chunk_index=chunk_index,
        content_hash=content_hash,
        support=support,
    )


def test_score_oracle_context_measures_claim_support_and_noise():
    case = RagContextEvalCase(id="case", question="question")
    oracle_case = OracleCase(
        case_id="case",
        claims=[
            OracleClaim(
                id="c1",
                claim="Directly supported claim",
                reviewed=True,
                supporting_chunks=[_support("a.md", 0, "ha")],
            ),
            OracleClaim(
                id="c2",
                claim="Partially supported claim",
                reviewed=True,
                min_partial_supports=2,
                supporting_chunks=[
                    _support("b.md", 0, "hb", "partial"),
                    _support("c.md", 0, "hc", "partial"),
                ],
            ),
        ],
    )
    docs = [
        _doc("a", "a.md", 0, "ha"),
        _doc("b", "b.md", 0, "hb"),
        _doc("noise", "noise.md", 0, "hn"),
    ]

    result = score_oracle_context(case, oracle_case, docs, latency_ms=12.0)

    assert result["evaluable"] is True
    assert result["sufficient_context"] is False
    assert result["supported_claims"] == 1
    assert result["required_claims"] == 2
    assert result["required_claim_coverage"] == 0.5
    assert result["first_support_mrr"] == pytest.approx(0.75)
    assert result["gold_context_recall"] == pytest.approx(2 / 3)
    assert result["context_noise_rate"] == pytest.approx(1 / 3)
    assert result["supporting_sources"] == ["a.md", "b.md"]


def test_score_oracle_context_ignores_unreviewed_claims_by_default():
    case = RagContextEvalCase(id="case", question="question")
    oracle_case = OracleCase(
        case_id="case",
        claims=[
            OracleClaim(
                id="c1",
                claim="Draft claim",
                reviewed=False,
                supporting_chunks=[_support("a.md", 0, "ha")],
            )
        ],
    )
    docs = [_doc("a", "a.md", 0, "ha")]

    default_result = score_oracle_context(case, oracle_case, docs, latency_ms=1.0)
    draft_result = score_oracle_context(
        case,
        oracle_case,
        docs,
        latency_ms=1.0,
        include_unreviewed=True,
    )

    assert default_result["evaluable"] is False
    assert draft_result["evaluable"] is True
    assert draft_result["sufficient_context"] is True


def test_evaluate_oracle_context_uses_production_context_flow():
    case = RagContextEvalCase(
        id="case",
        question="progesterone thyroid",
        refined_queries=["progesterone synthesis", "thyroid function"],
    )
    oracle = RagContextOracle(
        cases=[
            OracleCase(
                case_id="case",
                claims=[
                    OracleClaim(
                        id="c1",
                        claim="Expected answer claim",
                        reviewed=True,
                        supporting_chunks=[_support("expected.md", 0, "he")],
                    )
                ],
            )
        ],
    )
    retrieve_calls = []

    def fake_retrieve(queries, settings):
        assert settings.retrieval_strategy == "hybrid_v2"
        retrieve_calls.append(queries)
        if queries == ["progesterone thyroid"]:
            return [_doc("base", "base.md", 0, "hb")]
        return [_doc("expected", "expected.md", 0, "he")]

    def fake_rerank(query, docs, settings):
        assert query == "progesterone thyroid"
        assert [doc.metadata["source_path"] for doc in docs] == [
            "base.md",
            "expected.md",
        ]
        return list(reversed(docs))

    result = evaluate_oracle_context_strategy(
        Settings(),
        "hybrid_v2",
        cases=[case],
        oracle=oracle,
        validate_corpus=False,
        retrieve=fake_retrieve,
        rerank=fake_rerank,
    )

    assert retrieve_calls == [
        ["progesterone thyroid"],
        ["progesterone synthesis", "thyroid function"],
    ]
    assert result["summary"]["sufficient_context_rate"] == 1.0
    assert result["summary"]["mean_required_claim_coverage"] == 1.0
    assert result["cases"][0]["supporting_sources"] == ["expected.md"]


def test_evaluate_oracle_context_requires_evaluable_labels():
    case = RagContextEvalCase(id="case", question="question")

    with pytest.raises(ValueError, match="no evaluable reviewed claims"):
        evaluate_oracle_context_strategy(
            Settings(),
            "legacy",
            cases=[case],
            oracle=RagContextOracle(),
            validate_corpus=False,
            retrieve=lambda queries, settings: [],
            rerank=lambda query, docs, settings: docs,
        )
