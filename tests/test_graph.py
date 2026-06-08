import json
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from health_agent.config import Settings
from health_agent.graph import (
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    SYNTHESIS_FALLBACK_TEXT,
    SearchFinding,
    _coerce_findings_payload,
    _normalize_findings,
    _normalize_queries,
    build_graph,
)


def _make_search_model(content=None, *, error: Exception | None = None, capture_bind=None):
    model = MagicMock()
    configured = MagicMock()
    bound = MagicMock()

    def fake_with_config(cfg):
        return configured

    def fake_bind(**kwargs):
        if capture_bind is not None:
            capture_bind["kwargs"] = kwargs
        return bound

    def fake_invoke(messages, **kwargs):
        if error is not None:
            raise error
        return AIMessage(content=content)

    model.with_config = fake_with_config
    configured.bind = fake_bind
    bound.invoke = fake_invoke
    return model


def _make_claude_model(captured=None):
    model = MagicMock()

    def fake_invoke(messages, **kwargs):
        if captured is not None:
            captured["messages"] = messages
        return AIMessage(content="final answer")

    model.invoke = fake_invoke
    return model


def _make_json_model(payload):
    model = MagicMock()
    model.invoke = lambda messages, **kwargs: AIMessage(content=json.dumps(payload))
    return model


def _make_tracked_json_model(payload):
    """Like `_make_json_model` but `model.invoke` is itself a MagicMock so tests
    can assert on `.call_count` (e.g. to prove a gate short-circuit didn't call
    the judge)."""
    model = MagicMock()
    model.invoke = MagicMock(return_value=AIMessage(content=json.dumps(payload)))
    return model


def _finding(
    claim="Magnesium can support sleep quality.",
    source_urls=None,
    relevance="This directly addresses the user's question.",
):
    return {
        "claim": claim,
        "source_urls": (
            ["https://x.com/example/status/123"]
            if source_urls is None
            else source_urls
        ),
        "relevance": relevance,
    }


def _build_graph(*, trusted=None, unrestricted=None, claude=None, judge=None, classifier=None):
    settings = Settings(
        voyage_api_key="test-voyage-key",
        anthropic_api_key="test-anthropic-key",
        xai_api_key="test-xai-key",
    )
    with (
        patch("health_agent.graph.get_trusted_grok_model", return_value=trusted or MagicMock()),
        patch(
            "health_agent.graph.get_unrestricted_grok_model",
            return_value=unrestricted or MagicMock(),
        ),
        patch("health_agent.graph.get_claude_synthesis_model", return_value=claude or MagicMock()),
        patch(
            "health_agent.graph.get_claude_judge_model",
            return_value=judge or _make_json_model({"sufficient": False, "reason": "test"}),
        ),
        patch(
            "health_agent.graph.get_claude_classifier_model",
            return_value=classifier or _make_json_model({"is_womens_health": False}),
        ),
    ):
        graph = build_graph(settings)
    return graph


def _get_node_func(
    node_name,
    *,
    trusted=None,
    unrestricted=None,
    claude=None,
    judge=None,
    classifier=None,
):
    graph = _build_graph(
        trusted=trusted,
        unrestricted=unrestricted,
        claude=claude,
        judge=judge,
        classifier=classifier,
    )
    return graph.get_graph().nodes[node_name].data.func


def test_graph_compiles():
    graph = _build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = _build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert "classify_womens_health" in node_names
    assert "trusted_grok_search" in node_names
    assert "womens_health_grok_search" in node_names
    assert "unrestricted_grok_search" in node_names
    assert "rag_retrieve_base" in node_names
    assert "rag_retrieve_enrich" in node_names
    assert "rag_merge" in node_names
    assert "sufficiency_gate" in node_names
    assert "claude_synthesize" in node_names


def test_graph_has_expected_topology():
    graph = _build_graph()
    graph_data = graph.get_graph()

    start_targets = {e.target for e in graph_data.edges if e.source == "__start__"}
    assert start_targets == {"classify_womens_health"}

    classify_targets = {
        e.target for e in graph_data.edges if e.source == "classify_womens_health"
    }
    assert "trusted_grok_search" in classify_targets
    assert "womens_health_grok_search" in classify_targets
    assert "rag_retrieve_base" in classify_targets

    assert (
        "rag_retrieve_enrich",
        "trusted_grok_search",
    ) in {(e.target, e.source) for e in graph_data.edges}
    assert ("rag_merge", "rag_retrieve_base") in {
        (e.target, e.source) for e in graph_data.edges
    }
    assert ("rag_merge", "rag_retrieve_enrich") in {
        (e.target, e.source) for e in graph_data.edges
    }

    gate_sources = {e.source for e in graph_data.edges if e.target == "sufficiency_gate"}
    assert "trusted_grok_search" in gate_sources
    assert "womens_health_grok_search" in gate_sources
    assert "rag_merge" in gate_sources

    synthesize_sources = {
        e.source for e in graph_data.edges if e.target == "claude_synthesize"
    }
    assert "unrestricted_grok_search" in synthesize_sources

    end_sources = {e.source for e in graph_data.edges if e.target == "__end__"}
    assert "claude_synthesize" in end_sources


def test_trusted_grok_search_parses_json_and_sets_status():
    content = json.dumps(
        {
            "findings": [_finding()],
            "refined_queries": ["magnesium sleep", "magnesium recovery"],
        }
    )
    trusted_model = _make_search_model(content)
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert result["trusted_search_findings"] == [_finding()]
    assert json.loads(result["trusted_search_response"]) == [_finding()]
    assert result["trusted_refined_queries"] == ["magnesium sleep", "magnesium recovery"]
    assert result["trusted_search_status"] == STATUS_SUCCESS


def test_trusted_grok_search_malformed_json_degrades_gracefully():
    trusted_model = _make_search_model("Not valid JSON but still helpful.")
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    result = fn({"messages": [HumanMessage(content="benefits of magnesium")]})

    findings = json.loads(result["trusted_search_response"])
    assert findings[0]["claim"] == "Not valid JSON but still helpful."
    assert findings[0]["source_urls"] == []
    assert result["trusted_refined_queries"] == ["benefits of magnesium"]
    assert result["trusted_search_status"] == STATUS_SUCCESS


def test_trusted_grok_search_strips_render_tags():
    content = json.dumps(
        {
            "findings": [
                _finding(
                    claim=(
                        'Magnesium helps <grok:render type="cite">@hubermanlab'
                        "</grok:render> sleep."
                    ),
                    source_urls=[
                        "https://x.com/a/status/1",
                        "notaurl",
                        "https://x.com/a/status/1",
                        "https://x.com/b/status/2",
                        "https://x.com/c/status/3",
                        "https://x.com/d/status/4",
                    ],
                )
            ],
            "refined_queries": ["q1"],
        }
    )
    trusted_model = _make_search_model(content)
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert "<grok:render" not in result["trusted_search_response"]
    assert result["trusted_search_findings"][0]["source_urls"] == [
        "https://x.com/a/status/1",
        "https://x.com/b/status/2",
        "https://x.com/c/status/3",
    ]
    assert result["trusted_search_status"] == STATUS_SUCCESS


def test_unrestricted_grok_search_parses_json_and_sets_status():
    content = json.dumps(
        {
            "findings": [
                _finding(
                    claim="Broader X discussion mentions magnesium.",
                    source_urls=[],
                )
            ]
        }
    )
    unrestricted_model = _make_search_model(content)
    fn = _get_node_func("unrestricted_grok_search", unrestricted=unrestricted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert result["unrestricted_search_findings"] == [
        _finding(claim="Broader X discussion mentions magnesium.", source_urls=[])
    ]
    assert result["unrestricted_search_status"] == STATUS_SUCCESS


def test_womens_health_grok_search_parses_json_and_sets_status():
    content = json.dumps(
        {
            "findings": [
                _finding(
                    claim="Thyroid medication often needs closer monitoring in pregnancy.",
                    source_urls=["https://x.com/iam_preethi/status/123"],
                    relevance="The user asked about Hashimoto's while trying to conceive.",
                )
            ]
        }
    )
    trusted_model = _make_search_model(content)
    fn = _get_node_func("womens_health_grok_search", trusted=trusted_model)

    result = fn(
        {
            "messages": [HumanMessage(content="Hashimoto's and pregnancy")],
            "is_womens_health": True,
        }
    )

    assert result["womens_health_search_findings"] == [
        _finding(
            claim="Thyroid medication often needs closer monitoring in pregnancy.",
            source_urls=["https://x.com/iam_preethi/status/123"],
            relevance="The user asked about Hashimoto's while trying to conceive.",
        )
    ]
    assert result["womens_health_search_status"] == STATUS_SUCCESS


def test_unrestricted_grok_search_error_sets_error_status():
    unrestricted_model = _make_search_model(error=RuntimeError("boom"))
    fn = _get_node_func("unrestricted_grok_search", unrestricted=unrestricted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert result["unrestricted_search_findings"] == []
    assert result["unrestricted_search_response"] == "[]"
    assert result["unrestricted_search_status"] == STATUS_ERROR


def test_trusted_grok_search_binds_allowed_handles():
    captured = {}
    trusted_model = _make_search_model(
        json.dumps({"findings": [_finding(claim="ok")], "refined_queries": ["q1"]}),
        capture_bind=captured,
    )
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    fn({"messages": [HumanMessage(content="test")]})

    assert captured["kwargs"]["tools"][0]["type"] == "x_search"
    assert "allowed_x_handles" in captured["kwargs"]["tools"][0]


def test_unrestricted_grok_search_has_no_account_filter():
    captured = {}
    unrestricted_model = _make_search_model(
        json.dumps({"findings": [_finding(claim="ok")]}),
        capture_bind=captured,
    )
    fn = _get_node_func("unrestricted_grok_search", unrestricted=unrestricted_model)

    fn({"messages": [HumanMessage(content="test")]})

    assert captured["kwargs"]["tools"][0]["type"] == "x_search"
    assert "allowed_x_handles" not in captured["kwargs"]["tools"][0]


def test_rag_retrieve_base_uses_original_query_only():
    fn = _get_node_func("rag_retrieve_base")

    with patch("health_agent.graph._run_search_retrieval", return_value=[]) as mock_retrieve:
        result = fn({"messages": [HumanMessage(content="benefits of magnesium")]})

    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args[0][0] == ["benefits of magnesium"]
    assert result["base_rag_status"] == STATUS_EMPTY


def test_rag_retrieve_enrich_uses_trusted_refined_queries_only():
    fn = _get_node_func("rag_retrieve_enrich")

    with patch("health_agent.graph._run_search_retrieval", return_value=[]) as mock_retrieve:
        result = fn(
            {
                "messages": [HumanMessage(content="benefits of magnesium")],
                "trusted_refined_queries": [
                    "benefits of magnesium",
                    "magnesium sleep",
                    "magnesium recovery",
                ],
            }
        )

    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args[0][0] == ["magnesium sleep", "magnesium recovery"]
    assert result["enrich_rag_status"] == STATUS_EMPTY


def test_rag_merge_dedupes_and_reranks():
    fn = _get_node_func("rag_merge")
    doc = Document(page_content="same doc", metadata={"source": "a"})

    with patch(
        "health_agent.graph.rerank_documents",
        side_effect=lambda query, docs, settings: docs,
    ) as mock_rerank:
        result = fn(
            {
                "messages": [HumanMessage(content="benefits of magnesium")],
                "base_rag_docs": [doc],
                "enrich_rag_docs": [doc],
                "base_rag_status": STATUS_SUCCESS,
                "enrich_rag_status": STATUS_SUCCESS,
            }
        )

    assert len(mock_rerank.call_args[0][1]) == 1
    assert result["rag_status"] == STATUS_SUCCESS
    assert result["rag_context"] == "same doc"


def test_claude_synthesize_includes_question_and_statuses():
    captured = {}
    claude_model = _make_claude_model(captured)
    fn = _get_node_func("claude_synthesize", claude=claude_model)

    fn(
        {
            "messages": [HumanMessage(content="What are the benefits of magnesium?")],
            "trusted_search_response": "Trusted analysis",
            "trusted_refined_queries": ["magnesium sleep"],
            "trusted_search_status": STATUS_SUCCESS,
            "unrestricted_search_response": "Broader X analysis",
            "unrestricted_search_status": STATUS_EMPTY,
            "base_rag_docs": [],
            "base_rag_status": STATUS_SUCCESS,
            "enrich_rag_docs": [],
            "enrich_rag_status": STATUS_EMPTY,
            "merged_rag_docs": [],
            "rag_status": STATUS_SUCCESS,
            "rag_context": "Resource context",
        }
    )

    human_messages = [m for m in captured["messages"] if isinstance(m, HumanMessage)]
    content = human_messages[-1].content

    assert "What are the benefits of magnesium?" in content
    assert "RAG system over a curated research archive" in content
    assert "Trusted X Search: success" in content
    assert "Unrestricted X Search: empty" in content
    assert "## Trusted X Findings" in content
    assert "## Retrieved Documents" in content


def test_claude_synthesize_falls_back_on_empty_string(caplog):
    claude_model = MagicMock()
    claude_model.invoke = lambda messages, **kwargs: AIMessage(content="")
    fn = _get_node_func("claude_synthesize", claude=claude_model)

    with caplog.at_level("WARNING", logger="health_agent.graph"):
        result = fn(
            {
                "messages": [HumanMessage(content="How do I sleep better?")],
                "trusted_search_response": "trusted",
                "trusted_refined_queries": [],
                "trusted_search_status": STATUS_SUCCESS,
                "unrestricted_search_response": "",
                "unrestricted_search_status": "skipped",
                "base_rag_docs": [],
                "base_rag_status": STATUS_SUCCESS,
                "enrich_rag_docs": [],
                "enrich_rag_status": STATUS_EMPTY,
                "merged_rag_docs": [],
                "rag_status": STATUS_SUCCESS,
                "rag_context": "rag",
            }
        )

    assert result["messages"][-1].content == SYNTHESIS_FALLBACK_TEXT
    assert any("empty content" in rec.message for rec in caplog.records)


def test_claude_synthesize_falls_back_on_tool_only_blocks(caplog):
    claude_model = MagicMock()
    claude_model.invoke = lambda messages, **kwargs: AIMessage(
        content=[{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]
    )
    fn = _get_node_func("claude_synthesize", claude=claude_model)

    with caplog.at_level("WARNING", logger="health_agent.graph"):
        result = fn(
            {
                "messages": [HumanMessage(content="Why do I feel tired?")],
                "trusted_search_response": "trusted",
                "trusted_refined_queries": [],
                "trusted_search_status": STATUS_SUCCESS,
                "unrestricted_search_response": "",
                "unrestricted_search_status": "skipped",
                "base_rag_docs": [],
                "base_rag_status": STATUS_SUCCESS,
                "enrich_rag_docs": [],
                "enrich_rag_status": STATUS_EMPTY,
                "merged_rag_docs": [],
                "rag_status": STATUS_SUCCESS,
                "rag_context": "rag",
            }
        )

    assert result["messages"][-1].content == SYNTHESIS_FALLBACK_TEXT


def test_graph_invokes_claude_even_if_trusted_search_fails():
    trusted_model = _make_search_model(error=RuntimeError("trusted boom"))
    unrestricted_model = _make_search_model(json.dumps({"initial_response": "Broader X analysis"}))
    claude_model = _make_claude_model()
    graph = _build_graph(
        trusted=trusted_model,
        unrestricted=unrestricted_model,
        claude=claude_model,
    )

    with (
        patch("health_agent.graph._run_search_retrieval", return_value=[]),
        patch("health_agent.graph.rerank_documents", side_effect=lambda query, docs, settings: docs),
    ):
        result = graph.invoke({"messages": [HumanMessage(content="benefits of magnesium")]})

    assert result["messages"][-1].content == "final answer"


# ---------------------------------------------------------------------------
# Group B: pure helper unit tests (no graph build, no LLM)
# ---------------------------------------------------------------------------


def test_normalize_findings_drops_non_http_urls():
    findings = [
        SearchFinding(
            claim="ok",
            source_urls=[
                "https://x.com/a/status/1",
                "notaurl",
                "ftp://x.com/a/status/2",
                "example.com/no-scheme",
            ],
            relevance="why",
        )
    ]
    normalized = _normalize_findings(findings)
    assert normalized[0]["source_urls"] == ["https://x.com/a/status/1"]


def test_normalize_findings_caps_urls_at_three_per_finding_and_dedupes():
    findings = [
        SearchFinding(
            claim="ok",
            source_urls=[
                "https://x.com/a/1",
                "https://x.com/a/1",  # duplicate
                "https://x.com/b/2",
                "https://x.com/c/3",
                "https://x.com/d/4",  # over cap
            ],
            relevance="why",
        )
    ]
    normalized = _normalize_findings(findings)
    assert normalized[0]["source_urls"] == [
        "https://x.com/a/1",
        "https://x.com/b/2",
        "https://x.com/c/3",
    ]


def test_normalize_findings_caps_total_findings_at_ten():
    findings = [
        SearchFinding(claim=f"claim {i}", source_urls=[], relevance="why")
        for i in range(15)
    ]
    normalized = _normalize_findings(findings)
    assert len(normalized) == 10
    assert normalized[0]["claim"] == "claim 0"
    assert normalized[-1]["claim"] == "claim 9"


def test_normalize_findings_drops_empty_claim_or_relevance():
    findings = [
        SearchFinding(claim="", source_urls=[], relevance="why"),
        SearchFinding(claim="ok", source_urls=[], relevance=""),
        SearchFinding(claim="kept", source_urls=[], relevance="why"),
    ]
    normalized = _normalize_findings(findings)
    assert [f["claim"] for f in normalized] == ["kept"]


def test_normalize_findings_strips_grok_render_tags_from_text_fields():
    findings = [
        SearchFinding(
            claim='Magnesium helps <grok:render type="cite">@hubermanlab</grok:render> sleep.',
            source_urls=[],
            relevance='See <grok:render type="cite">@foundmyfitness</grok:render> on minerals.',
        )
    ]
    normalized = _normalize_findings(findings)
    assert "<grok:render" not in normalized[0]["claim"]
    assert "<grok:render" not in normalized[0]["relevance"]


def test_normalize_queries_dedupes_case_insensitively_and_drops_blanks():
    result = _normalize_queries(
        ["Magnesium sleep", "magnesium sleep", "  ", "magnesium recovery"],
        original_query="benefits of magnesium",
    )
    assert result == ["Magnesium sleep", "magnesium recovery"]


def test_normalize_queries_falls_back_to_original_when_all_empty():
    result = _normalize_queries(["", "   ", ""], original_query="benefits of magnesium")
    assert result == ["benefits of magnesium"]


def test_coerce_findings_payload_lifts_legacy_initial_response():
    parsed = {"initial_response": "Magnesium supports sleep."}
    coerced = _coerce_findings_payload(parsed, fallback_relevance="legacy")
    assert coerced["findings"][0]["claim"] == "Magnesium supports sleep."
    assert coerced["findings"][0]["source_urls"] == []
    assert coerced["findings"][0]["relevance"] == "legacy"


def test_coerce_findings_payload_passes_through_when_findings_present():
    parsed = {"findings": [{"claim": "kept", "source_urls": [], "relevance": "why"}]}
    coerced = _coerce_findings_payload(parsed, fallback_relevance="unused")
    assert coerced == parsed


# ---------------------------------------------------------------------------
# Group A: control-flow / routing invariants (mocked LLMs, no network)
# ---------------------------------------------------------------------------


def test_womens_health_search_skips_when_flag_false():
    # Model would raise if invoked — node should short-circuit before reaching it.
    trusted_model = _make_search_model(error=AssertionError("WH model must not be invoked"))
    fn = _get_node_func("womens_health_grok_search", trusted=trusted_model)

    result = fn(
        {
            "messages": [HumanMessage(content="general magnesium question")],
            "is_womens_health": False,
        }
    )

    assert result["womens_health_search_status"] == STATUS_SKIPPED
    assert result["womens_health_search_findings"] == []
    assert result["womens_health_search_response"] == ""


def _base_gate_state(*, trusted_status, base_status, enrich_status):
    return {
        "messages": [HumanMessage(content="benefits of magnesium")],
        "trusted_search_response": "trusted findings",
        "trusted_search_status": trusted_status,
        "base_rag_status": base_status,
        "enrich_rag_status": enrich_status,
        "rag_context": "some retrieved context",
    }


def test_sufficiency_gate_short_circuits_when_trusted_search_errored():
    tracked_judge = _make_tracked_json_model({"sufficient": True, "reason": "should be ignored"})
    fn = _get_node_func("sufficiency_gate", judge=tracked_judge)

    result = fn(
        _base_gate_state(
            trusted_status=STATUS_ERROR,
            base_status=STATUS_SUCCESS,
            enrich_status=STATUS_SUCCESS,
        )
    )

    assert result == {"sufficient": False}
    assert tracked_judge.invoke.call_count == 0


def test_sufficiency_gate_short_circuits_when_both_rag_branches_errored():
    tracked_judge = _make_tracked_json_model({"sufficient": True, "reason": "should be ignored"})
    fn = _get_node_func("sufficiency_gate", judge=tracked_judge)

    result = fn(
        _base_gate_state(
            trusted_status=STATUS_SUCCESS,
            base_status=STATUS_ERROR,
            enrich_status=STATUS_ERROR,
        )
    )

    assert result == {"sufficient": False}
    assert tracked_judge.invoke.call_count == 0


def test_sufficiency_gate_does_not_short_circuit_when_only_one_rag_branch_errored():
    # Single RAG branch erroring is not enough; the judge should still be consulted.
    tracked_judge = _make_tracked_json_model({"sufficient": True, "reason": "ok"})
    fn = _get_node_func("sufficiency_gate", judge=tracked_judge)

    result = fn(
        _base_gate_state(
            trusted_status=STATUS_SUCCESS,
            base_status=STATUS_ERROR,
            enrich_status=STATUS_SUCCESS,
        )
    )

    assert tracked_judge.invoke.call_count == 1
    assert result["sufficient"] is True


def test_sufficiency_gate_sufficient_pre_sets_unrestricted_skip_fields():
    judge = _make_json_model({"sufficient": True, "reason": "evidence covers it"})
    fn = _get_node_func("sufficiency_gate", judge=judge)

    result = fn(
        _base_gate_state(
            trusted_status=STATUS_SUCCESS,
            base_status=STATUS_SUCCESS,
            enrich_status=STATUS_SUCCESS,
        )
    )

    assert result["sufficient"] is True
    assert result["unrestricted_search_findings"] == []
    assert result["unrestricted_search_response"] == ""
    assert result["unrestricted_search_status"] == STATUS_SKIPPED


def test_sufficiency_gate_insufficient_does_not_pre_set_skip_fields():
    judge = _make_json_model({"sufficient": False, "reason": "thin evidence"})
    fn = _get_node_func("sufficiency_gate", judge=judge)

    result = fn(
        _base_gate_state(
            trusted_status=STATUS_SUCCESS,
            base_status=STATUS_SUCCESS,
            enrich_status=STATUS_SUCCESS,
        )
    )

    assert result == {"sufficient": False}


def test_rag_retrieve_enrich_skips_when_only_original_query():
    fn = _get_node_func("rag_retrieve_enrich")

    with patch("health_agent.graph._run_search_retrieval", return_value=[]) as mock_retrieve:
        result = fn(
            {
                "messages": [HumanMessage(content="benefits of magnesium")],
                "trusted_refined_queries": ["benefits of magnesium"],
            }
        )

    mock_retrieve.assert_not_called()
    assert result["enrich_rag_docs"] == []
    assert result["enrich_rag_status"] == STATUS_EMPTY


def test_rag_merge_status_is_error_when_empty_and_both_branches_errored():
    fn = _get_node_func("rag_merge")

    with patch(
        "health_agent.graph.rerank_documents",
        side_effect=lambda query, docs, settings: [],
    ):
        result = fn(
            {
                "messages": [HumanMessage(content="benefits of magnesium")],
                "base_rag_docs": [],
                "enrich_rag_docs": [],
                "base_rag_status": STATUS_ERROR,
                "enrich_rag_status": STATUS_ERROR,
            }
        )

    assert result["merged_rag_docs"] == []
    assert result["rag_status"] == STATUS_ERROR


# ---------------------------------------------------------------------------
# Group C: end-to-end trajectory (graph.invoke with mocked models + RAG)
# ---------------------------------------------------------------------------


_VALID_FINDINGS_PAYLOAD = json.dumps(
    {"findings": [_finding()], "refined_queries": ["magnesium sleep"]}
)


def _patched_rag():
    """Patch the RAG boundary so end-to-end graph invocations stay offline."""
    return (
        patch("health_agent.graph._run_search_retrieval", return_value=[]),
        patch(
            "health_agent.graph.rerank_documents",
            side_effect=lambda query, docs, settings: docs,
        ),
    )


def test_graph_runs_womens_health_branch_when_classifier_says_true():
    classifier = _make_json_model({"is_womens_health": True})
    trusted_model = _make_search_model(_VALID_FINDINGS_PAYLOAD)
    # Unrestricted should be skipped because the judge says sufficient — pass a
    # model that would raise if invoked as a strong negative assertion.
    unrestricted_model = _make_search_model(
        error=AssertionError("unrestricted must be skipped when judge=sufficient")
    )
    judge = _make_json_model({"sufficient": True, "reason": "evidence covers it"})
    claude_model = _make_claude_model()
    graph = _build_graph(
        trusted=trusted_model,
        unrestricted=unrestricted_model,
        claude=claude_model,
        judge=judge,
        classifier=classifier,
    )

    patches = _patched_rag()
    with patches[0], patches[1]:
        result = graph.invoke(
            {"messages": [HumanMessage(content="Hashimoto's and pregnancy")]}
        )

    assert result["is_womens_health"] is True
    assert result["womens_health_search_status"] == STATUS_SUCCESS
    assert result["unrestricted_search_status"] == STATUS_SKIPPED
    assert result["messages"][-1].content == "final answer"


def test_graph_routes_to_unrestricted_when_judge_says_insufficient():
    classifier = _make_json_model({"is_womens_health": False})
    trusted_model = _make_search_model(_VALID_FINDINGS_PAYLOAD)
    unrestricted_model = _make_search_model(
        json.dumps({"findings": [_finding(claim="broader X view", source_urls=[])]})
    )
    judge = _make_json_model({"sufficient": False, "reason": "thin"})
    claude_model = _make_claude_model()
    graph = _build_graph(
        trusted=trusted_model,
        unrestricted=unrestricted_model,
        claude=claude_model,
        judge=judge,
        classifier=classifier,
    )

    patches = _patched_rag()
    with patches[0], patches[1]:
        result = graph.invoke(
            {"messages": [HumanMessage(content="benefits of magnesium")]}
        )

    assert result["is_womens_health"] is False
    assert result["womens_health_search_status"] == STATUS_SKIPPED
    assert result["unrestricted_search_status"] == STATUS_SUCCESS
    assert result["messages"][-1].content == "final answer"


def test_graph_skips_unrestricted_when_judge_says_sufficient():
    classifier = _make_json_model({"is_womens_health": False})
    trusted_model = _make_search_model(_VALID_FINDINGS_PAYLOAD)
    # If sufficient, this must never be reached.
    unrestricted_model = _make_search_model(
        error=AssertionError("unrestricted must be skipped when judge=sufficient")
    )
    judge = _make_json_model({"sufficient": True, "reason": "evidence covers it"})
    claude_model = _make_claude_model()
    graph = _build_graph(
        trusted=trusted_model,
        unrestricted=unrestricted_model,
        claude=claude_model,
        judge=judge,
        classifier=classifier,
    )

    patches = _patched_rag()
    with patches[0], patches[1]:
        result = graph.invoke(
            {"messages": [HumanMessage(content="benefits of magnesium")]}
        )

    assert result["unrestricted_search_status"] == STATUS_SKIPPED
    assert result["unrestricted_search_findings"] == []
    assert result["messages"][-1].content == "final answer"
