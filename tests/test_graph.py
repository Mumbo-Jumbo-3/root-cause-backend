import json
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from health_agent.config import Settings
from health_agent.graph import (
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_SUCCESS,
    SYNTHESIS_FALLBACK_TEXT,
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


def _build_graph(*, trusted=None, unrestricted=None, claude=None):
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
    ):
        graph = build_graph(settings)
    return graph


def _get_node_func(node_name, *, trusted=None, unrestricted=None, claude=None):
    graph = _build_graph(trusted=trusted, unrestricted=unrestricted, claude=claude)
    return graph.get_graph().nodes[node_name].data.func


def test_graph_compiles():
    graph = _build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = _build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert "trusted_grok_search" in node_names
    assert "unrestricted_grok_search" in node_names
    assert "rag_retrieve_base" in node_names
    assert "rag_retrieve_enrich" in node_names
    assert "rag_merge" in node_names
    assert "claude_synthesize" in node_names


def test_graph_has_expected_topology():
    graph = _build_graph()
    graph_data = graph.get_graph()

    start_targets = {e.target for e in graph_data.edges if e.source == "__start__"}
    assert "trusted_grok_search" in start_targets
    assert "unrestricted_grok_search" in start_targets
    assert "rag_retrieve_base" in start_targets

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

    synthesize_sources = {e.source for e in graph_data.edges if e.target == "claude_synthesize"}
    assert "trusted_grok_search" in synthesize_sources
    assert "unrestricted_grok_search" in synthesize_sources
    assert "rag_merge" in synthesize_sources

    end_sources = {e.source for e in graph_data.edges if e.target == "__end__"}
    assert "claude_synthesize" in end_sources


def test_trusted_grok_search_parses_json_and_sets_status():
    content = json.dumps(
        {
            "initial_response": "Magnesium helps sleep.",
            "refined_queries": ["magnesium sleep", "magnesium recovery"],
        }
    )
    trusted_model = _make_search_model(content)
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert result["trusted_search_response"] == "Magnesium helps sleep."
    assert result["trusted_refined_queries"] == ["magnesium sleep", "magnesium recovery"]
    assert result["trusted_search_status"] == STATUS_SUCCESS


def test_trusted_grok_search_malformed_json_degrades_gracefully():
    trusted_model = _make_search_model("Not valid JSON but still helpful.")
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    result = fn({"messages": [HumanMessage(content="benefits of magnesium")]})

    assert result["trusted_search_response"] == "Not valid JSON but still helpful."
    assert result["trusted_refined_queries"] == ["benefits of magnesium"]
    assert result["trusted_search_status"] == STATUS_SUCCESS


def test_trusted_grok_search_strips_render_tags():
    content = json.dumps(
        {
            "initial_response": (
                'Magnesium helps <grok:render type="cite">@hubermanlab</grok:render> sleep.'
            ),
            "refined_queries": ["q1"],
        }
    )
    trusted_model = _make_search_model(content)
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert "<grok:render" not in result["trusted_search_response"]
    assert result["trusted_search_status"] == STATUS_SUCCESS


def test_unrestricted_grok_search_parses_json_and_sets_status():
    content = json.dumps({"initial_response": "Broader X discussion mentions magnesium."})
    unrestricted_model = _make_search_model(content)
    fn = _get_node_func("unrestricted_grok_search", unrestricted=unrestricted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert result["unrestricted_search_response"] == "Broader X discussion mentions magnesium."
    assert result["unrestricted_search_status"] == STATUS_SUCCESS


def test_unrestricted_grok_search_error_sets_error_status():
    unrestricted_model = _make_search_model(error=RuntimeError("boom"))
    fn = _get_node_func("unrestricted_grok_search", unrestricted=unrestricted_model)

    result = fn({"messages": [HumanMessage(content="test")]})

    assert result["unrestricted_search_status"] == STATUS_ERROR
    assert "failed" in result["unrestricted_search_response"].lower()


def test_trusted_grok_search_binds_allowed_handles():
    captured = {}
    trusted_model = _make_search_model(
        json.dumps({"initial_response": "ok", "refined_queries": ["q1"]}),
        capture_bind=captured,
    )
    fn = _get_node_func("trusted_grok_search", trusted=trusted_model)

    fn({"messages": [HumanMessage(content="test")]})

    assert captured["kwargs"]["tools"][0]["type"] == "x_search"
    assert "allowed_x_handles" in captured["kwargs"]["tools"][0]


def test_unrestricted_grok_search_has_no_account_filter():
    captured = {}
    unrestricted_model = _make_search_model(
        json.dumps({"initial_response": "ok"}),
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
    assert "## Trusted X Analysis" in content
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
