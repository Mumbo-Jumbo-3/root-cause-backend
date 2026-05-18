import json
import logging
import re
from hashlib import sha256

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from health_agent.config import Settings
from health_agent.models import (
    get_claude_classifier_model,
    get_claude_judge_model,
    get_claude_synthesis_model,
    get_trusted_grok_x_search_model,
    get_unrestricted_grok_x_search_model,
)
from health_agent.rag.retriever import query_keyword_chunks, query_vector_chunks, rerank_documents
from health_agent.state import AgentState


logger = logging.getLogger(__name__)

STATUS_SUCCESS = "success"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"

NO_RAG_RESULTS = "No relevant documents found."

SYNTHESIS_FALLBACK_TEXT = (
    "I wasn't able to assemble a complete answer from the evidence I gathered. "
    "Try rephrasing your question or adding specifics."
)


def _emit_phase(phase: str, status: str, meta: dict | None = None) -> None:
    """Emit a phase lifecycle event to the LangGraph custom stream writer.

    Consumed by server._stream_events and forwarded as SSE `event: phase` to the
    frontend's stage timeline. Silently no-ops outside a streaming context
    (e.g. in unit tests), so node functions can call this unconditionally.
    """
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"kind": "phase", "phase": phase, "status": status, "meta": meta or {}})
    except Exception:
        logger.debug("stream writer dropped phase event", exc_info=True)


def _content_id(doc: Document) -> str:
    """Stable fingerprint including source for correct provenance."""
    source = doc.metadata.get("source", "")
    return sha256(f"{source}:{doc.page_content}".encode()).hexdigest()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    k: int = 60,
) -> list[Document]:
    """Fuse multiple ranked lists using weighted Reciprocal Rank Fusion."""
    doc_scores: dict[str, tuple[float, Document]] = {}
    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results):
            doc_id = _content_id(doc)
            score = weight / (k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + score, doc_scores[doc_id][1])
            else:
                doc_scores[doc_id] = (score, doc)
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs]


class TrustedSearchAnalysis(BaseModel):
    initial_response: str
    refined_queries: list[str]


class UnrestrictedSearchAnalysis(BaseModel):
    initial_response: str


class WomensHealthSearchAnalysis(BaseModel):
    initial_response: str


class WomensHealthClassification(BaseModel):
    is_womens_health: bool


def _strip_grok_render_tags(content: str) -> str:
    return re.sub(r"<grok:render[^>]*>.*?</grok:render>", "", content, flags=re.DOTALL)


def _extract_raw_content(raw: AIMessage) -> str:
    if isinstance(raw.content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw.content
            if not (isinstance(block, dict) and block.get("type") == "tool_use")
        ).strip()
    else:
        content = str(raw.content).strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return _strip_grok_render_tags(content).strip()


def _parse_json_content(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            return json.loads(match.group())
        raise


def _normalize_queries(queries: list[str], original_query: str) -> list[str]:
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

    if not normalized:
        return [original_query]
    return normalized


def _search_status(content: str) -> str:
    return STATUS_SUCCESS if content.strip() else STATUS_EMPTY


def _run_search_retrieval(
    queries: list[str],
    settings: Settings,
) -> list[Document]:
    if not queries:
        return []

    result_lists: list[list[Document]] = []
    weights: list[float] = []
    query_weight_divisor = max(len(queries), 1)

    for query in queries:
        vector_results = query_vector_chunks(query, settings)
        keyword_results = query_keyword_chunks(query, settings)

        result_lists.extend([vector_results, keyword_results])
        weights.extend([
            settings.vector_weight / query_weight_divisor,
            settings.keyword_weight / query_weight_divisor,
        ])

    fused = reciprocal_rank_fusion(result_lists, weights)
    return fused[: settings.retrieval_fetch_k]


def _docs_status(docs: list[Document]) -> str:
    return STATUS_SUCCESS if docs else STATUS_EMPTY


def _format_rag_context(docs: list[Document]) -> str:
    if not docs:
        return NO_RAG_RESULTS
    return "\n---\n".join(doc.page_content for doc in docs)


def build_graph(settings: Settings, checkpointer=None):
    trusted_grok_x_search = get_trusted_grok_x_search_model(
        settings, settings.trusted_x_accounts
    )
    womens_health_grok_x_search = get_trusted_grok_x_search_model(
        settings, settings.womens_health_x_accounts
    )
    unrestricted_grok_x_search = get_unrestricted_grok_x_search_model(settings)
    claude = get_claude_synthesis_model(settings)
    judge = get_claude_judge_model(settings)
    classifier = get_claude_classifier_model(settings)
    accounts = ", ".join(f"@{a}" for a in settings.trusted_x_accounts)
    womens_health_accounts = ", ".join(
        f"@{a}" for a in settings.womens_health_x_accounts
    )

    trusted_search_system = f"""You are a knowledgeable health and wellness assistant.
Use X Search results to answer the user's question while prioritizing these trusted
accounts: {accounts}.

Return ONLY a JSON object with:
- "initial_response": a practical analysis grounded in relevant posts from those accounts
- "refined_queries": 3-4 short natural-language queries (each <=10 words) for a
  wellness RAG library. Each query must focus on ONE distinct angle of the
  user's question (avoid stacking unrelated keywords in a single query) and the
  set should cover different facets rather than paraphrasing each other.

Do not include any text outside the JSON object."""

    unrestricted_search_system = """You are a knowledgeable health and wellness assistant.
Use unrestricted X Search results to answer the user's question.

Return ONLY a JSON object with:
- "initial_response": a practical analysis grounded in relevant posts from X Search

Do not include any text outside the JSON object."""

    womens_health_search_system = f"""You are a knowledgeable women's health assistant.
Use X Search results to answer the user's question, prioritizing these trusted
women's-health accounts: {womens_health_accounts}.

Focus on aspects specific to female physiology, pregnancy, postpartum,
breastfeeding, fertility, menstrual/hormonal cycles, or motherhood as
relevant to the question.

Return ONLY a JSON object with:
- "initial_response": a practical analysis grounded in relevant posts from those accounts

Do not include any text outside the JSON object."""

    classifier_system = """Classify whether the user's question is specifically related to
female physiology, pregnancy, postpartum, breastfeeding, fertility, menstrual or
hormonal cycles, menopause, or motherhood. General health questions that apply to
both sexes are NOT women's-health-specific.

Return ONLY a JSON object: {"is_womens_health": true} or {"is_womens_health": false}.
Do not include any text outside the JSON object."""

    synthesis_system = """You are a knowledgeable health and wellness assistant.
You will receive evidence from up to four channels:
1. A RAG system over a curated research archive
2. Trusted X accounts
3. Trusted women's-health X accounts (only present for women's-health questions)
4. Unrestricted X Search (may be absent)

Write a comprehensive, practical response that prioritizes those sources in that order.
If evidence conflicts, prefer the higher-priority source and briefly explain the conflict.
If a channel is empty or failed, briefly note that its evidence was limited or unavailable.
If a channel was not consulted (women's-health or unrestricted), do not mention it at all.
Keep the response integrated rather than source-separated, but include a brief hierarchy note
that the curated research archive and trusted accounts were weighted above broader X findings.

Disclaimers are provided elsewhere. Do not mention filenames or internal implementation details."""

    judge_system = """You are evaluating whether the provided evidence adequately answers
the user's health question.

Given:
- The user's question
- Retrieved RAG documents from the curated research archive
- Analysis from trusted X accounts

Return ONLY a JSON object with:
- "sufficient": boolean — true if the evidence is specific, on-topic, and comprehensive
  enough to form a practical answer; false if key aspects of the question are uncovered,
  evidence is thin, or sources conflict in ways that need external corroboration.
- "reason": a short sentence (for logs only; not shown to users).

Do not include any text outside the JSON object."""

    def classify_womens_health(state: AgentState):
        _emit_phase("classify_womens_health", "started")
        last_message = state["messages"][-1]
        try:
            raw = classifier.invoke(
                [
                    SystemMessage(content=classifier_system),
                    HumanMessage(content=str(last_message.content)),
                ]
            )
            parsed = _parse_json_content(_extract_raw_content(raw))
            result = WomensHealthClassification(**parsed)
            _emit_phase(
                "classify_womens_health",
                "completed",
                {"status": STATUS_SUCCESS, "is_womens_health": result.is_womens_health},
            )
            return {"is_womens_health": result.is_womens_health}
        except Exception:
            logger.exception("Women's-health classifier failed; defaulting to false")
            _emit_phase(
                "classify_womens_health",
                "completed",
                {"status": STATUS_ERROR, "is_womens_health": False},
            )
            return {"is_womens_health": False}

    def womens_health_grok_search(state: AgentState):
        if not state.get("is_womens_health"):
            _emit_phase(
                "womens_health_search", "completed", {"status": STATUS_SKIPPED}
            )
            return {
                "womens_health_search_response": "",
                "womens_health_search_status": STATUS_SKIPPED,
            }

        _emit_phase("womens_health_search", "started")
        last_message = state["messages"][-1]
        search_llm = womens_health_grok_x_search.with_config({"tags": ["nostream"]})

        try:
            raw = search_llm.invoke(
                [SystemMessage(content=womens_health_search_system), last_message]
            )
            content = _extract_raw_content(raw)
            try:
                parsed = _parse_json_content(content)
            except json.JSONDecodeError:
                logger.warning(
                    "Women's-health Grok returned malformed JSON; using content fallback"
                )
                parsed = {"initial_response": content}

            result = WomensHealthSearchAnalysis(**parsed)
            cleaned_response = _strip_grok_render_tags(result.initial_response).strip()
            status = _search_status(cleaned_response)
            _emit_phase(
                "womens_health_search", "completed", {"status": status}
            )
            return {
                "womens_health_search_response": cleaned_response,
                "womens_health_search_status": status,
            }
        except Exception:
            logger.exception("Women's-health Grok search failed")
            _emit_phase(
                "womens_health_search", "completed", {"status": STATUS_ERROR}
            )
            return {
                "womens_health_search_response": "Women's-health X search failed.",
                "womens_health_search_status": STATUS_ERROR,
            }

    def trusted_grok_search(state: AgentState):
        _emit_phase("trusted_search", "started")
        last_message = state["messages"][-1]
        search_llm = trusted_grok_x_search.with_config({"tags": ["nostream"]})

        try:
            raw = search_llm.invoke(
                [SystemMessage(content=trusted_search_system), last_message]
            )
            content = _extract_raw_content(raw)
            try:
                parsed = _parse_json_content(content)
            except json.JSONDecodeError:
                logger.warning("Trusted Grok returned malformed JSON; using content fallback")
                parsed = {
                    "initial_response": content,
                    "refined_queries": [str(last_message.content)],
                }

            result = TrustedSearchAnalysis(**parsed)
            cleaned_response = _strip_grok_render_tags(result.initial_response).strip()
            refined_queries = _normalize_queries(result.refined_queries, str(last_message.content))
            status = _search_status(cleaned_response)
            _emit_phase(
                "trusted_search",
                "completed",
                {"status": status, "refined_queries": len(refined_queries)},
            )
            return {
                "trusted_search_response": cleaned_response,
                "trusted_refined_queries": refined_queries,
                "trusted_search_status": status,
            }
        except Exception:
            logger.exception("Trusted Grok search failed")
            _emit_phase("trusted_search", "completed", {"status": STATUS_ERROR})
            return {
                "trusted_search_response": "Trusted-account X search failed.",
                "trusted_refined_queries": [str(last_message.content)],
                "trusted_search_status": STATUS_ERROR,
            }

    def unrestricted_grok_search(state: AgentState):
        _emit_phase("unrestricted_search", "started")
        last_message = state["messages"][-1]
        search_llm = unrestricted_grok_x_search.with_config({"tags": ["nostream"]})

        try:
            raw = search_llm.invoke(
                [SystemMessage(content=unrestricted_search_system), last_message]
            )
            content = _extract_raw_content(raw)
            try:
                parsed = _parse_json_content(content)
            except json.JSONDecodeError:
                logger.warning("Unrestricted Grok returned malformed JSON; using content fallback")
                parsed = {"initial_response": content}

            result = UnrestrictedSearchAnalysis(**parsed)
            cleaned_response = _strip_grok_render_tags(result.initial_response).strip()
            status = _search_status(cleaned_response)
            _emit_phase("unrestricted_search", "completed", {"status": status})
            return {
                "unrestricted_search_response": cleaned_response,
                "unrestricted_search_status": status,
            }
        except Exception:
            logger.exception("Unrestricted Grok search failed")
            _emit_phase("unrestricted_search", "completed", {"status": STATUS_ERROR})
            return {
                "unrestricted_search_response": "Unrestricted X search failed.",
                "unrestricted_search_status": STATUS_ERROR,
            }

    def rag_retrieve_base(state: AgentState):
        _emit_phase("rag_base", "started")
        original_query = str(state["messages"][-1].content)
        try:
            docs = _run_search_retrieval([original_query], settings)
            status = _docs_status(docs)
            _emit_phase("rag_base", "completed", {"status": status, "docs": len(docs)})
            return {
                "base_rag_docs": docs,
                "base_rag_status": status,
            }
        except Exception:
            logger.exception("Base RAG retrieval failed")
            _emit_phase("rag_base", "completed", {"status": STATUS_ERROR, "docs": 0})
            return {
                "base_rag_docs": [],
                "base_rag_status": STATUS_ERROR,
            }

    def rag_retrieve_enrich(state: AgentState):
        _emit_phase("rag_enrich", "started")
        original_query = str(state["messages"][-1].content)
        candidate_queries = [
            query
            for query in state["trusted_refined_queries"]
            if query.strip().lower() != original_query.strip().lower()
        ]

        if not candidate_queries:
            _emit_phase(
                "rag_enrich", "completed", {"status": STATUS_SKIPPED, "docs": 0}
            )
            return {
                "enrich_rag_docs": [],
                "enrich_rag_status": STATUS_EMPTY,
            }

        try:
            docs = _run_search_retrieval(candidate_queries, settings)
            status = _docs_status(docs)
            _emit_phase(
                "rag_enrich",
                "completed",
                {"status": status, "docs": len(docs), "queries": len(candidate_queries)},
            )
            return {
                "enrich_rag_docs": docs,
                "enrich_rag_status": status,
            }
        except Exception:
            logger.exception("Enriched RAG retrieval failed")
            _emit_phase("rag_enrich", "completed", {"status": STATUS_ERROR, "docs": 0})
            return {
                "enrich_rag_docs": [],
                "enrich_rag_status": STATUS_ERROR,
            }

    def rag_merge(state: AgentState):
        _emit_phase("rag_merge", "started")
        original_query = str(state["messages"][-1].content)
        try:
            merged_docs = reciprocal_rank_fusion(
                [state["base_rag_docs"], state["enrich_rag_docs"]],
                [1.0, 1.0],
            )
            reranked_docs = rerank_documents(original_query, merged_docs, settings)
            rag_status = _docs_status(reranked_docs)
            if rag_status == STATUS_EMPTY and (
                state["base_rag_status"] == STATUS_ERROR
                and state["enrich_rag_status"] == STATUS_ERROR
            ):
                rag_status = STATUS_ERROR

            _emit_phase(
                "rag_merge",
                "completed",
                {"status": rag_status, "docs": len(reranked_docs)},
            )
            return {
                "merged_rag_docs": reranked_docs,
                "rag_status": rag_status,
                "rag_context": _format_rag_context(reranked_docs),
            }
        except Exception:
            logger.exception("RAG merge failed")
            _emit_phase("rag_merge", "completed", {"status": STATUS_ERROR, "docs": 0})
            return {
                "merged_rag_docs": [],
                "rag_status": STATUS_ERROR,
                "rag_context": NO_RAG_RESULTS,
            }

    def sufficiency_gate(state: AgentState):
        _emit_phase("gate", "started")
        rag_errored = (
            state["base_rag_status"] == STATUS_ERROR
            and state["enrich_rag_status"] == STATUS_ERROR
        )
        if state["trusted_search_status"] == STATUS_ERROR or rag_errored:
            logger.info("Sufficiency gate: upstream ERROR, routing to unrestricted Grok")
            _emit_phase(
                "gate", "completed", {"sufficient": False, "reason": "upstream_error"}
            )
            return {"sufficient": False}

        user_question = str(state["messages"][-1].content)
        judge_sections = [
            f"## User Question\n{user_question}",
            f"## Trusted X Analysis\n{state['trusted_search_response']}",
        ]
        if state.get("womens_health_search_status") == STATUS_SUCCESS:
            judge_sections.append(
                f"## Women's-Health X Analysis\n{state['womens_health_search_response']}"
            )
        judge_sections.append(f"## Retrieved Documents\n{state['rag_context']}")
        judge_user = "\n\n".join(judge_sections)
        try:
            raw = judge.invoke(
                [SystemMessage(content=judge_system), HumanMessage(content=judge_user)]
            )
            parsed = _parse_json_content(_extract_raw_content(raw))
            sufficient = bool(parsed.get("sufficient", False))
            logger.info(
                "Sufficiency judge decided sufficient=%s reason=%s",
                sufficient,
                parsed.get("reason", ""),
            )
        except Exception:
            logger.exception("Sufficiency judge failed; defaulting to insufficient")
            sufficient = False

        _emit_phase("gate", "completed", {"sufficient": sufficient})

        if sufficient:
            return {
                "sufficient": True,
                "unrestricted_search_response": "",
                "unrestricted_search_status": STATUS_SKIPPED,
            }
        return {"sufficient": False}

    def route_from_gate(state: AgentState) -> str:
        return "unrestricted_grok_search" if not state["sufficient"] else "claude_synthesize"

    def claude_synthesize(state: AgentState):
        _emit_phase("synthesize", "started")
        original_question = str(state["messages"][-1].content)
        unrestricted_status = state["unrestricted_search_status"]
        unrestricted_skipped = unrestricted_status == STATUS_SKIPPED
        womens_health_status = state.get("womens_health_search_status", STATUS_SKIPPED)
        womens_health_present = womens_health_status != STATUS_SKIPPED

        priority_lines = [
            "1. RAG system over a curated research archive",
            "2. Trusted X accounts",
        ]
        next_idx = 3
        if womens_health_present:
            priority_lines.append(f"{next_idx}. Trusted women's-health X accounts")
            next_idx += 1
        if not unrestricted_skipped:
            priority_lines.append(f"{next_idx}. Unrestricted X Search")

        sections = [
            f"## User Question\n{original_question}",
            "## Evidence Priority\n" + "\n".join(priority_lines),
        ]

        status_lines = [
            f"- Trusted X Search: {state['trusted_search_status']}",
            f"- Base RAG: {state['base_rag_status']}",
            f"- Enriched RAG: {state['enrich_rag_status']}",
            f"- RAG Aggregate: {state['rag_status']}",
        ]
        if womens_health_present:
            status_lines.insert(
                1, f"- Women's-Health X Search: {womens_health_status}"
            )
        if not unrestricted_skipped:
            status_lines.insert(1, f"- Unrestricted X Search: {unrestricted_status}")
        sections.append("## Branch Status\n" + "\n".join(status_lines))

        sections.append(f"## Trusted X Analysis\n{state['trusted_search_response']}")
        if womens_health_present:
            sections.append(
                f"## Women's-Health X Analysis\n{state['womens_health_search_response']}"
            )
        if not unrestricted_skipped:
            sections.append(
                f"## Unrestricted X Analysis\n{state['unrestricted_search_response']}"
            )
        sections.append(f"## Retrieved Documents\n{state['rag_context']}")

        user_content = "\n\n".join(sections)

        messages = [
            SystemMessage(content=synthesis_system),
            *state["messages"][:-1],
            HumanMessage(content=user_content),
        ]

        logger.info(
            "claude_synthesize invoke: trusted=%s womens_health=%s base_rag=%s enrich_rag=%s rag=%s unrestricted=%s user_chars=%d",
            state["trusted_search_status"],
            womens_health_status,
            state["base_rag_status"],
            state["enrich_rag_status"],
            state["rag_status"],
            unrestricted_status,
            len(user_content),
        )

        response = claude.invoke(messages)
        extracted = _extract_raw_content(response)
        content_blocks = (
            len(response.content) if isinstance(response.content, list) else None
        )
        logger.info(
            "claude_synthesize response: type=%s content_type=%s blocks=%s extracted_chars=%d raw_preview=%s",
            type(response).__name__,
            type(response.content).__name__,
            content_blocks,
            len(extracted),
            repr(response.content)[:500],
        )

        if not extracted.strip():
            logger.warning(
                "claude_synthesize returned empty content; emitting fallback. raw=%s",
                repr(response.content)[:2000],
            )
            response = AIMessage(content=SYNTHESIS_FALLBACK_TEXT)

        _emit_phase("synthesize", "completed", {"status": STATUS_SUCCESS})
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("classify_womens_health", classify_womens_health)
    graph.add_node("trusted_grok_search", trusted_grok_search)
    graph.add_node("womens_health_grok_search", womens_health_grok_search)
    graph.add_node("unrestricted_grok_search", unrestricted_grok_search)
    graph.add_node("rag_retrieve_base", rag_retrieve_base)
    graph.add_node("rag_retrieve_enrich", rag_retrieve_enrich)
    graph.add_node("rag_merge", rag_merge)
    graph.add_node("sufficiency_gate", sufficiency_gate)
    graph.add_node("claude_synthesize", claude_synthesize)

    graph.add_edge(START, "classify_womens_health")
    graph.add_edge("classify_womens_health", "trusted_grok_search")
    graph.add_edge("classify_womens_health", "womens_health_grok_search")
    graph.add_edge("classify_womens_health", "rag_retrieve_base")

    graph.add_edge("trusted_grok_search", "rag_retrieve_enrich")
    graph.add_edge(["rag_retrieve_base", "rag_retrieve_enrich"], "rag_merge")

    graph.add_edge(
        ["trusted_grok_search", "womens_health_grok_search", "rag_merge"],
        "sufficiency_gate",
    )

    graph.add_conditional_edges(
        "sufficiency_gate",
        route_from_gate,
        {
            "unrestricted_grok_search": "unrestricted_grok_search",
            "claude_synthesize": "claude_synthesize",
        },
    )
    graph.add_edge("unrestricted_grok_search", "claude_synthesize")
    graph.add_edge("claude_synthesize", END)

    return graph.compile(checkpointer=checkpointer)
