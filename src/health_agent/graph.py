import json
import logging
import re
import time

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from health_agent.config import Settings
from health_agent.models import (
    get_claude_classifier_model,
    get_claude_judge_model,
    get_claude_synthesis_model,
    get_trusted_grok_model,
    get_unrestricted_grok_model,
)
from health_agent.rag.retriever import (
    reciprocal_rank_fusion,
    rerank_documents,
    retrieve_documents,
)
from health_agent.state import AgentState
from health_agent.trace import (
    TRACE_RESET,
    TRACE_VERSION,
    archive_document,
    cited_text_and_citations,
    doc_id,
    doc_info,
    docs_detail,
    filter_uncited_urls,
    finding_document,
    finding_items,
    findings_detail,
    grok_cited_urls,
    phase_event,
    post_urls,
)
from health_agent.x_posts import attach_matched_posts


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


def _emit_phase(
    phase: str, status: str, meta: dict | None = None, detail: dict | None = None
) -> None:
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
        writer({"kind": "phase", **phase_event(phase, status, meta, detail)})
    except Exception:
        logger.debug("stream writer dropped phase event", exc_info=True)


def _complete_phase(phase: str, meta: dict, detail: dict | None = None) -> dict:
    """Emit a phase's completed event and return it as an `AgentState.trace` update."""
    _emit_phase(phase, "completed", meta, detail)
    return {"trace": [phase_event(phase, "completed", meta, detail)]}


class SearchFinding(BaseModel):
    claim: str = ""
    source_urls: list[str] = Field(default_factory=list)
    relevance: str = ""

    @field_validator("source_urls", mode="before")
    @classmethod
    def _coerce_source_urls(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [url for url in value if isinstance(url, str)]
        return value


class TrustedSearchAnalysis(BaseModel):
    findings: list[SearchFinding] = Field(default_factory=list)
    refined_queries: list[str] = Field(default_factory=list)


class UnrestrictedSearchAnalysis(BaseModel):
    findings: list[SearchFinding] = Field(default_factory=list)


class WomensHealthSearchAnalysis(BaseModel):
    findings: list[SearchFinding] = Field(default_factory=list)


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


def _coerce_findings_payload(parsed: dict, fallback_relevance: str) -> dict:
    if "findings" not in parsed and parsed.get("initial_response"):
        parsed = {
            **parsed,
            "findings": [
                {
                    "claim": str(parsed["initial_response"]),
                    "source_urls": [],
                    "relevance": fallback_relevance,
                }
            ],
        }
    return parsed


def _fallback_findings(content: str, relevance: str) -> list[SearchFinding]:
    cleaned = _strip_grok_render_tags(content).strip()
    if not cleaned:
        return []
    return [SearchFinding(claim=cleaned, source_urls=[], relevance=relevance)]


def _normalize_findings(findings: list[SearchFinding]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for finding in findings[:10]:
        claim = _strip_grok_render_tags(finding.claim).strip()
        relevance = _strip_grok_render_tags(finding.relevance).strip()
        source_urls: list[str] = []
        for url in finding.source_urls:
            cleaned_url = url.strip()
            if not cleaned_url.startswith(("http://", "https://")):
                continue
            if cleaned_url in source_urls:
                continue
            source_urls.append(cleaned_url)
            if len(source_urls) == 3:
                break

        if claim and relevance:
            normalized.append(
                {
                    "claim": claim,
                    "source_urls": source_urls,
                    "relevance": relevance,
                }
            )
    return normalized


def _findings_status(findings: list[dict[str, object]]) -> str:
    return STATUS_SUCCESS if findings else STATUS_EMPTY


def _format_findings(findings: list[dict[str, object]]) -> str:
    return json.dumps(findings, indent=2)


def _run_search_retrieval(
    queries: list[str],
    settings: Settings,
) -> list[Document]:
    return retrieve_documents(queries, settings)


def _docs_status(docs: list[Document]) -> str:
    return STATUS_SUCCESS if docs else STATUS_EMPTY


def _format_rag_context(docs: list[Document]) -> str:
    if not docs:
        return NO_RAG_RESULTS
    return "\n---\n".join(doc.page_content for doc in docs)


def build_graph(settings: Settings, checkpointer=None):
    trusted_grok = get_trusted_grok_model(settings)
    unrestricted_grok = get_unrestricted_grok_model(settings)
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
- "findings": 3-10 objects grounded in relevant posts from those accounts. Prefer
  3-5 findings unless the question has multiple distinct clinical or practical facets.
  Each finding must include:
  - "claim": a specific, non-duplicative, evidence-grounded point
  - "source_urls": an array of exact X post URLs supporting the claim
  - "relevance": why this finding matters for the user's question
- "refined_queries": 3-4 short natural-language queries (each <=10 words) for a
  wellness RAG library. Each query must focus on ONE distinct angle of the
  user's question (avoid stacking unrelated keywords in a single query) and the
  set should cover different facets rather than paraphrasing each other.

Use [] for "source_urls" if exact post URLs are unavailable. Never infer,
reconstruct, or fabricate URLs. Include at most 3 URLs per finding.

Do not include any text outside the JSON object."""

    unrestricted_search_system = """You are a knowledgeable health and wellness assistant.
Use unrestricted X Search results to answer the user's question.

Return ONLY a JSON object with:
- "findings": 3-10 objects grounded in relevant posts from X Search. Prefer
  3-5 findings unless the question has multiple distinct clinical or practical facets.
  Each finding must include:
  - "claim": a specific, non-duplicative, evidence-grounded point
  - "source_urls": an array of exact X post URLs supporting the claim
  - "relevance": why this finding matters for the user's question

Use [] for "source_urls" if exact post URLs are unavailable. Never infer,
reconstruct, or fabricate URLs. Include at most 3 URLs per finding.

Do not include any text outside the JSON object."""

    womens_health_search_system = f"""You are a knowledgeable women's health assistant.
Use X Search results to answer the user's question, prioritizing these trusted
women's-health accounts: {womens_health_accounts}.

Focus on aspects specific to female physiology, pregnancy, postpartum,
breastfeeding, fertility, menstrual/hormonal cycles, or motherhood as
relevant to the question.

Return ONLY a JSON object with:
- "findings": 3-10 objects grounded in relevant posts from those accounts. Prefer
  3-5 findings unless the question has multiple distinct clinical or practical facets.
  Each finding must include:
  - "claim": a specific, non-duplicative, evidence-grounded point
  - "source_urls": an array of exact X post URLs supporting the claim
  - "relevance": why this finding matters for the user's question

Use [] for "source_urls" if exact post URLs are unavailable. Never infer,
reconstruct, or fabricate URLs. Include at most 3 URLs per finding.

Do not include any text outside the JSON object."""

    classifier_system = """Classify whether the user's question is specifically related to
female physiology, pregnancy, postpartum, breastfeeding, fertility, menstrual or
hormonal cycles, menopause, or motherhood. General health questions that apply to
both sexes are NOT women's-health-specific.

Return ONLY a JSON object: {"is_womens_health": true} or {"is_womens_health": false}.
Do not include any text outside the JSON object."""

    synthesis_system = """You are a knowledgeable health and wellness assistant.
Evidence is attached as documents from up to four channels:
1. A curated research archive (titles name the source work and author)
2. Trusted X accounts
3. Trusted women's-health X accounts (only present for women's-health questions)
4. Broader X search (may be absent)

Each X document holds one finding distilled from posts; its context explains why it
matters. Use findings as evidence, not as prose to repeat verbatim.
Write a comprehensive, practical response that prioritizes those sources in that order.
Ground specific claims in the attached documents so they carry citations. Do not write
your own reference numbers, footnotes, or sources list; citations are attached automatically.
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
- Structured findings from trusted X accounts

Return ONLY a JSON object with:
- "sufficient": boolean — true if the evidence is specific, on-topic, and comprehensive
  enough to form a practical answer; false if key aspects of the question are uncovered,
  evidence is thin, or sources conflict in ways that need external corroboration.
- "reason": one short sentence shown to the user explaining the verdict: what the
  evidence covers well, or which aspect of the question it leaves thin. Plain language;
  don't name internal channels or systems.

Do not include any text outside the JSON object."""

    def classify_womens_health(state: AgentState):
        _emit_phase("classify_womens_health", "started")
        turn_start = {"trace": TRACE_RESET, "run_started_at": time.time()}
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
            return {**turn_start, "is_womens_health": result.is_womens_health}
        except Exception:
            logger.exception("Women's-health classifier failed; defaulting to false")
            _emit_phase(
                "classify_womens_health",
                "completed",
                {"status": STATUS_ERROR, "is_womens_health": False},
            )
            return {**turn_start, "is_womens_health": False}

    def womens_health_grok_search(state: AgentState):
        if not state.get("is_womens_health"):
            return {
                **_complete_phase("womens_health_search", {"status": STATUS_SKIPPED}),
                "womens_health_search_findings": [],
                "womens_health_search_response": "",
                "womens_health_search_status": STATUS_SKIPPED,
            }

        _emit_phase("womens_health_search", "started")
        last_message = state["messages"][-1]
        search_llm = trusted_grok.with_config({"tags": ["nostream"]}).bind(
            tools=[
                {
                    "type": "x_search",
                    "allowed_x_handles": settings.womens_health_x_accounts,
                }
            ]
        )

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
                parsed = {
                    "findings": _fallback_findings(
                        content,
                        "Malformed JSON fallback from women's-health X Search.",
                    )
                }

            parsed = _coerce_findings_payload(
                parsed,
                "Legacy response fallback from women's-health X Search.",
            )
            result = WomensHealthSearchAnalysis(**parsed)
            findings = _normalize_findings(result.findings)
            cited_urls = grok_cited_urls(raw)
            findings = filter_uncited_urls(findings, cited_urls)
            posts = post_urls(cited_urls)
            findings = attach_matched_posts(findings, posts, settings)
            cleaned_response = _format_findings(findings)
            status = _findings_status(findings)
            return {
                **_complete_phase(
                    "womens_health_search",
                    {"status": status, "findings": len(findings), "posts": len(posts)},
                    findings_detail(findings, "x-womens", posts=posts),
                ),
                "womens_health_search_findings": findings,
                "womens_health_search_response": cleaned_response,
                "womens_health_search_status": status,
            }
        except Exception:
            logger.exception("Women's-health Grok search failed")
            return {
                **_complete_phase("womens_health_search", {"status": STATUS_ERROR}),
                "womens_health_search_findings": [],
                "womens_health_search_response": "[]",
                "womens_health_search_status": STATUS_ERROR,
            }

    def trusted_grok_search(state: AgentState):
        _emit_phase("trusted_search", "started")
        last_message = state["messages"][-1]
        search_llm = trusted_grok.with_config({"tags": ["nostream"]}).bind(
            tools=[
                {
                    "type": "x_search",
                    "allowed_x_handles": settings.trusted_x_accounts,
                }
            ]
        )

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
                    "findings": _fallback_findings(
                        content,
                        "Malformed JSON fallback from trusted-account X Search.",
                    ),
                    "refined_queries": [str(last_message.content)],
                }

            parsed = _coerce_findings_payload(
                parsed,
                "Legacy response fallback from trusted-account X Search.",
            )
            result = TrustedSearchAnalysis(**parsed)
            findings = _normalize_findings(result.findings)
            cited_urls = grok_cited_urls(raw)
            findings = filter_uncited_urls(findings, cited_urls)
            posts = post_urls(cited_urls)
            findings = attach_matched_posts(findings, posts, settings)
            cleaned_response = _format_findings(findings)
            refined_queries = _normalize_queries(result.refined_queries, str(last_message.content))
            status = _findings_status(findings)
            return {
                **_complete_phase(
                    "trusted_search",
                    {
                        "status": status,
                        "findings": len(findings),
                        "posts": len(posts),
                        "refined_queries": len(refined_queries),
                    },
                    findings_detail(findings, "x-trusted", refined_queries, posts),
                ),
                "trusted_search_findings": findings,
                "trusted_search_response": cleaned_response,
                "trusted_refined_queries": refined_queries,
                "trusted_search_status": status,
            }
        except Exception:
            logger.exception("Trusted Grok search failed")
            return {
                **_complete_phase("trusted_search", {"status": STATUS_ERROR}),
                "trusted_search_findings": [],
                "trusted_search_response": "[]",
                "trusted_refined_queries": [str(last_message.content)],
                "trusted_search_status": STATUS_ERROR,
            }

    def unrestricted_grok_search(state: AgentState):
        _emit_phase("unrestricted_search", "started")
        last_message = state["messages"][-1]
        search_llm = unrestricted_grok.with_config({"tags": ["nostream"]}).bind(
            tools=[{"type": "x_search"}]
        )

        try:
            raw = search_llm.invoke(
                [SystemMessage(content=unrestricted_search_system), last_message]
            )
            content = _extract_raw_content(raw)
            try:
                parsed = _parse_json_content(content)
            except json.JSONDecodeError:
                logger.warning("Unrestricted Grok returned malformed JSON; using content fallback")
                parsed = {
                    "findings": _fallback_findings(
                        content,
                        "Malformed JSON fallback from unrestricted X Search.",
                    )
                }

            parsed = _coerce_findings_payload(
                parsed,
                "Legacy response fallback from unrestricted X Search.",
            )
            result = UnrestrictedSearchAnalysis(**parsed)
            findings = _normalize_findings(result.findings)
            cited_urls = grok_cited_urls(raw)
            findings = filter_uncited_urls(findings, cited_urls)
            posts = post_urls(cited_urls)
            findings = attach_matched_posts(findings, posts, settings)
            cleaned_response = _format_findings(findings)
            status = _findings_status(findings)
            return {
                **_complete_phase(
                    "unrestricted_search",
                    {"status": status, "findings": len(findings), "posts": len(posts)},
                    findings_detail(findings, "x-broad", posts=posts),
                ),
                "unrestricted_search_findings": findings,
                "unrestricted_search_response": cleaned_response,
                "unrestricted_search_status": status,
            }
        except Exception:
            logger.exception("Unrestricted Grok search failed")
            return {
                **_complete_phase("unrestricted_search", {"status": STATUS_ERROR}),
                "unrestricted_search_findings": [],
                "unrestricted_search_response": "[]",
                "unrestricted_search_status": STATUS_ERROR,
            }

    def rag_retrieve_base(state: AgentState):
        _emit_phase("rag_base", "started")
        original_query = str(state["messages"][-1].content)
        try:
            docs = _run_search_retrieval([original_query], settings)
            status = _docs_status(docs)
            return {
                **_complete_phase(
                    "rag_base", {"status": status, "docs": len(docs)}, docs_detail(docs)
                ),
                "base_rag_docs": docs,
                "base_rag_status": status,
            }
        except Exception:
            logger.exception("Base RAG retrieval failed")
            return {
                **_complete_phase("rag_base", {"status": STATUS_ERROR, "docs": 0}),
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
            return {
                **_complete_phase("rag_enrich", {"status": STATUS_SKIPPED, "docs": 0}),
                "enrich_rag_docs": [],
                "enrich_rag_status": STATUS_EMPTY,
            }

        try:
            docs = _run_search_retrieval(candidate_queries, settings)
            status = _docs_status(docs)
            return {
                **_complete_phase(
                    "rag_enrich",
                    {"status": status, "docs": len(docs), "queries": len(candidate_queries)},
                    docs_detail(
                        docs,
                        queries=candidate_queries,
                        # rag_base ran a superstep earlier, so its event already carries these.
                        sent_ids={doc_id(doc) for doc in state.get("base_rag_docs") or []},
                    ),
                ),
                "enrich_rag_docs": docs,
                "enrich_rag_status": status,
            }
        except Exception:
            logger.exception("Enriched RAG retrieval failed")
            return {
                **_complete_phase("rag_enrich", {"status": STATUS_ERROR, "docs": 0}),
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
                k=settings.rrf_k,
            )
            reranked_docs = rerank_documents(original_query, merged_docs, settings)
            rag_status = _docs_status(reranked_docs)
            base_ids = {doc_id(doc) for doc in state["base_rag_docs"]}
            enrich_ids = {doc_id(doc) for doc in state["enrich_rag_docs"]}
            if rag_status == STATUS_EMPTY and (
                state["base_rag_status"] == STATUS_ERROR
                and state["enrich_rag_status"] == STATUS_ERROR
            ):
                rag_status = STATUS_ERROR

            return {
                **_complete_phase(
                    "rag_merge",
                    {"status": rag_status, "docs": len(reranked_docs)},
                    docs_detail(
                        reranked_docs,
                        base_ids=base_ids,
                        enrich_ids=enrich_ids,
                        # Every merged doc came from base or enrich, whose events carry its text.
                        sent_ids=base_ids | enrich_ids,
                    ),
                ),
                "merged_rag_docs": reranked_docs,
                "rag_status": rag_status,
                "rag_context": _format_rag_context(reranked_docs),
            }
        except Exception:
            logger.exception("RAG merge failed")
            return {
                **_complete_phase("rag_merge", {"status": STATUS_ERROR, "docs": 0}),
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
            reason = "A search step failed, so the search was broadened to cover the gap."
            return {
                **_complete_phase(
                    "gate", {"sufficient": False}, {"sufficient": False, "reason": reason}
                ),
                "sufficient": False,
                "gate_reason": reason,
            }

        user_question = str(state["messages"][-1].content)
        judge_sections = [
            f"## User Question\n{user_question}",
            f"## Trusted X Findings\n{state['trusted_search_response']}",
        ]
        if state.get("womens_health_search_status") == STATUS_SUCCESS:
            judge_sections.append(
                f"## Women's-Health X Findings\n{state['womens_health_search_response']}"
            )
        judge_sections.append(f"## Retrieved Documents\n{state['rag_context']}")
        judge_user = "\n\n".join(judge_sections)
        try:
            raw = judge.invoke(
                [SystemMessage(content=judge_system), HumanMessage(content=judge_user)]
            )
            parsed = _parse_json_content(_extract_raw_content(raw))
            sufficient = bool(parsed.get("sufficient", False))
            reason = str(parsed.get("reason", "")).strip()
            logger.info(
                "Sufficiency judge decided sufficient=%s reason=%s",
                sufficient,
                parsed.get("reason", ""),
            )
        except Exception:
            logger.exception("Sufficiency judge failed; defaulting to insufficient")
            sufficient = False
            reason = ""

        gate_update = {
            **_complete_phase(
                "gate", {"sufficient": sufficient}, {"sufficient": sufficient, "reason": reason}
            ),
            "gate_reason": reason,
        }

        if sufficient:
            return {
                **gate_update,
                "sufficient": True,
                "unrestricted_search_findings": [],
                "unrestricted_search_response": "",
                "unrestricted_search_status": STATUS_SKIPPED,
            }
        return {**gate_update, "sufficient": False}

    def route_from_gate(state: AgentState) -> str:
        return "unrestricted_grok_search" if not state["sufficient"] else "claude_synthesize"

    def claude_synthesize(state: AgentState):
        research_elapsed_ms = int(
            (time.time() - state.get("run_started_at", time.time())) * 1000
        )
        original_question = str(state["messages"][-1].content)
        unrestricted_status = state["unrestricted_search_status"]
        unrestricted_skipped = unrestricted_status == STATUS_SKIPPED
        womens_health_status = state.get("womens_health_search_status", STATUS_SKIPPED)
        womens_health_present = womens_health_status != STATUS_SKIPPED

        # Evidence goes to Claude as citable documents; evidence_ids[i] names
        # document i so citations resolve back to items in the research trace.
        documents: list[dict] = []
        evidence_ids: list[str] = []
        for doc in state["merged_rag_docs"]:
            info = doc_info(doc)
            documents.append(archive_document(info))
            evidence_ids.append(str(info["id"]))
        archive_count = len(documents)
        x_channels = [
            ("Trusted X accounts", "x-trusted", state.get("trusted_search_findings", []))
        ]
        if womens_health_present:
            x_channels.append(
                (
                    "Trusted women's-health X accounts",
                    "x-womens",
                    state.get("womens_health_search_findings", []),
                )
            )
        if not unrestricted_skipped:
            x_channels.append(
                ("Broader X search", "x-broad", state.get("unrestricted_search_findings", []))
            )
        for channel, prefix, findings in x_channels:
            for item in finding_items(findings, prefix):
                documents.append(finding_document(item, channel))
                evidence_ids.append(str(item["id"]))

        _emit_phase("synthesize", "started", detail={"evidence": evidence_ids})

        priority_lines = [
            "1. Curated research archive",
            "2. Trusted X accounts",
        ]
        next_idx = 3
        if womens_health_present:
            priority_lines.append(f"{next_idx}. Trusted women's-health X accounts")
            next_idx += 1
        if not unrestricted_skipped:
            priority_lines.append(f"{next_idx}. Broader X search")

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

        if documents:
            sections.append(
                f"## Evidence\n{archive_count} curated research archive excerpts and "
                f"{len(documents) - archive_count} X findings are attached as documents."
            )
        else:
            sections.append("## Evidence\nNo evidence documents were retrieved.")

        user_content = "\n\n".join(sections)
        human_content = (
            [*documents, {"type": "text", "text": user_content}] if documents else user_content
        )

        messages = [
            SystemMessage(content=synthesis_system),
            *state["messages"][:-1],
            HumanMessage(content=human_content),
        ]

        logger.info(
            "claude_synthesize invoke: trusted=%s womens_health=%s base_rag=%s enrich_rag=%s rag=%s unrestricted=%s documents=%d user_chars=%d",
            state["trusted_search_status"],
            womens_health_status,
            state["base_rag_status"],
            state["enrich_rag_status"],
            state["rag_status"],
            unrestricted_status,
            len(documents),
            len(user_content),
        )

        response = claude.invoke(messages)
        text, citations = cited_text_and_citations(response.content, evidence_ids)
        content_blocks = (
            len(response.content) if isinstance(response.content, list) else None
        )
        logger.info(
            "claude_synthesize response: type=%s content_type=%s blocks=%s chars=%d citations=%d raw_preview=%s",
            type(response).__name__,
            type(response.content).__name__,
            content_blocks,
            len(text),
            len(citations),
            repr(response.content)[:500],
        )

        if not text.strip():
            logger.warning(
                "claude_synthesize returned empty content; emitting fallback. raw=%s",
                repr(response.content)[:2000],
            )
            text, citations = SYNTHESIS_FALLBACK_TEXT, []

        completed = _complete_phase(
            "synthesize", {"status": STATUS_SUCCESS}, {"evidence": evidence_ids}
        )
        research_trace = {
            "version": TRACE_VERSION,
            "elapsed_ms": research_elapsed_ms,
            "events": [*state.get("trace", []), *completed["trace"]],
        }
        # Keep the streamed message's id so LangGraph doesn't re-emit it as new.
        answer = AIMessage(
            content=text,
            id=response.id,
            additional_kwargs={"research_trace": research_trace, "citations": citations},
            response_metadata=response.response_metadata,
            usage_metadata=response.usage_metadata,
        )
        return {"messages": [answer], **completed}

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
