from langchain_core.documents import Document
from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from health_agent.trace import merge_trace


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # The latest message, rewritten to stand alone when it is a follow-up; what research uses.
    question: str
    is_womens_health: bool
    trusted_search_findings: list[dict[str, object]]
    trusted_search_response: str
    trusted_refined_queries: list[str]
    trusted_search_status: str
    womens_health_search_findings: list[dict[str, object]]
    womens_health_search_response: str
    womens_health_search_status: str
    unrestricted_search_findings: list[dict[str, object]]
    unrestricted_search_response: str
    unrestricted_search_status: str
    base_rag_docs: list[Document]
    base_rag_status: str
    enrich_rag_docs: list[Document]
    enrich_rag_status: str
    merged_rag_docs: list[Document]
    rag_status: str
    rag_context: str
    sufficient: bool
    gate_reason: str
    run_started_at: float
    trace: Annotated[list[dict], merge_trace]
