from langchain_core.documents import Document
from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    is_womens_health: bool
    trusted_search_response: str
    trusted_refined_queries: list[str]
    trusted_search_status: str
    womens_health_search_response: str
    womens_health_search_status: str
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
