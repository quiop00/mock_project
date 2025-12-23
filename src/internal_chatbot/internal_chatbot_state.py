"""
State definitions for the internal RAG chatbot.

The chatbot keeps track of:
- conversation history
- last user message
- retrieved context and sources
- error messages for routing to an error handler node
"""

import operator
from typing import List, Optional, Sequence, Tuple

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from typing_extensions import Annotated, TypedDict


class ChatbotState(TypedDict):
    """State carried across LangGraph nodes for the internal chatbot."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_message: str
    retrieval_context: Annotated[List[str], operator.add]
    retrieval_sources: Annotated[List[str], operator.add]
    error: Optional[str]


class ChatbotInput(TypedDict):
    """Input schema for each graph run."""

    message: str


def format_docs_as_context(docs: List[Document]) -> Tuple[List[str], List[str]]:
    """Format documents into context strings and source list."""
    contexts: List[str] = []
    sources: List[str] = []

    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source") or "internal_doc"
        page = metadata.get("page")
        position = f" (page {page})" if page is not None else ""
        sources.append(f"{source}{position}")
        contexts.append(f"[{source}{position}] {doc.page_content}")

    return contexts, sources

