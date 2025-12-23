"""
Internal RAG Chatbot built with LangGraph.

This file only wires the graph; logic is split across:
- internal_chatbot_config.py
- internal_chatbot_retriever.py
- internal_chatbot_prompts.py
- internal_chatbot_nodes.py
"""

from langgraph.graph import StateGraph, START, END

from internal_chatbot.internal_chatbot_state import ChatbotInput, ChatbotState
from internal_chatbot.internal_chatbot_nodes import (
    receive_user_message,
    retrieve_relevant_docs,
    generate_answer_with_context,
    update_conversation_state,
    error_handler,
    has_error,
)

builder = StateGraph(ChatbotState, input_schema=ChatbotInput)

builder.add_node("receive_user_message", receive_user_message)
builder.add_node("retrieve_relevant_docs", retrieve_relevant_docs)
builder.add_node("generate_answer_with_context", generate_answer_with_context)
builder.add_node("update_conversation_state", update_conversation_state)
builder.add_node("error_handler", error_handler)

builder.add_edge(START, "receive_user_message")
builder.add_conditional_edges(
    "receive_user_message",
    has_error,
    {True: "error_handler", False: "retrieve_relevant_docs"},
)
builder.add_conditional_edges(
    "retrieve_relevant_docs",
    has_error,
    {True: "error_handler", False: "generate_answer_with_context"},
)
builder.add_conditional_edges(
    "generate_answer_with_context",
    has_error,
    {True: "error_handler", False: "update_conversation_state"},
)
builder.add_edge("error_handler", END)
builder.add_edge("update_conversation_state", END)

chatbot_graph = builder.compile(
    interrupt_before=[],
)

__all__ = ["chatbot_graph"]

