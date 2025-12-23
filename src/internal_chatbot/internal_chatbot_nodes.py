"""
LangGraph node implementations for the internal RAG chatbot.
"""

import logging
from typing import Dict, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from internal_chatbot.internal_chatbot_prompts import answer_prompt
from internal_chatbot.internal_chatbot_retriever import RETRIEVER
from internal_chatbot.internal_chatbot_state import ChatbotState, format_docs_as_context
from internal_chatbot import internal_chatbot_config as cfg

logger = logging.getLogger(__name__)

_chat_model = init_chat_model(model="openai:gpt-4o-mini")


@traceable(run_type="chain", metadata={"node": "receive_user_message"})
def receive_user_message(state: ChatbotState, config: Optional[RunnableConfig] = None):
    message = state.get("message") or state.get("user_message")
    if not message:
        logger.warning("Missing user message in state")
        return {"error": "Missing user message."}

    logger.info("Received user message", extra={"message": message})
    return {
        "user_message": message,
        "messages": [HumanMessage(content=message)],
    }


@traceable(run_type="chain", metadata={"node": "retrieve_relevant_docs"})
def retrieve_relevant_docs(
    state: ChatbotState, config: Optional[RunnableConfig] = None
) -> Dict:
    try:
        results = RETRIEVER.get_relevant_documents(state["user_message"])
        contexts, sources = format_docs_as_context(results)
        logger.info(
            "Retrieved documents",
            extra={
                "query": state["user_message"],
                "num_results": len(results),
                "sources": sources,
            },
        )
        return {
            "retrieval_context": contexts,
            "retrieval_sources": sources,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed", extra={"query": state.get("user_message")})
        return {"error": f"Retrieval failed: {exc}"}


@traceable(run_type="chain", metadata={"node": "generate_answer_with_context"})
async def generate_answer_with_context(
    state: ChatbotState, config: Optional[RunnableConfig] = None
) -> Dict:
    # Provide default LangSmith tracing metadata/tags when none supplied.
    config = config or {
        "run_name": "internal_chatbot_answer",
        "tags": ["internal_chatbot", "rag"],
        "metadata": {"graph": "internal_chatbot"},
    }
    if state.get("error"):
        return {}

    contexts = state.get("retrieval_context", [])
    if not contexts:
        fallback = (
            "I could not find this information in the internal documents. "
            "Would you like me to help in another way?"
        )
        return {"messages": [AIMessage(content=fallback)]}

    context_block = "\n\n".join(contexts)

    logger.info(
        "Generating answer",
        extra={
            "query": state["user_message"],
            "num_context_chunks": len(contexts),
            "sources": state.get("retrieval_sources", []),
            "history_len": len(state.get("messages", [])),
        },
    )

    prompt_messages = answer_prompt.format_messages(
        question=state["user_message"],
        context=context_block,
        chat_history=state.get("messages", []),
    )

    try:
        response = await _chat_model.ainvoke(prompt_messages, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Generation failed",
            extra={
                "query": state.get("user_message"),
                "num_context_chunks": len(contexts),
            },
        )
        return {"error": f"Generation failed: {exc}"}

    return {"messages": [AIMessage(content=response.content)]}


@traceable(run_type="chain", metadata={"node": "update_conversation_state"})
def update_conversation_state(
    state: ChatbotState, config: Optional[RunnableConfig] = None
) -> Dict:
    history = state.get("messages", [])
    trimmed = history[-cfg.MAX_HISTORY_MESSAGES:] if cfg.MAX_HISTORY_MESSAGES else history
    logger.info(
        "Updated conversation history",
        extra={"history_len": len(trimmed), "max_history": cfg.MAX_HISTORY_MESSAGES},
    )
    return {"messages": list(trimmed)}


@traceable(run_type="chain", metadata={"node": "error_handler"})
def error_handler(state: ChatbotState, config: Optional[RunnableConfig] = None) -> Dict:
    err = state.get("error") or "Unknown error"
    logger.error("Handling error", extra={"error": err})
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Xin lỗi, có lỗi khi xử lý yêu cầu: {err}. "
                    "Bạn có thể thử lại sau."
                )
            )
        ]
    }


def has_error(state: ChatbotState) -> bool:
    return bool(state.get("error"))


__all__ = [
    "receive_user_message",
    "retrieve_relevant_docs",
    "generate_answer_with_context",
    "update_conversation_state",
    "error_handler",
    "has_error",
]

