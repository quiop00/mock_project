"""
Prompt templates for the internal chatbot.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from internal_chatbot import internal_chatbot_config as cfg

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", cfg.SYSTEM_PROMPT.strip()),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Question: {question}\n\n"
            "Retrieved internal context:\n{context}\n\n"
            "Answer concisely and accurately based only on the context. "
            "If the context is insufficient, use the default fallback.",
        ),
    ]
)

__all__ = ["answer_prompt"]

