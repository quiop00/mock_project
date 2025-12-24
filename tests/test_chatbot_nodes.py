"""
Unit tests for chatbot nodes with LangSmith tracing.

These tests verify individual node functionality and use LangSmith
for tracing and evaluation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from internal_chatbot.internal_chatbot_state import ChatbotState
from internal_chatbot.internal_chatbot_nodes import (
    receive_user_message,
    retrieve_relevant_docs,
    generate_answer_with_context,
    update_conversation_state,
    error_handler,
    has_error,
)


class TestReceiveUserMessage:
    """Test receive_user_message node."""

    def test_receive_valid_message(self):
        """Test receiving a valid user message."""
        state: ChatbotState = {
            "message": "What is our company policy?",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = receive_user_message(state)
        
        assert "user_message" in result
        assert result["user_message"] == "What is our company policy?"
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert "error" not in result

    def test_receive_message_from_user_message_field(self):
        """Test receiving message from user_message field."""
        state: ChatbotState = {
            "user_message": "Tell me about products",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = receive_user_message(state)
        
        assert result["user_message"] == "Tell me about products"
        assert len(result["messages"]) == 1

    def test_missing_message(self):
        """Test handling missing message."""
        state: ChatbotState = {
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = receive_user_message(state)
        
        assert "error" in result
        assert "Missing user message" in result["error"]


class TestRetrieveRelevantDocs:
    """Test retrieve_relevant_docs node."""

    @patch("internal_chatbot.internal_chatbot_nodes.RETRIEVER")
    def test_retrieve_success(self, mock_retriever):
        """Test successful document retrieval."""
        # Mock retriever response
        mock_docs = [
            Document(
                page_content="Company policy states...",
                metadata={"source": "policy.pdf", "page": 1}
            ),
            Document(
                page_content="Employee handbook...",
                metadata={"source": "handbook.pdf", "page": 2}
            ),
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        state: ChatbotState = {
            "user_message": "What is the company policy?",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = retrieve_relevant_docs(state)
        
        assert "retrieval_context" in result
        assert "retrieval_sources" in result
        assert len(result["retrieval_context"]) == 2
        assert len(result["retrieval_sources"]) == 2
        assert "error" not in result
        mock_retriever.invoke.assert_called_once_with("What is the company policy?")

    @patch("internal_chatbot.internal_chatbot_nodes.RETRIEVER")
    def test_retrieve_failure(self, mock_retriever):
        """Test retrieval failure handling."""
        mock_retriever.invoke = Mock(side_effect=Exception("Connection error"))
        
        state: ChatbotState = {
            "user_message": "Test query",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = retrieve_relevant_docs(state)
        
        assert "error" in result
        assert "Retrieval failed" in result["error"]


class TestGenerateAnswerWithContext:
    """Test generate_answer_with_context node."""

    @pytest.mark.asyncio
    @patch("internal_chatbot.internal_chatbot_nodes._chat_model")
    async def test_generate_with_context(self, mock_model):
        """Test answer generation with retrieved context."""
        mock_response = Mock()
        mock_response.content = "Based on the documents, the policy states..."
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        
        state: ChatbotState = {
            "user_message": "What is the policy?",
            "messages": [HumanMessage(content="What is the policy?")],
            "retrieval_context": [
                "[policy.pdf (page 1)] Company policy states...",
                "[handbook.pdf (page 2)] Employee handbook...",
            ],
            "retrieval_sources": ["policy.pdf (page 1)", "handbook.pdf (page 2)"],
        }
        
        result = await generate_answer_with_context(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "Based on the documents" in result["messages"][0].content
        assert "error" not in result
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_no_context_fallback(self):
        """Test fallback message when no context is retrieved."""
        state: ChatbotState = {
            "user_message": "Random question",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = await generate_answer_with_context(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "I could not find this information" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_generate_with_error_state(self):
        """Test generation skips when error exists."""
        state: ChatbotState = {
            "user_message": "Test",
            "messages": [],
            "retrieval_context": ["Some context"],
            "retrieval_sources": ["source.pdf"],
            "error": "Previous error",
        }
        
        result = await generate_answer_with_context(state)
        
        # Should return empty dict when error exists
        assert result == {}

    @pytest.mark.asyncio
    @patch("internal_chatbot.internal_chatbot_nodes._chat_model")
    async def test_generate_model_error(self, mock_model):
        """Test handling model generation errors."""
        mock_model.ainvoke = AsyncMock(side_effect=Exception("API error"))
        
        state: ChatbotState = {
            "user_message": "Test query",
            "messages": [],
            "retrieval_context": ["Some context"],
            "retrieval_sources": ["source.pdf"],
        }
        
        result = await generate_answer_with_context(state)
        
        assert "error" in result
        assert "Generation failed" in result["error"]


class TestUpdateConversationState:
    """Test update_conversation_state node."""

    def test_update_with_history_limit(self):
        """Test conversation history trimming."""
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(15)
        ]
        
        state: ChatbotState = {
            "user_message": "Latest question",
            "messages": messages,
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = update_conversation_state(state)
        
        assert "messages" in result
        # Should trim to MAX_HISTORY_MESSAGES (default 12)
        assert len(result["messages"]) <= 12

    def test_update_with_short_history(self):
        """Test update with short history that doesn't need trimming."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        
        state: ChatbotState = {
            "user_message": "Test",
            "messages": messages,
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = update_conversation_state(state)
        
        assert len(result["messages"]) == 2


class TestErrorHandler:
    """Test error_handler node."""

    def test_error_handler_with_error(self):
        """Test error handler with specific error."""
        state: ChatbotState = {
            "user_message": "Test",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
            "error": "Retrieval failed: Connection timeout",
        }
        
        result = error_handler(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "Xin lỗi" in result["messages"][0].content
        assert "Connection timeout" in result["messages"][0].content

    def test_error_handler_no_error(self):
        """Test error handler with no error (uses default)."""
        state: ChatbotState = {
            "user_message": "Test",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        result = error_handler(state)
        
        assert "messages" in result
        assert "Unknown error" in result["messages"][0].content


class TestHasError:
    """Test has_error conditional function."""

    def test_has_error_true(self):
        """Test has_error returns True when error exists."""
        state: ChatbotState = {
            "user_message": "Test",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
            "error": "Some error",
        }
        
        assert has_error(state) is True

    def test_has_error_false(self):
        """Test has_error returns False when no error."""
        state: ChatbotState = {
            "user_message": "Test",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
        }
        
        assert has_error(state) is False

    def test_has_error_none(self):
        """Test has_error returns False when error is None."""
        state: ChatbotState = {
            "user_message": "Test",
            "messages": [],
            "retrieval_context": [],
            "retrieval_sources": [],
            "error": None,
        }
        
        assert has_error(state) is False

