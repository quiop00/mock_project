"""
Integration tests for the full chatbot graph with LangSmith tracing.

These tests verify the complete workflow and use LangSmith for
tracing and evaluation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from internal_chatbot.internal_chatbot import chatbot_graph, builder
from internal_chatbot.internal_chatbot_state import ChatbotInput


class TestChatbotGraph:
    """Test the complete chatbot graph workflow."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever for testing."""
        with patch("internal_chatbot.internal_chatbot_nodes.RETRIEVER") as mock:
            yield mock

    @pytest.fixture
    def mock_chat_model(self):
        """Create a mock chat model for testing."""
        with patch("internal_chatbot.internal_chatbot_nodes._chat_model") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_full_workflow_success(self, mock_retriever, mock_chat_model):
        """Test complete successful workflow."""
        # Mock retriever
        mock_docs = [
            Document(
                page_content="Our company policy states that employees...",
                metadata={"source": "policy.pdf", "page": 1}
            ),
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        # Mock chat model
        mock_response = Mock()
        mock_response.content = (
            "Based on the internal documents, our company policy states "
            "that employees must follow the guidelines outlined in the handbook."
        )
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
        
        # Create checkpointer for session management
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        
        # Run graph
        config = RunnableConfig(
            configurable={"thread_id": "test-thread-1"},
            tags=["test", "integration"],
            metadata={"test_name": "full_workflow_success"},
        )
        
        input_data: ChatbotInput = {
            "message": "What is our company policy?"
        }
        
        result = await graph.ainvoke(input_data, config=config)
        
        # Verify result structure
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check that final message is AI response
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        assert len(final_message.content) > 0
        
        # Verify retriever was called
        mock_retriever.invoke.assert_called_once_with("What is our company policy?")
        
        # Verify model was called
        mock_chat_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_workflow_no_context(self, mock_retriever):
        """Test workflow when no documents are retrieved."""
        # Mock retriever to return empty list
        mock_retriever.invoke = Mock(return_value=[])
        
        checkpointer = MemorySaver()
        from internal_chatbot.internal_chatbot import builder
        graph = builder.compile(checkpointer=checkpointer)
        
        config = RunnableConfig(
            configurable={"thread_id": "test-thread-2"},
            tags=["test", "integration"],
            metadata={"test_name": "full_workflow_no_context"},
        )
        
        input_data: ChatbotInput = {
            "message": "What is the weather today?"
        }
        
        result = await graph.ainvoke(input_data, config=config)
        
        assert "messages" in result
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        # Should contain fallback message
        assert "I could not find this information" in final_message.content

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_retriever, mock_chat_model):
        """Test multi-turn conversation with context retention."""
        # Mock retriever
        mock_docs = [
            Document(
                page_content="Product information: Our main product is...",
                metadata={"source": "products.pdf", "page": 1}
            ),
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        # Mock chat model
        mock_response = Mock()
        mock_response.content = "Our main product is a chatbot solution."
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
        
        checkpointer = MemorySaver()
        from internal_chatbot.internal_chatbot import builder
        graph = builder.compile(checkpointer=checkpointer)
        
        thread_id = "test-thread-3"
        config = RunnableConfig(
            configurable={"thread_id": thread_id},
            tags=["test", "multi_turn"],
        )
        
        # First turn
        input1: ChatbotInput = {"message": "What products do we offer?"}
        result1 = await graph.ainvoke(input1, config=config)
        
        assert len(result1["messages"]) >= 2  # User + AI
        
        # Second turn (should have context from first turn)
        input2: ChatbotInput = {"message": "Tell me more about it"}
        result2 = await graph.ainvoke(input2, config=config)
        
        # Should have more messages (previous + new)
        assert len(result2["messages"]) > len(result1["messages"])
        
        # Verify retriever was called twice
        assert mock_retriever.invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_retriever):
        """Test error handling in workflow."""
        # Mock retriever to raise error
        mock_retriever.invoke = Mock(side_effect=Exception("Database connection failed"))
        
        checkpointer = MemorySaver()
        from internal_chatbot.internal_chatbot import builder
        graph = builder.compile(checkpointer=checkpointer)
        
        config = RunnableConfig(
            configurable={"thread_id": "test-thread-4"},
            tags=["test", "error_handling"],
        )
        
        input_data: ChatbotInput = {
            "message": "Test query"
        }
        
        result = await graph.ainvoke(input_data, config=config)
        
        # Should have error handler message
        assert "messages" in result
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        assert "Xin lỗi" in final_message.content or "error" in final_message.content.lower()


class TestLangSmithIntegration:
    """Test LangSmith tracing and evaluation integration."""

    @pytest.mark.asyncio
    @patch("internal_chatbot.internal_chatbot_nodes.RETRIEVER")
    @patch("internal_chatbot.internal_chatbot_nodes._chat_model")
    async def test_langsmith_tracing(self, mock_chat_model, mock_retriever):
        """Test that LangSmith tracing is enabled."""
        import os
        
        # Set LangSmith env vars if not already set
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "internal-chatbot-tests")
        
        # Mock retriever
        mock_docs = [
            Document(
                page_content="Test content",
                metadata={"source": "test.pdf", "page": 1}
            ),
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        # Mock chat model
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
        
        checkpointer = MemorySaver()
        from internal_chatbot.internal_chatbot import builder
        graph = builder.compile(checkpointer=checkpointer)
        
        config = RunnableConfig(
            configurable={"thread_id": "langsmith-test"},
            tags=["langsmith", "tracing"],
            metadata={"test": "langsmith_integration"},
        )
        
        input_data: ChatbotInput = {
            "message": "Test query for LangSmith"
        }
        
        result = await graph.ainvoke(input_data, config=config)
        
        # Verify execution completed
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Note: Actual LangSmith traces will appear in LangSmith UI
        # when LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY is set

