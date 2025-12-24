"""
LangSmith evaluation tests for the chatbot.

These tests create evaluation datasets and run evaluations
to measure chatbot performance metrics.
"""

import pytest
from langsmith import Client, traceable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from unittest.mock import Mock, patch, AsyncMock

from internal_chatbot.internal_chatbot import chatbot_graph, builder
from internal_chatbot.internal_chatbot_state import ChatbotInput


class TestLangSmithEvaluation:
    """Test LangSmith evaluation functionality."""

    @pytest.fixture
    def langsmith_client(self):
        """Create LangSmith client for evaluation."""
        try:
            client = Client()
            return client
        except Exception:
            pytest.skip("LangSmith client not available (check LANGSMITH_API_KEY)")

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        with patch("internal_chatbot.internal_chatbot_nodes.RETRIEVER") as mock:
            yield mock

    @pytest.fixture
    def mock_chat_model(self):
        """Create a mock chat model."""
        with patch("internal_chatbot.internal_chatbot_nodes._chat_model") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_evaluate_answer_accuracy(self, mock_retriever, mock_chat_model):
        """Test evaluation of answer accuracy."""
        # Mock retriever with relevant docs
        mock_docs = [
            Document(
                page_content="Our company policy requires all employees to...",
                metadata={"source": "policy.pdf", "page": 1}
            ),
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        # Mock chat model with accurate response
        mock_response = Mock()
        mock_response.content = (
            "According to our company policy, all employees must follow "
            "the guidelines outlined in the employee handbook."
        )
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
        
        # Evaluation criteria
        def accuracy_evaluator(run, example):
            """Evaluate if answer is accurate based on retrieved context."""
            answer = run.outputs.get("messages", [])[-1].content if isinstance(run.outputs, dict) else ""
            context = example.outputs.get("expected_context", "")
            
            # Simple check: answer should reference policy/handbook
            keywords = ["policy", "handbook", "guidelines", "employees"]
            has_keywords = any(keyword.lower() in answer.lower() for keyword in keywords)
            
            return {
                "key": "accuracy",
                "score": 1.0 if has_keywords else 0.0,
                "comment": "Answer contains relevant keywords" if has_keywords else "Answer missing key terms"
            }
        
        # This would be used with LangSmith evaluate() function
        # For now, we just test the evaluator logic
        example = Mock()
        example.outputs = {"expected_context": "policy handbook guidelines"}
        
        run = Mock()
        run.outputs = {
            "messages": [
                AIMessage(content=mock_response.content)
            ]
        }
        
        result = accuracy_evaluator(run, example)
        
        assert result["key"] == "accuracy"
        assert result["score"] == 1.0
        assert "keywords" in result["comment"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_hallucination_prevention(self, mock_retriever, mock_chat_model):
        """Test that chatbot doesn't hallucinate when no context is found."""
        # Mock retriever to return empty (no relevant docs)
        mock_retriever.invoke = Mock(return_value=[])
        
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        
        config = RunnableConfig(
            configurable={"thread_id": "hallucination-test"},
            tags=["evaluation", "hallucination"],
        )
        
        input_data: ChatbotInput = {
            "message": "What is the company's secret formula?"
        }
        
        result = await graph.ainvoke(input_data, config=config)
        
        final_message = result["messages"][-1]
        answer = final_message.content
        
        # Should use fallback message, not make up information
        assert "I could not find this information" in answer
        assert "secret formula" not in answer.lower()  # Shouldn't invent answer

    @pytest.mark.asyncio
    async def test_evaluate_relevance(self, mock_retriever, mock_chat_model):
        """Test evaluation of answer relevance to question."""
        # Mock retriever
        mock_docs = [
            Document(
                page_content="Product A is our flagship solution with features X, Y, Z.",
                metadata={"source": "products.pdf", "page": 1}
            ),
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        # Mock chat model
        mock_response = Mock()
        mock_response.content = "Product A is our flagship solution with features X, Y, and Z."
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
        
        def relevance_evaluator(run, example):
            """Evaluate if answer is relevant to the question."""
            question = example.inputs.get("message", "")
            answer = run.outputs.get("messages", [])[-1].content if isinstance(run.outputs, dict) else ""
            
            # Simple relevance check: answer should address the question topic
            question_topic = "product" if "product" in question.lower() else ""
            answer_relevant = question_topic.lower() in answer.lower() if question_topic else True
            
            return {
                "key": "relevance",
                "score": 1.0 if answer_relevant else 0.0,
                "comment": "Answer addresses question topic" if answer_relevant else "Answer off-topic"
            }
        
        example = Mock()
        example.inputs = {"message": "Tell me about your products"}
        
        run = Mock()
        run.outputs = {
            "messages": [
                AIMessage(content=mock_response.content)
            ]
        }
        
        result = relevance_evaluator(run, example)
        
        assert result["key"] == "relevance"
        assert result["score"] == 1.0

    @traceable(name="chatbot_evaluation_run")
    async def run_chatbot_evaluation(self, question: str, expected_keywords: list):
        """Helper function to run chatbot and evaluate with LangSmith tracing."""
        from langgraph.checkpoint.memory import MemorySaver
        
        with patch("internal_chatbot.internal_chatbot_nodes.RETRIEVER") as mock_retriever, \
             patch("internal_chatbot.internal_chatbot_nodes._chat_model") as mock_chat_model:
            
            mock_docs = [
                Document(
                    page_content="Test content with keywords",
                    metadata={"source": "test.pdf", "page": 1}
                ),
            ]
            mock_retriever.invoke = Mock(return_value=mock_docs)
            
            mock_response = Mock()
            mock_response.content = " ".join(expected_keywords)
            mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
            
            checkpointer = MemorySaver()
            graph = builder.compile(checkpointer=checkpointer)
            
            config = RunnableConfig(
                configurable={"thread_id": "eval-test"},
                tags=["evaluation"],
                metadata={"question": question, "expected_keywords": expected_keywords},
            )
            
            input_data: ChatbotInput = {"message": question}
            result = await graph.ainvoke(input_data, config=config)
            
            return result

    @pytest.mark.asyncio
    async def test_traced_evaluation_run(self):
        """Test evaluation run with LangSmith tracing."""
        result = await self.run_chatbot_evaluation(
            question="What is the policy?",
            expected_keywords=["policy", "guidelines"]
        )
        
        assert "messages" in result
        assert len(result["messages"]) > 0
        # This run will be traced in LangSmith if tracing is enabled

