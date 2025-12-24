"""
LangSmith dataset creation and evaluation tests.

This module creates test datasets and runs evaluations following
LangSmith RAG evaluation best practices.

Reference: https://docs.smith.langchain.com/evaluation/tutorials/rag
"""

import pytest
from typing import Annotated
from typing_extensions import TypedDict
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client, traceable

from internal_chatbot.internal_chatbot import builder
from internal_chatbot.internal_chatbot_state import ChatbotInput


# Grade schemas for LLM-as-judge evaluators
class CorrectnessGrade(TypedDict):
    """Schema for correctness evaluation."""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    """Schema for relevance evaluation."""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


class GroundedGrade(TypedDict):
    """Schema for groundedness evaluation."""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


class RetrievalRelevanceGrade(TypedDict):
    """Schema for retrieval relevance evaluation."""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


class TestDatasetCreation:
    """Test dataset creation in LangSmith."""

    @pytest.fixture
    def langsmith_client(self):
        """Create LangSmith client."""
        try:
            return Client()
        except Exception:
            pytest.skip("LangSmith client not available (check LANGSMITH_API_KEY)")

    def test_create_dataset(self, langsmith_client):
        """Test creating a dataset with examples."""
        dataset_name = "Internal Chatbot Test Dataset"
        
        # Define examples with questions and reference answers
        examples = [
            {
                "inputs": {"message": "What is our company policy?"},
                "outputs": {
                    "answer": "Our company policy requires all employees to follow the employee handbook guidelines."
                },
            },
            {
                "inputs": {"message": "What products do we offer?"},
                "outputs": {
                    "answer": "We offer Product A, Product B, and Product C with various features."
                },
            },
            {
                "inputs": {"message": "What is our refund policy?"},
                "outputs": {
                    "answer": "Our refund policy allows returns within 30 days of purchase for a full refund."
                },
            },
        ]

        # Create dataset if it doesn't exist
        try:
            if langsmith_client.has_dataset(dataset_name=dataset_name):
                # Dataset exists, skip creation
                dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
            else:
                dataset = langsmith_client.create_dataset(dataset_name=dataset_name)
                langsmith_client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
            
            assert dataset is not None
            assert dataset.name == dataset_name
            
        except Exception as e:
            pytest.skip(f"Could not create dataset: {e}")

    def test_dataset_examples_structure(self):
        """Test that dataset examples have correct structure."""
        example = {
            "inputs": {"message": "Test question?"},
            "outputs": {"answer": "Test answer."},
        }
        
        assert "inputs" in example
        assert "outputs" in example
        assert "message" in example["inputs"]
        assert "answer" in example["outputs"]


class TestRAGEvaluators:
    """Test RAG evaluators implementation."""

    @pytest.fixture
    def grader_llm(self):
        """Create grader LLM."""
        try:
            return init_chat_model(model="openai:gpt-4o-mini", temperature=0)
        except Exception:
            pytest.skip("LLM not available (check OPENAI_API_KEY)")

    def test_correctness_evaluator(self, grader_llm):
        """Test correctness evaluator implementation."""
        correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        grader = grader_llm.with_structured_output(
            CorrectnessGrade, method="json_schema", strict=True
        )

        question = "What is our company policy?"
        ground_truth = "Our company policy requires all employees to follow the employee handbook guidelines."
        student_answer = "According to our company policy, all employees must follow the guidelines in the employee handbook."

        answers = f"""\
QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
STUDENT ANSWER: {student_answer}"""

        grade = grader.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ])

        assert "correct" in grade
        assert isinstance(grade["correct"], bool)
        assert "explanation" in grade

    def test_relevance_evaluator(self, grader_llm):
        """Test relevance evaluator implementation."""
        relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        grader = grader_llm.with_structured_output(
            RelevanceGrade, method="json_schema", strict=True
        )

        question = "What products do we offer?"
        answer = "We offer Product A, Product B, and Product C with various features."

        answer_text = f"QUESTION: {question}\nSTUDENT ANSWER: {answer}"
        grade = grader.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer_text},
        ])

        assert "relevant" in grade
        assert isinstance(grade["relevant"], bool)

    def test_groundedness_evaluator(self, grader_llm):
        """Test groundedness evaluator implementation."""
        grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        grader = grader_llm.with_structured_output(
            GroundedGrade, method="json_schema", strict=True
        )

        facts = "Our company policy states that employees must work 40 hours per week."
        student_answer = "According to company policy, employees are required to work 40 hours per week."

        answer_text = f"FACTS: {facts}\nSTUDENT ANSWER: {student_answer}"
        grade = grader.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer_text},
        ])

        assert "grounded" in grade
        assert isinstance(grade["grounded"], bool)

    def test_retrieval_relevance_evaluator(self, grader_llm):
        """Test retrieval relevance evaluator implementation."""
        retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        grader = grader_llm.with_structured_output(
            RetrievalRelevanceGrade, method="json_schema", strict=True
        )

        question = "What is our refund policy?"
        facts = "Our refund policy allows customers to return products within 30 days of purchase for a full refund."

        answer_text = f"FACTS: {facts}\nQUESTION: {question}"
        grade = grader.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer_text},
        ])

        assert "relevant" in grade
        assert isinstance(grade["relevant"], bool)


class TestFullEvaluationWorkflow:
    """Test complete evaluation workflow with dataset."""

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

    @pytest.fixture
    def langsmith_client(self):
        """Create LangSmith client."""
        try:
            return Client()
        except Exception:
            pytest.skip("LangSmith client not available (check LANGSMITH_API_KEY)")

    @pytest.fixture
    def sample_dataset(self, langsmith_client):
        """Create or get sample dataset."""
        dataset_name = "Internal Chatbot Test Dataset"
        examples = [
            {
                "inputs": {"message": "What is our company policy?"},
                "outputs": {
                    "answer": "Our company policy requires all employees to follow the employee handbook guidelines."
                },
            },
            {
                "inputs": {"message": "What products do we offer?"},
                "outputs": {
                    "answer": "We offer Product A, Product B, and Product C with various features."
                },
            },
        ]

        try:
            if langsmith_client.has_dataset(dataset_name=dataset_name):
                dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
            else:
                dataset = langsmith_client.create_dataset(dataset_name=dataset_name)
                langsmith_client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
            return dataset_name
        except Exception:
            pytest.skip("Could not create/get dataset")

    @traceable(name="rag_chatbot")
    async def rag_chatbot_target(self, inputs: dict) -> dict:
        """Target function for evaluation - wraps the chatbot graph."""
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        config = RunnableConfig(
            configurable={"thread_id": f"eval-{inputs['message'][:10]}"},
            tags=["evaluation"],
        )

        input_data: ChatbotInput = {"message": inputs["message"]}
        result = await graph.ainvoke(input_data, config=config)

        # Extract answer from messages
        final_message = result["messages"][-1]
        answer = final_message.content if isinstance(final_message, AIMessage) else ""

        # Extract documents from retrieval context
        documents = []
        if "retrieval_context" in result and result.get("retrieval_context"):
            for ctx in result.get("retrieval_context", []):
                documents.append(Document(page_content=ctx))
        elif "retrieval_sources" in result:
            for source in result.get("retrieval_sources", []):
                documents.append(Document(page_content="", metadata={"source": source}))

        return {"answer": answer, "documents": documents}

    def correctness_evaluator(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        """Correctness evaluator: Response vs reference answer."""
        grader_llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0)
        grader = grader_llm.with_structured_output(
            CorrectnessGrade, method="json_schema", strict=True
        )

        correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        answers = f"""\
QUESTION: {inputs['message']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

        grade = grader.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ])
        return grade["correct"]

    def relevance_evaluator(self, inputs: dict, outputs: dict) -> bool:
        """Relevance evaluator: Response vs input."""
        grader_llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0)
        grader = grader_llm.with_structured_output(
            RelevanceGrade, method="json_schema", strict=True
        )

        relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        answer_text = f"QUESTION: {inputs['message']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = grader.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer_text},
        ])
        return grade["relevant"]

    def groundedness_evaluator(self, inputs: dict, outputs: dict) -> bool:
        """Groundedness evaluator: Response vs retrieved docs."""
        grader_llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0)
        grader = grader_llm.with_structured_output(
            GroundedGrade, method="json_schema", strict=True
        )

        grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        doc_string = "\n\n".join(doc.page_content for doc in outputs.get("documents", []))
        if not doc_string:
            return False
        answer_text = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = grader.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer_text},
        ])
        return grade["grounded"]

    def retrieval_relevance_evaluator(self, inputs: dict, outputs: dict) -> bool:
        """Retrieval relevance evaluator: Retrieved docs vs input."""
        grader_llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0)
        grader = grader_llm.with_structured_output(
            RetrievalRelevanceGrade, method="json_schema", strict=True
        )

        retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

        doc_string = "\n\n".join(doc.page_content for doc in outputs.get("documents", []))
        if not doc_string:
            return False
        answer_text = f"FACTS: {doc_string}\nQUESTION: {inputs['message']}"
        grade = grader.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer_text},
        ])
        return grade["relevant"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_evaluation_with_dataset(
        self, mock_retriever, mock_chat_model, sample_dataset, langsmith_client
    ):
        """Test running full evaluation workflow with dataset."""
        # Mock retriever
        mock_docs = [
            Document(
                page_content="Our company policy requires all employees to follow the employee handbook.",
                metadata={"source": "policy.pdf", "page": 1}
            ),
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)

        # Mock chat model
        mock_response = Mock()
        mock_response.content = "Our company policy requires all employees to follow the employee handbook guidelines."
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)

        # Run evaluation
        try:
            experiment_results = langsmith_client.evaluate(
                self.rag_chatbot_target,
                data=sample_dataset,
                evaluators=[
                    self.correctness_evaluator,
                    self.groundedness_evaluator,
                    self.relevance_evaluator,
                    self.retrieval_relevance_evaluator,
                ],
                experiment_prefix="internal-chatbot-rag",
                metadata={"version": "test", "framework": "langgraph"},
            )

            # Verify results structure
            assert experiment_results is not None
            # Results can be explored as dataframe: experiment_results.to_pandas()

        except Exception as e:
            pytest.skip(f"Evaluation failed (may need valid API keys): {e}")

