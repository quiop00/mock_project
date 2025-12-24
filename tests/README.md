# Tests for Internal Chatbot

This directory contains unit and integration tests for the internal RAG chatbot, with LangSmith tracing and evaluation support.

## Setup

### Install test dependencies:
```bash
pip install -e ".[dev]"
```

Or install from requirements:
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio python-dotenv
```

### Environment Variables

Tests automatically load `.env` file from project root. Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=internal-chatbot-tests
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

**Note**: Tests use mocks for API calls, so you don't need a real API key for most tests. However, if you want to test with real API calls, set a valid `OPENAI_API_KEY`.

## Running Tests

### Run all tests:
```bash
pytest
```

### Run with verbose output:
```bash
pytest -v
```

### Run specific test file:
```bash
pytest tests/test_chatbot_nodes.py
```

### Run specific test class:
```bash
pytest tests/test_chatbot_nodes.py::TestReceiveUserMessage
```

### Run with coverage:
```bash
pytest --cov=internal_chatbot --cov-report=html
```

## Test Structure

- `conftest.py` - Pytest configuration, fixtures, and .env loading
- `test_chatbot_nodes.py` - Unit tests for individual graph nodes
- `test_chatbot_graph.py` - Integration tests for the complete graph workflow
- `test_langsmith_evaluation.py` - LangSmith evaluation and tracing tests

## LangSmith Integration

Tests use LangSmith's `@traceable` decorator and evaluation functions to:
- Trace all node executions
- Measure answer accuracy
- Detect hallucination
- Evaluate relevance
- Track performance metrics

### Enable LangSmith Tracing:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=internal-chatbot-tests
export LANGSMITH_API_KEY=your_api_key
pytest -v
```

All test runs will appear in LangSmith UI when tracing is enabled.

## Troubleshooting

### "api_key client option must be set" error

This happens when tests try to initialize the retriever without an API key. The `conftest.py` file automatically:
1. Loads `.env` file
2. Sets a test API key if none is found
3. Mocks retriever initialization in test mode

If you still see this error:
1. Make sure `.env` file exists in project root
2. Or set `OPENAI_API_KEY` environment variable
3. Or ensure tests are properly mocking the retriever (they should by default)

### Tests fail with "No documents found"

Tests mock the retriever, so this shouldn't happen. If it does:
- Check that `conftest.py` is being loaded
- Verify mocks are applied before retriever initialization
- Run with `pytest -v` to see detailed error messages
