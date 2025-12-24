"""
Pytest configuration and fixtures for chatbot tests.

This file automatically loads .env file and provides common fixtures
for all tests.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded .env from {env_file}")
else:
    # Try example.env as fallback
    example_env = project_root / "example.env"
    if example_env.exists():
        load_dotenv(example_env)
        print(f"Loaded example.env from {example_env}")
    else:
        print("Warning: No .env or example.env file found. Using environment variables only.")

# Set default test environment variables if not already set
# Only use real API key if it exists, otherwise use test key
real_openai_key = os.getenv("OPENAI_API_KEY")
if not real_openai_key or real_openai_key == "your_openai_api_key_here":
    os.environ["OPENAI_API_KEY"] = "test-key-for-mocking"
    print("Using test API key (tests will mock API calls)")
else:
    print(f"Using real API key from environment (prefix: {real_openai_key[:10]}...)")

os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "internal-chatbot-tests"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests."""
    # Ensure test data directory exists
    test_data_dir = project_root / "data" / "internal_docs"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy test document if none exists
    test_doc = test_data_dir / "test_document.txt"
    if not test_doc.exists() and not any(test_data_dir.glob("*.pdf")) and not any(test_data_dir.glob("*.docx")):
        test_doc.write_text("This is a test document for unit testing purposes.")
    
    yield
    
    # Cleanup if needed
    pass


@pytest.fixture
def mock_openai_api_key(monkeypatch):
    """Mock OpenAI API key for tests that don't need real API calls."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    yield "test-openai-key"


@pytest.fixture
def disable_api_calls(monkeypatch):
    """Disable real API calls by setting invalid keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-real-calls")
    yield


@pytest.fixture(autouse=True)
def reset_environment_variables():
    """Reset environment variables before each test."""
    # Store original values
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
    original_project = os.environ.get("LANGCHAIN_PROJECT")
    
    yield
    
    # Restore original values
    if original_openai_key:
        os.environ["OPENAI_API_KEY"] = original_openai_key
    if original_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = original_tracing
    if original_project:
        os.environ["LANGCHAIN_PROJECT"] = original_project


@pytest.fixture(autouse=True)
def mock_retriever_initialization():
    """Mock retriever initialization to avoid needing real API keys during import."""
    # Mock the retriever module before it's imported
    # This is done via patches in individual test files
    pass

