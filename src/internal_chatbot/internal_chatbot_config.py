"""
Configuration constants for the internal RAG chatbot.
"""

import os

# Paths and storage
INTERNAL_DOCS_PATH = os.getenv("INTERNAL_DOCS_PATH", "data/internal_docs")
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")  # faiss | chroma
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "var/vector_store/internal")

# Embeddings
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")  # openai | hf
HF_EMBEDDINGS_MODEL = os.getenv(
    "HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# Chunking and retrieval
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))

SYSTEM_PROMPT = """
You are the company’s internal assistant. Mandatory rules:
- Always rely on information from internal documents. Do not invent or infer.
- Respond concisely, clearly, professionally, and kindly.
- Always cite the source (file/page) when available.
- If no relevant info is found, reply exactly:
  "I could not find this information in the internal documents. Would you like me to help in another way?"
"""

__all__ = [
    "INTERNAL_DOCS_PATH",
    "VECTOR_STORE",
    "VECTOR_STORE_PATH",
    "EMBEDDINGS_PROVIDER",
    "HF_EMBEDDINGS_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "CHUNK_SIZE_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "RETRIEVAL_K",
    "MAX_HISTORY_MESSAGES",
    "SYSTEM_PROMPT",
]

