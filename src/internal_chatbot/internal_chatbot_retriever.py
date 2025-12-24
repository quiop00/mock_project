"""
Document ingestion, chunking, embeddings, and retriever setup for the chatbot.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from internal_chatbot import internal_chatbot_config as cfg


def _load_internal_documents(doc_path: str) -> List[Document]:
    base = Path(doc_path)
    # Create directory if it doesn't exist
    base.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []

    for pdf_file in base.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    for docx_file in base.rglob("*.docx"):
        loader = Docx2txtLoader(str(docx_file))
        docs.extend(loader.load())

    if not docs:
        raise ValueError(
            f"No documents found under '{doc_path}'. Add PDF/DOCX files for retrieval."
        )

    return docs


def _get_embeddings():
    return OpenAIEmbeddings(model=cfg.OPENAI_EMBEDDING_MODEL)


def _ensure_vector_store_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_vector_store(docs: List[Document]):
    embeddings = _get_embeddings()
    _ensure_vector_store_dir(cfg.VECTOR_STORE_PATH)

    if cfg.VECTOR_STORE.lower() == "chroma":
        return Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=cfg.VECTOR_STORE_PATH,
            collection_name="internal_docs",
        )

    return FAISS.from_documents(docs, embedding=embeddings)


def _build_retriever():
    docs = _load_internal_documents(cfg.INTERNAL_DOCS_PATH)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=cfg.CHUNK_SIZE_TOKENS,
        chunk_overlap=cfg.CHUNK_OVERLAP_TOKENS,
    )
    chunked_docs = splitter.split_documents(docs)

    store = _build_vector_store(chunked_docs)

    retriever = store.as_retriever(search_kwargs={"k": cfg.RETRIEVAL_K})
    return retriever


RETRIEVER = _build_retriever()

__all__ = ["RETRIEVER"]

