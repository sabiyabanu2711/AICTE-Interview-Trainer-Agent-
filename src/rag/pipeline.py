"""
RAG pipeline: document ingestion, embedding, vector store, and retrieval.
Uses HuggingFace sentence-transformers for embeddings and ChromaDB as the
local vector store. IBM Slate embeddings via watsonx.ai are used when available.
This module has NO dependency on LangChain to avoid DLL conflicts.
"""

import os
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.config import config

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Simple Document container (replaces LangChain Document)
# ---------------------------------------------------------------------------
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text splitter (replaces LangChain RecursiveCharacterTextSplitter)
# ---------------------------------------------------------------------------
def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def split_documents(docs: List[Document], chunk_size: int = 512, overlap: int = 64) -> List[Document]:
    """Split a list of Documents into smaller chunks."""
    result = []
    for doc in docs:
        parts = split_text(doc.page_content, chunk_size, overlap)
        for part in parts:
            result.append(Document(page_content=part, metadata=doc.metadata.copy()))
    return result


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------
def _get_embedding_fn():
    """
    Returns a callable(list[str]) -> list[list[float]].
    Uses ChromaDB's built-in ONNX embedding (no PyTorch / no DLL issues).
    """
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        logger.info("Using ChromaDB built-in ONNX embedding (all-MiniLM-L6-v2).")
        _onnx = ONNXMiniLM_L6_V2()

        class OnnxFn:
            def __call__(self, input: List[str]) -> List[List[float]]:
                return _onnx(input)

        return OnnxFn()
    except Exception as exc:
        logger.warning("ONNX embedding failed (%s), using TF-IDF fallback.", exc)
        return _tfidf_embedding_fn()


def _tfidf_embedding_fn():
    """
    Pure-Python TF-IDF based embedding fallback.
    No external DLLs, no network calls — works everywhere.
    Produces 512-dim sparse vectors good enough for keyword-based retrieval.
    """
    import math
    import hashlib

    def _hash_embed(texts: List[str]) -> List[List[float]]:
        DIM = 512
        result = []
        for text in texts:
            vec = [0.0] * DIM
            words = text.lower().split()
            word_freq: Dict[str, int] = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            for word, freq in word_freq.items():
                # Deterministic bucket using hash
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % DIM
                tf = freq / max(len(words), 1)
                idf = math.log(1 + 1.0 / (1 + freq))
                vec[idx] += tf * idf
            # L2 normalise
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            result.append([x / norm for x in vec])
        return result

    logger.info("Using pure-Python TF-IDF embedding fallback.")

    class TFIDFFn:
        def __call__(self, input: List[str]) -> List[List[float]]:
            return _hash_embed(input)

    return TFIDFFn()


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    """
    Manages the full RAG pipeline using ChromaDB directly (no LangChain).
    """

    def __init__(self):
        self._embedding_fn = None
        self._collection = None
        self._store_path = Path(config.vector_store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_fn(self):
        if self._embedding_fn is None:
            self._embedding_fn = _get_embedding_fn()
        return self._embedding_fn

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self._store_path))
            self._collection = client.get_or_create_collection(
                name="interview_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_documents(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_fn(texts)
        ids = [f"doc_{self.collection.count()}_{i}" for i in range(len(chunks))]
        metadatas = [c.metadata for c in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Ingested %d chunks.", len(chunks))
        return len(chunks)

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        return self.ingest_documents([Document(page_content=text, metadata=metadata or {})])

    def ingest_knowledge_base(self, kb_path: str = "./data/knowledge_base") -> int:
        kb_dir = Path(kb_path)
        if not kb_dir.exists():
            logger.warning("Knowledge base directory not found: %s", kb_path)
            return 0

        total = 0
        for json_file in kb_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    records = json.load(f)
                docs = []
                for record in records:
                    content = record.get("content", "")
                    if content:
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": json_file.name,
                                "role": record.get("role", ""),
                                "category": record.get("category", ""),
                                "difficulty": record.get("difficulty", ""),
                            }
                        ))
                total += self.ingest_documents(docs)
            except Exception as exc:
                logger.error("Failed to load %s: %s", json_file, exc)
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        k = k or config.top_k_retrieval
        query_embedding = self.embedding_fn([query])[0]

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, max(self.collection.count(), 1)),
            "include": ["documents", "metadatas"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        try:
            results = self.collection.query(**kwargs)
            docs = []
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                docs.append(Document(page_content=text, metadata=meta or {}))
            return docs
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []

    def retrieve_as_context(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        docs = self.retrieve(query, k=k, filter_metadata=filter_metadata)
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "knowledge base")
            parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
        return "\n\n".join(parts)

    def collection_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0


# Singleton
rag_pipeline = RAGPipeline()
