"""
Embeddings Manager — handles creation and management of embeddings + vector store.

Covers Steps 8-9 of the application flow:
    8. Build / load a FAISS vector store for retrieval
    9. Provide a retriever that searches for similar passages
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Default persist path (relative to project root)
_DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[1] / "models" / "retriever"


class EmbeddingsManager:
    """Manage HuggingFace embedding models and FAISS vector store lifecycle."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        persist_dir: str | Path | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.persist_dir = Path(persist_dir) if persist_dir else _DEFAULT_PERSIST_DIR
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vector_store: FAISS | None = None

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialise (once) and return the HuggingFace embedding model."""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Loaded embedding model: %s", self.model_name)
        return self.embeddings

    # ------------------------------------------------------------------
    # Vector store — create from documents
    # ------------------------------------------------------------------

    def create_vector_store(self, documents: list[Document]) -> FAISS:
        """Create a FAISS vector store from document chunks and persist it."""
        embeddings = self.get_embeddings()
        self.vector_store = FAISS.from_documents(documents, embeddings)

        # Persist to disk
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.persist_dir))
        logger.info(
            "Created FAISS store with %d vectors → %s",
            len(documents),
            self.persist_dir,
        )
        return self.vector_store

    # ------------------------------------------------------------------
    # Vector store — add new documents to existing store
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> FAISS:
        """Add new document chunks to the existing vector store."""
        if self.vector_store is None:
            return self.create_vector_store(documents)

        self.vector_store.add_documents(documents)

        # Re-persist
        self.vector_store.save_local(str(self.persist_dir))
        logger.info("Added %d new vectors to FAISS store", len(documents))
        return self.vector_store

    # ------------------------------------------------------------------
    # Vector store — load from disk
    # ------------------------------------------------------------------

    def load_vector_store(self, persist_dir: str | Path | None = None) -> FAISS:
        """Load a previously persisted FAISS vector store."""
        directory = Path(persist_dir) if persist_dir else self.persist_dir
        embeddings = self.get_embeddings()
        self.vector_store = FAISS.load_local(
            str(directory),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS store from %s", directory)
        return self.vector_store

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_persisted_store(self) -> bool:
        """Check if a persisted FAISS index already exists on disk."""
        index_file = self.persist_dir / "index.faiss"
        return index_file.exists()
