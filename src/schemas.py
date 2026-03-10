"""
Pydantic schemas for request / response validation and configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for an incoming user question."""

    question: str = Field(..., min_length=1, description="The user's question")


class QueryResponse(BaseModel):
    """Schema for the RAG pipeline response."""

    answer: str
    source_documents: list[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata attached to each ingested document chunk."""

    source: str
    page: int | None = None
    chunk_index: int | None = None


class IngestionResult(BaseModel):
    """Summary returned after PDF ingestion."""

    filenames: list[str]
    total_chunks: int
    status: str = "success"


class AppConfig(BaseModel):
    """Typed representation of config.yaml (top-level)."""

    class PDFProcessing(BaseModel):
        chunk_size: int = 1000
        chunk_overlap: int = 200
        loader: str = "pypdf"

    class Embeddings(BaseModel):
        provider: str = "huggingface"
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    class VectorStore(BaseModel):
        provider: str = "faiss"
        persist_directory: str = "models/retriever"

    class LLM(BaseModel):
        provider: str = "huggingface_hub"
        model_name: str = ""
        temperature: float = 0.3
        max_new_tokens: int = 512

    class Retriever(BaseModel):
        search_type: str = "similarity"
        search_kwargs: dict = Field(default_factory=lambda: {"k": 4})

    pdf_processing: PDFProcessing = Field(default_factory=PDFProcessing)
    embeddings: Embeddings = Field(default_factory=Embeddings)
    vector_store: VectorStore = Field(default_factory=VectorStore)
    llm: LLM = Field(default_factory=LLM)
    retriever: Retriever = Field(default_factory=Retriever)
