"""
PDF Processor — handles loading, parsing, preprocessing, and chunking of PDFs.

Covers Steps 3-6 of the application flow:
    3. Load the PDF file
    4. Extract text (supports structured & unstructured PDFs)
    5. Preprocess: tokenization, sentence segmentation, cleanup
    6. Save preprocessed text to data/processed/
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.text_utils import preprocess_text
from src.utils.file_utils import save_processed_text

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Load, extract, preprocess, and chunk PDF documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        loader_type: str = "pypdf",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader_type = loader_type  # "pypdf" or "unstructured"
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Step 3-4: Load & extract
    # ------------------------------------------------------------------

    def _get_loader(self, file_path: str | Path):
        """Return the appropriate document loader."""
        file_path = str(file_path)
        if self.loader_type == "unstructured":
            from langchain_community.document_loaders import UnstructuredFileLoader
            return UnstructuredFileLoader(file_path, mode="elements")
        return PyPDFLoader(file_path)

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """Load a single PDF and return LangChain Document objects."""
        loader = self._get_loader(file_path)
        documents = loader.load()
        logger.info("Loaded %d page(s) from %s", len(documents), file_path)
        return documents

    def load_multiple(self, file_paths: list[str | Path]) -> list[Document]:
        """Load multiple PDFs and return aggregated documents."""
        all_docs: list[Document] = []
        for fp in file_paths:
            all_docs.extend(self.load_pdf(fp))
        return all_docs

    # ------------------------------------------------------------------
    # Step 5: Preprocess extracted text
    # ------------------------------------------------------------------

    def preprocess_documents(self, documents: list[Document]) -> list[Document]:
        """Apply text preprocessing to every document's page_content."""
        processed: list[Document] = []
        for doc in documents:
            clean_content = preprocess_text(doc.page_content)
            if clean_content:  # skip empty pages
                processed.append(
                    Document(
                        page_content=clean_content,
                        metadata=doc.metadata,
                    )
                )
        return processed

    # ------------------------------------------------------------------
    # Step 6: Save preprocessed text
    # ------------------------------------------------------------------

    def save_preprocessed(self, documents: list[Document], source_name: str) -> Path:
        """Concatenate preprocessed pages and save to data/processed/."""
        full_text = "\n\n".join(doc.page_content for doc in documents)
        return save_processed_text(source_name, full_text)

    # ------------------------------------------------------------------
    # Chunking for embedding
    # ------------------------------------------------------------------

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks for embedding."""
        chunks = self._splitter.split_documents(documents)
        logger.info("Split into %d chunk(s)", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Convenience: full pipeline for a single PDF
    # ------------------------------------------------------------------

    def process_pdf(self, file_path: str | Path) -> list[Document]:
        """Load → preprocess → save → split. Returns ready-to-embed chunks."""
        documents = self.load_pdf(file_path)
        documents = self.preprocess_documents(documents)
        self.save_preprocessed(documents, Path(file_path).name)
        chunks = self.split_documents(documents)
        return chunks
