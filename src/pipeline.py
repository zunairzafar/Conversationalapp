"""
RAG Pipeline — orchestrates the full Retrieval-Augmented Generation flow.

This is the central module that ties every step together:

    Ingestion  (Steps 3-6):  PDF → extract → preprocess → chunk → embed → store
    Querying   (Steps 7-11): query → retrieve passages → generate answer
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingsManager
from src.utils.langchain_utils import get_llm, get_qa_prompt_template, get_query_rewrite_template, get_retriever

logger = logging.getLogger(__name__)

# Project root (one level above src/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = _PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class RAGPipeline:
    """End-to-end RAG pipeline: ingest PDFs → answer questions."""

    def __init__(self, config: dict | None = None):
        self.config = config or _load_config()

        # Sub-components
        pdf_cfg = self.config.get("pdf_processing", {})
        emb_cfg = self.config.get("embeddings", {})

        self.pdf_processor = PDFProcessor(
            chunk_size=pdf_cfg.get("chunk_size", 1000),
            chunk_overlap=pdf_cfg.get("chunk_overlap", 200),
            loader_type=pdf_cfg.get("loader", "pypdf"),
        )
        self.embeddings_manager = EmbeddingsManager(
            model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=emb_cfg.get("device", "cpu"),
            persist_dir=self.config.get("vector_store", {}).get(
                "persist_directory",
                str(_PROJECT_ROOT / "models" / "retriever"),
            ),
        )

        self.retriever = None
        self.chain = None
        self.llm = None
        self._rewrite_chain = None

    # ------------------------------------------------------------------
    # Steps 3-6: Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_paths: list[str | Path]) -> int:
        """Load PDFs, preprocess, chunk, embed, and store.

        Returns the number of chunks ingested.
        """
        all_chunks: list[Document] = []
        for fp in file_paths:
            chunks = self.pdf_processor.process_pdf(fp)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from the provided PDFs.")
            return 0

        # Build (or extend) the vector store
        if self.embeddings_manager.vector_store is not None:
            self.embeddings_manager.add_documents(all_chunks)
        else:
            self.embeddings_manager.create_vector_store(all_chunks)

        # Rebuild retriever & chain after new ingestion
        self._build_chain()

        logger.info("Ingested %d chunk(s) from %d file(s)", len(all_chunks), len(file_paths))
        return len(all_chunks)

    # ------------------------------------------------------------------
    # Steps 8-11: Build the retrieval-QA chain
    # ------------------------------------------------------------------

    def _build_chain(self) -> None:
        """Construct the RAG chain using LCEL (LangChain Expression Language)."""
        if self.embeddings_manager.vector_store is None:
            logger.warning("No vector store available — cannot build chain.")
            return

        ret_cfg = self.config.get("retriever", {})
        self.retriever = get_retriever(
            self.embeddings_manager.vector_store,
            search_type=ret_cfg.get("search_type", "similarity"),
            search_kwargs=ret_cfg.get("search_kwargs", {"k": 4}),
        )

        llm_cfg = self.config.get("llm", {})
        self.llm = get_llm(
            model_name=llm_cfg.get("model_name") or None,
            temperature=llm_cfg.get("temperature", 0.1),
            max_new_tokens=llm_cfg.get("max_new_tokens", 512),
        )

        prompt = get_qa_prompt_template()
        rewrite_prompt = get_query_rewrite_template()

        def _format_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # Query rewriter: turns follow-ups into standalone questions
        self._rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()

        # LCEL chain: accepts {"question": str, "chat_history": str}
        # 1. Retrieve docs using the (rewritten) question
        # 2. Format context + inject chat history
        # 3. Prompt → LLM → parse
        self.chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | self.retriever | _format_docs,
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("LCEL RAG chain built successfully.")

    # ------------------------------------------------------------------
    # Steps 7, 10-11: Query
    # ------------------------------------------------------------------

    def query(self, question: str, chat_history: list[dict] | None = None) -> dict:
        """Run a question through the RAG pipeline.

        Args:
            question: The user's question.
            chat_history: List of {"role": ..., "content": ...} message dicts.

        Returns:
            dict with keys "result" (str) and "source_documents" (list[Document])
        """
        if self.chain is None:
            # Try to load a persisted store if one exists
            if self.embeddings_manager.has_persisted_store():
                self.embeddings_manager.load_vector_store()
                self._build_chain()
            else:
                return {
                    "result": "No documents have been ingested yet. Please upload a PDF first.",
                    "source_documents": [],
                }

        # Format chat history into a readable string for the prompt
        history_str = ""
        if chat_history:
            lines = []
            for msg in chat_history[-10:]:  # keep last 10 messages to avoid token overflow
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_str = "\n".join(lines)

        # Rewrite follow-up questions into standalone queries for better retrieval
        search_query = question
        if history_str and self._rewrite_chain:
            try:
                rewritten = self._rewrite_chain.invoke(
                    {"chat_history": history_str, "question": question}
                ).strip()
                if rewritten:
                    search_query = rewritten
                    logger.info("Rewrote query: '%s' → '%s'", question, search_query)
            except Exception as e:
                logger.warning("Query rewrite failed, using original: %s", e)

        # Retrieve source documents using the rewritten query
        source_documents = self.retriever.invoke(search_query)
        answer = self.chain.invoke({"question": search_query, "chat_history": history_str})

        return {
            "result": answer,
            "source_documents": source_documents,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True when the pipeline has an active chain (vector store loaded)."""
        return self.chain is not None
