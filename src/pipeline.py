"""
RAG Pipeline — orchestrates the full Retrieval-Augmented Generation flow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingsManager
from src.utils.langchain_utils import (
    get_llm,
    get_qa_prompt_template,
    get_query_rewrite_template,
    get_retriever,
    FALLBACK_MODELS,
)

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    config_path = _PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class RAGPipeline:
    """End-to-end RAG pipeline: ingest PDFs → answer questions."""

    def __init__(self, config: dict | None = None):
        self.config = config or _load_config()

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
        self.llm = None
        self._chain = None
        self._rewrite_chain = None
        self._active_model: str | None = None

    # ------------------------------------------------------------------
    # Ingestion — embeddings only, no LLM needed
    # ------------------------------------------------------------------

    def ingest(self, file_paths: list[str | Path]) -> int:
        all_chunks: list[Document] = []
        for fp in file_paths:
            chunks = self.pdf_processor.process_pdf(fp)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from the provided PDFs.")
            return 0

        if self.embeddings_manager.vector_store is not None:
            self.embeddings_manager.add_documents(all_chunks)
        else:
            self.embeddings_manager.create_vector_store(all_chunks)

        self._build_retriever()
        logger.info("Ingested %d chunk(s) from %d file(s)", len(all_chunks), len(file_paths))
        return len(all_chunks)

    # ------------------------------------------------------------------
    # Build retriever
    # ------------------------------------------------------------------

    def _build_retriever(self) -> None:
        if self.embeddings_manager.vector_store is None:
            return
        ret_cfg = self.config.get("retriever", {})
        self.retriever = get_retriever(
            self.embeddings_manager.vector_store,
            search_type=ret_cfg.get("search_type", "similarity"),
            search_kwargs=ret_cfg.get("search_kwargs", {"k": 4}),
        )

    # ------------------------------------------------------------------
    # Build LLM chain for a specific model
    # ------------------------------------------------------------------

    def _build_chain_for_model(self, model_name: str) -> bool:
        """Try to build the full LLM chain. Returns True on success."""
        llm_cfg = self.config.get("llm", {})
        try:
            llm = get_llm(
                model_name=model_name,
                temperature=llm_cfg.get("temperature", 0.1),
                max_new_tokens=llm_cfg.get("max_new_tokens", 512),
            )

            def _format_docs(docs: list[Document]) -> str:
                return "\n\n".join(doc.page_content for doc in docs)

            prompt = get_qa_prompt_template()
            rewrite_prompt = get_query_rewrite_template()

            self._rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            self._chain = (
                {
                    "context": RunnableLambda(lambda x: x["question"]) | self.retriever | _format_docs,
                    "question": RunnableLambda(lambda x: x["question"]),
                    "chat_history": RunnableLambda(lambda x: x["chat_history"]),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            self._active_model = model_name
            self.llm = llm
            logger.info("Chain built with model: %s", model_name)
            return True

        except Exception as e:
            logger.warning("Could not build chain with %s: %s", model_name, e)
            return False

    # ------------------------------------------------------------------
    # Query — with live fallback across models
    # ------------------------------------------------------------------

    def query(self, question: str, chat_history: list[dict] | None = None) -> dict:
        # Ensure retriever is ready
        if self.retriever is None:
            if self.embeddings_manager.has_persisted_store():
                self.embeddings_manager.load_vector_store()
                self._build_retriever()
            else:
                return {
                    "result": "No documents have been ingested yet. Please upload a PDF first.",
                    "source_documents": [],
                }

        # Format chat history string
        history_str = ""
        if chat_history:
            lines = []
            for msg in chat_history[-10:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_str = "\n".join(lines)

        # Retrieve source docs via embeddings (always works, no LLM needed)
        source_documents = self.retriever.invoke(question)

        # Rewrite query if chain already exists
        search_query = question
        if self._chain and history_str and self._rewrite_chain:
            try:
                rewritten = self._rewrite_chain.invoke(
                    {"chat_history": history_str, "question": question}
                ).strip()
                if rewritten:
                    search_query = rewritten
            except Exception:
                pass

        # Build fallback candidate list
        llm_cfg = self.config.get("llm", {})
        primary = llm_cfg.get("model_name") or FALLBACK_MODELS[0]
        candidates = [primary] + [m for m in FALLBACK_MODELS if m != primary]

        for candidate in candidates:
            # Build chain if not built or if switching to a new candidate
            if self._chain is None or self._active_model != candidate:
                if not self._build_chain_for_model(candidate):
                    continue

            try:
                answer = self._chain.invoke({
                    "question": search_query,
                    "chat_history": history_str,
                })
                return {"result": answer, "source_documents": source_documents}

            except Exception as e:
                err = str(e).lower()
                is_endpoint_error = any(
                    kw in err for kw in ["paused", "bad request", "503", "unavailable", "endpoint", "rate limit"]
                )
                if is_endpoint_error:
                    logger.warning("Model %s unavailable (%s), trying next.", candidate, e)
                    self._chain = None
                    self._active_model = None
                    continue
                # Unexpected error — don't swallow it
                logger.error("LLM error with %s: %s", candidate, e)
                return {
                    "result": f"⚠️ Error generating answer: {e}",
                    "source_documents": source_documents,
                }

        return {
            "result": (
                "⚠️ All LLM endpoints are currently unavailable. "
                "Please try again in a few minutes."
            ),
            "source_documents": source_documents,
        }

    @property
    def is_ready(self) -> bool:
        return self.retriever is not None
