"""
LangChain-specific utilities — prompt templates, LLM factory, retriever builder.

Uses Groq as the LLM provider (free tier, cloud-reliable, no gated models).
Get a free API key at: https://console.groq.com
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# Groq free-tier models — tried in order if one is unavailable
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


def _load_groq_token() -> None:
    """Load GROQ_API_KEY from Streamlit secrets or .env fallback."""
    if os.environ.get("GROQ_API_KEY"):
        return
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
            return
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        load_dotenv(_PROJECT_ROOT / ".env")
    except Exception:
        pass


_load_groq_token()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RAG_QA_TEMPLATE = """\
You are a helpful assistant that answers questions based on the provided document context.
Use the chat history to understand follow-up questions.
If you cannot find the answer in the context, say "I don't have enough information to answer that."

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""

_QUERY_REWRITE_TEMPLATE = """\
Given the following chat history and a follow-up question, rewrite the follow-up question \
to be a standalone question that includes all necessary context.
If the question is already standalone, return it as-is. Only return the rewritten question, nothing else.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def get_qa_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=_RAG_QA_TEMPLATE,
    )


def get_query_rewrite_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=_QUERY_REWRITE_TEMPLATE,
    )


# ---------------------------------------------------------------------------
# LLM factory — Groq (free, fast, cloud-reliable)
# ---------------------------------------------------------------------------

def get_llm(
    model_name: str | None = None,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
    **kwargs,
):
    """Instantiate a ChatGroq LLM. Requires GROQ_API_KEY in env/secrets."""
    candidate = model_name or FALLBACK_MODELS[0]
    return ChatGroq(
        model=candidate,
        temperature=temperature,
        max_tokens=max_new_tokens,
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
    )


# ---------------------------------------------------------------------------
# Retriever helper
# ---------------------------------------------------------------------------

def get_retriever(vector_store, search_type: str = "similarity", search_kwargs: dict | None = None):
    search_kwargs = search_kwargs or {"k": 4}
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
