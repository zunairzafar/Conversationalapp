"""
Streamlit Chat UI for the RAG PDF project.

Covers every user-facing step of the application flow:
    Step 1-2 :  Upload PDF(s) via sidebar
    Step 3-6 :  Ingest → extract → preprocess → embed  (triggered on upload)
    Step 7   :  User enters a query in the chat box
    Step 8-9 :  Retriever searches for relevant passages
    Step 10-11: Generator produces an answer from retrieved context
    Step 12  :  Output displayed in the chat interface
    Step 13  :  User can download chat history or refine their query

Run with:
    cd "Rag project"
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path so `from src.…` imports work
# when Streamlit is launched from inside "Rag project/".
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import RAGPipeline
from src.utils.file_utils import save_uploaded_pdf, DATA_RAW_DIR

# ======================================================================
# Session-state helpers
# ======================================================================

def _get_pipeline() -> RAGPipeline:
    """Return (and cache) a single RAGPipeline instance per session."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline


def _init_session_state() -> None:
    """Initialise all session-state keys used by the app."""
    defaults = {
        "messages": [],
        "ingested_files": [],
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ======================================================================
# Main application
# ======================================================================

def main() -> None:
    st.set_page_config(
        page_title="RAG PDF Chat",
        page_icon="📄",
        layout="wide",
    )

    _init_session_state()
    pipeline = _get_pipeline()

    st.title("📄 Chat with your PDFs")
    st.caption("Upload any PDF — structured or unstructured — and ask questions about it.")

    # ==================================================================
    # Sidebar — Steps 1-2: Upload + Steps 3-6: Ingest
    # ==================================================================
    with st.sidebar:
        st.header("📁 Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose PDF file(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Supports structured tables, forms, and free-text PDFs.",
        )

        if uploaded_files:
            # Determine which files are new (not yet ingested)
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.ingested_files
            ]

            if new_files:
                if st.button(f"🚀 Click here to Process {len(new_files)} new PDF(s)", type="primary"):
                    saved_paths: list[Path] = []
                    with st.spinner("Saving uploads…"):
                        for uf in new_files:
                            saved_paths.append(save_uploaded_pdf(uf))

                    # Steps 3-6: ingest through the pipeline
                    with st.spinner("Extracting text, preprocessing & building index…"):
                        num_chunks = pipeline.ingest(saved_paths)

                    st.session_state.total_chunks += num_chunks
                    for f in new_files:
                        st.session_state.ingested_files.append(f.name)

                    st.success(
                        f"✅ Processed {len(new_files)} file(s) → "
                        f"{num_chunks} new chunk(s) indexed."
                    )
            else:
                st.info("All uploaded files have already been processed.")

        # Show ingestion status
        if st.session_state.ingested_files:
            st.divider()
            st.subheader("📄 Ingested Files")
            for fname in st.session_state.ingested_files:
                st.write(f"• {fname}")
            st.metric("Total chunks in index", st.session_state.total_chunks)
        else:
            st.info("Upload one or more PDFs to get started.")

        # Step 13: Download chat history
        st.divider()
        if st.session_state.messages:
            chat_export = _format_chat_for_download(st.session_state.messages)
            st.download_button(
                label="⬇️ Download chat history",
                data=chat_export,
                file_name="chat_history.txt",
                mime="text/plain",
            )

        # Reset button
        if st.button("🗑️ Clear session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ==================================================================
    # Main area — Steps 7-12: Chat interface
    # ==================================================================

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Step 7: User enters a query
    if user_input := st.chat_input("Ask a question about your documents…"):
        # Show the user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Steps 8-11: Retrieve + Generate
        with st.chat_message("assistant"):
            if not st.session_state.ingested_files:
                reply = "⚠️ No documents have been uploaded yet. Please upload a PDF in the sidebar first."
            else:
                with st.spinner("Thinking…"):
                    response = pipeline.query(
                        user_input,
                        chat_history=st.session_state.messages[:-1],  # exclude the just-added user msg
                    )
                    reply = response.get("result", "I could not generate an answer.")

            st.markdown(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )


# ======================================================================
# Utilities
# ======================================================================

def _format_chat_for_download(messages: list[dict]) -> str:
    """Format chat messages into a plain-text string for download."""
    lines: list[str] = []
    for msg in messages:
        role = msg["role"].upper()
        lines.append(f"[{role}]\n{msg['content']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
