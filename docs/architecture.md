# Architecture

## Overview

This RAG (Retrieval-Augmented Generation) project follows a modular pipeline architecture:

1. **PDF Ingestion** — Load and parse structured/unstructured PDFs
2. **Text Processing** — Chunk, clean, and prepare text for embedding
3. **Embedding & Indexing** — Generate vector embeddings and store in a vector store
4. **Retrieval** — Retrieve relevant chunks given a user query
5. **Generation** — Feed retrieved context + query into an LLM to produce an answer
6. **UI** — Streamlit chat interface for interactive Q&A

## Component Diagram

```
PDF Files ──▶ PDF Processor ──▶ Text Splitter ──▶ Embeddings ──▶ Vector Store
                                                                      │
User Query ──▶ Retriever (Vector Store) ──▶ LLM Chain ──▶ Response ◀──┘
```

## Tech Stack

- **Framework**: LangChain
- **Embeddings**: HuggingFace (sentence-transformers)
- **Vector Store**: FAISS (local) / Chroma
- **LLM**: HuggingFace Hub models
- **UI**: Streamlit
- **Validation**: Pydantic
