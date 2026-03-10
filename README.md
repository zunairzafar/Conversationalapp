# RAG PDF Chat

> Chat with any PDF — structured or unstructured — using Retrieval-Augmented Generation powered by LangChain and HuggingFace.

## Project Structure

```
Rag project/
├── assets/                     # Static assets (images, icons)
├── data/
│   ├── raw/                    # Upload raw PDFs here
│   └── processed/              # Processed / chunked data
├── docs/                       # Architecture & user guides
├── models/
│   ├── retriever/              # Persisted vector store files
│   ├── generator/              # Generator model artefacts
│   └── config/                 # Model configuration YAML
├── src/
│   ├── __init__.py
│   ├── app.py                  # Streamlit chat UI
│   ├── pipeline.py             # RAG pipeline orchestration
│   ├── pdf_processor.py        # PDF loading & chunking
│   ├── embeddings.py           # Embedding model & vector store management
│   ├── schemas.py              # Pydantic request/response schemas
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py       # File I/O helpers
│       ├── text_utils.py       # Text cleaning helpers
│       └── langchain_utils.py  # LangChain prompt/chain helpers
├── tests/
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_pipeline.py
│   ├── test_pdf_processor.py
│   ├── test_embeddings.py
│   └── test_schemas.py
├── .env                        # API keys (not committed)
├── .gitignore
├── config.yaml                 # Project-wide configuration
├── Dockerfile                  # Container build
├── README.md
├── requirements.txt
└── setup.py
```

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your HuggingFace token
#    Edit .env and add your HUGGINGFACEHUB_API_TOKEN

# 4. Run the app
streamlit run src/app.py
```

## Tech Stack

| Component     | Library                         |
|---------------|---------------------------------|
| Framework     | LangChain                       |
| Embeddings    | HuggingFace sentence-transformers |
| Vector Store  | FAISS                           |
| LLM           | HuggingFace Hub                 |
| UI            | Streamlit                       |
| Validation    | Pydantic                        |

## Status

🏗️ **Scaffold only** — module stubs are in place; implementation is TODO.
