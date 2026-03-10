from setuptools import setup, find_packages

setup(
    name="rag-pdf-chat",
    version="0.1.0",
    description="Chat with PDF documents using RAG (LangChain + HuggingFace)",
    python_requires=">=3.13",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.3",
        "langchain-community>=0.3",
        "langchain-core>=0.3",
        "langchain-huggingface>=0.1",
        "huggingface-hub>=0.25",
        "pydantic>=2.9",
        "streamlit>=1.38",
        "faiss-cpu>=1.8",
        "pypdf>=4.0",
        "python-dotenv>=1.0",
        "pyyaml>=6.0",
    ],
)
