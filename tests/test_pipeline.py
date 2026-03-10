"""Tests for the RAG pipeline."""

# TODO: add tests for ingest, build_chain, and query


def test_pipeline_init():
    """RAGPipeline can be instantiated with default config."""
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    assert pipeline.config == {}
    assert pipeline.vector_store is None
