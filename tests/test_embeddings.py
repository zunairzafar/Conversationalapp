"""Tests for embeddings manager."""

# TODO: add tests for get_embeddings, create_vector_store, load_vector_store


def test_embeddings_manager_init():
    """EmbeddingsManager can be instantiated with defaults."""
    from src.embeddings import EmbeddingsManager

    manager = EmbeddingsManager()
    assert "MiniLM" in manager.model_name
