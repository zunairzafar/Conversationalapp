"""Tests for PDF processing logic."""

# TODO: add tests for load_pdf, load_multiple, split_documents


def test_pdf_processor_init():
    """PDFProcessor can be instantiated with defaults."""
    from src.pdf_processor import PDFProcessor

    processor = PDFProcessor()
    assert processor.chunk_size == 1000
    assert processor.chunk_overlap == 200
