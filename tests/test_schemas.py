"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError


def test_query_request_valid():
    from src.schemas import QueryRequest

    req = QueryRequest(question="What is RAG?")
    assert req.question == "What is RAG?"


def test_query_request_empty_rejected():
    from src.schemas import QueryRequest

    with pytest.raises(ValidationError):
        QueryRequest(question="")
