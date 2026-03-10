"""
Text utilities — helpers for cleaning and transforming text.

Covers Steps 5 of the application flow:
    - Tokenization
    - Sentence segmentation
    - Removal of unwanted characters
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise whitespace, strip control chars, and tidy up extracted PDF text."""
    # Unicode normalisation (handles ligatures, accented chars, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove NULL bytes and most control characters (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse multiple blank lines into a single one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces / tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


def merge_hyphenated_words(text: str) -> str:
    """Rejoin words that were hyphenated at line breaks (e.g. 'pro-\\ncess' → 'process')."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def remove_unwanted_characters(text: str) -> str:
    """Remove non-printable / non-standard characters while keeping basic punctuation."""
    # Keep letters, digits, common punctuation, whitespace
    text = re.sub(r"[^\w\s.,;:!?'\"\-\(\)\[\]{}/@#&%$+=/\\<>°•–—…]", "", text)
    return text


# ---------------------------------------------------------------------------
# Tokenization & sentence segmentation (lightweight, no heavy NLP dep)
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])"   # lookbehind for sentence-ending punctuation
    r"\s+"           # followed by whitespace
    r"(?=[A-Z])"     # lookahead for capital letter (new sentence)
)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation-aware tokenizer."""
    # Split on whitespace, then separate trailing punctuation
    tokens: list[str] = []
    for word in text.split():
        # Separate leading/trailing punctuation
        match = re.match(r"^(\W*)(.*?)(\W*)$", word)
        if match:
            leading, core, trailing = match.groups()
            if leading:
                tokens.append(leading)
            if core:
                tokens.append(core)
            if trailing:
                tokens.append(trailing)
    return tokens


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences using regex heuristic."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (Steps 5)
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Run the complete preprocessing pipeline on raw extracted text.

    Pipeline:
        1. Merge hyphenated words
        2. Remove unwanted characters
        3. Normalise whitespace / control chars
    """
    text = merge_hyphenated_words(text)
    text = remove_unwanted_characters(text)
    text = clean_text(text)
    return text
