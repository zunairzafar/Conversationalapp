"""
File utilities — helpers for reading, writing, and managing files.

Covers:
    - Saving Streamlit UploadedFile objects to disk  (Step 3)
    - Saving preprocessed text to data/processed/    (Step 6)
    - Listing / cleaning data directories
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def ensure_dirs() -> None:
    """Create data directories if they don't exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdf(uploaded_file, target_dir: Path = DATA_RAW_DIR) -> Path:
    """Save a Streamlit UploadedFile to disk and return the saved path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getvalue())
    return file_path


def save_processed_text(
    filename: str,
    text: str,
    target_dir: Path = DATA_PROCESSED_DIR,
) -> Path:
    """Save preprocessed text to the processed data directory.

    Stores both a .txt and a companion .json with basic metadata.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem

    txt_path = target_dir / f"{stem}.txt"
    txt_path.write_text(text, encoding="utf-8")

    meta_path = target_dir / f"{stem}.meta.json"
    meta_path.write_text(
        json.dumps({"source": filename, "chars": len(text)}, indent=2),
        encoding="utf-8",
    )
    return txt_path


def list_pdfs(directory: Path = DATA_RAW_DIR) -> list[Path]:
    """Return all PDF file paths in the given directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pdf"))


def clean_data_dir(directory: Path) -> None:
    """Remove all files in the given data directory (keeps the folder)."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
