from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf

from .models import Chunk


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _windows(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    if not text:
        return []
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size // 6, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap // 6)
    return chunks


def extract_pdfs(pdf_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                page_text = _clean(page.get_text("text"))
                for position, text in enumerate(_windows(page_text)):
                    raw_id = f"{pdf_path.name}:{page_number}:{position}:{text[:80]}"
                    chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:32]
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=text,
                            source=pdf_path.name,
                            page=page_number,
                            metadata={"chunk": position},
                        )
                    )
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([chunk.as_dict() for chunk in chunks], indent=2), encoding="utf-8")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**record) for record in records]
