from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SearchHit:
    id: str
    text: str
    source: str
    page: int
    score: float
    channels: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class AgentState(TypedDict, total=False):
    query: str
    safe_query: str
    is_safe: bool
    safety_note: str
    dense_hits: list[SearchHit]
    sparse_hits: list[SearchHit]
    fused_hits: list[SearchHit]
    reranked_hits: list[SearchHit]
    confidence: float
    grounded: bool
    answer: str
    image_prompt: str
    image_path: str | None
    warnings: list[str]
