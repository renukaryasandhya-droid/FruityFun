from __future__ import annotations

import re
from collections import defaultdict

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from rank_bm25 import BM25Plus

from .config import Settings
from .models import Chunk, SearchHit


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, settings: Settings, chunks: list[Chunk]):
        self.settings = settings
        self.chunks = chunks
        self._by_id = {chunk.id: chunk for chunk in chunks}
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Plus(tokenized) if tokenized else None
        self._embeddings = None
        self._index = None

    def _dense_clients(self):
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_embedding_model,
            )
            self._index = Pinecone(api_key=self.settings.pinecone_api_key).Index(
                self.settings.pinecone_dense_index
            )
        return self._embeddings, self._index

    def dense_search(self, query: str) -> list[SearchHit]:
        if not self.settings.openai_api_key or not self.settings.pinecone_api_key:
            return []
        embeddings, index = self._dense_clients()
        vector = embeddings.embed_query(query)
        result = index.query(
            namespace=self.settings.pinecone_namespace,
            vector=vector,
            top_k=self.settings.top_k_dense,
            include_metadata=True,
        )
        hits = []
        for match in result.matches:
            metadata = dict(match.metadata or {})
            hits.append(
                SearchHit(
                    id=match.id,
                    text=str(metadata.get(self.settings.pinecone_text_field, "")),
                    source=str(metadata.get("source", "corpus")),
                    page=int(metadata.get("page", 0)),
                    score=float(match.score),
                    channels=["dense"],
                    metadata=metadata,
                )
            )
        return hits

    def sparse_search(self, query: str) -> list[SearchHit]:
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        query_terms = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        positive = [
            (idx, score)
            for idx, score in ranked
            if score > 0 and query_terms.intersection(tokenize(self.chunks[idx].text))
        ][: self.settings.top_k_sparse]
        max_score = positive[0][1] if positive else 1.0
        return [
            SearchHit(
                id=self.chunks[idx].id,
                text=self.chunks[idx].text,
                source=self.chunks[idx].source,
                page=self.chunks[idx].page,
                score=float(score / max_score),
                channels=["bm25"],
                metadata=self.chunks[idx].metadata,
            )
            for idx, score in positive
        ]

    def fuse(self, dense: list[SearchHit], sparse: list[SearchHit]) -> list[SearchHit]:
        scores: dict[str, float] = defaultdict(float)
        hits: dict[str, SearchHit] = {}
        channels: dict[str, set[str]] = defaultdict(set)
        channel_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for channel_hits in (dense, sparse):
            for rank, hit in enumerate(channel_hits, start=1):
                scores[hit.id] += 1 / (self.settings.rrf_k + rank)
                hits[hit.id] = hit
                channels[hit.id].update(hit.channels)
                for channel in hit.channels:
                    channel_scores[hit.id][f"{channel}_score"] = hit.score
        ordered = sorted(scores, key=scores.get, reverse=True)[: self.settings.top_k_final * 2]
        max_score = scores[ordered[0]] if ordered else 1.0
        return [
            SearchHit(
                id=hit_id,
                text=hits[hit_id].text,
                source=hits[hit_id].source,
                page=hits[hit_id].page,
                score=scores[hit_id] / max_score,
                channels=sorted(channels[hit_id]),
                metadata={**hits[hit_id].metadata, **channel_scores[hit_id]},
            )
            for hit_id in ordered
        ]

    def confidence(self, hits: list[SearchHit]) -> float:
        if not hits:
            return 0.0
        leaders = hits[:3]
        dense_score = max(float(hit.metadata.get("dense_score", 0)) for hit in leaders)
        sparse_score = max(float(hit.metadata.get("bm25_score", 0)) for hit in leaders)
        semantic = min(1.0, max(0.0, (dense_score - 0.25) / 0.55))
        lexical = min(1.0, max(0.0, sparse_score))
        agreement = 1.0 if any(len(hit.channels) > 1 for hit in leaders) else 0.0
        coverage = min(1.0, len(hits) / 3)
        confidence = 0.5 * semantic + 0.25 * lexical + 0.2 * agreement + 0.05 * coverage
        return round(min(1.0, confidence), 3)
