"""Re-ranking: a second, more expensive pass over the fused candidates — and the relevance floor.

A bi-encoder scores query and passage separately; a cross-encoder reads the pair together and is
far more accurate, which is why enterprise RAG re-ranks the top candidates instead of trusting
first-stage scores. `CrossEncoderReranker` is the production path. `LexicalReranker` scores the
fraction of the query's content words a passage contains — a real, blunt reranker that lets the
pipeline run offline.

Each reranker also states its `relevance_floor`: the score below which a passage is not about the
query. First-stage retrieval always returns k passages, even for a question the corpus never
covers; the floor is what lets the engine abstain instead of answering from the closest
irrelevant text. The floors are starting points to calibrate on your own data, not measurements.
"""

from __future__ import annotations

from typing import Any, Protocol

from entsearch import text
from entsearch.domain import Query, Retrieved


class Reranker(Protocol):
    name: str
    relevance_floor: float

    def rerank(self, query: Query, candidates: list[Retrieved]) -> list[Retrieved]: ...


class LexicalReranker:
    """Fraction of the query's content words (stemmed, stopwords and question words removed)
    that the passage contains. Ties keep first-stage order."""

    name = "lexical"
    relevance_floor = 0.34  # more than a third of the query's content words must be present

    def rerank(self, query: Query, candidates: list[Retrieved]) -> list[Retrieved]:
        q = set(text.content_stems(query.text))
        scored = [
            Retrieved(
                chunk=c.chunk,
                score=len(q & set(text.content_stems(c.chunk.indexed_text))) / len(q) if q else 0.0,
                source="rerank",
            )
            for c in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)  # stable: ties keep fused order
        return scored


class CrossEncoderReranker:
    """Production reranker: a trained cross-encoder (ms-marco MiniLM). Lazy; never run here.
    Its scores are logits; 0 is a starting floor — relevant pairs usually score well above it."""

    relevance_floor = 0.0

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.name = model
        self._model_name = model
        self._model: Any = None

    def rerank(self, query: Query, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        scores = self._model.predict([(query.text, c.chunk.indexed_text) for c in candidates])
        reranked = [
            Retrieved(chunk=c.chunk, score=float(s), source="rerank")
            for c, s in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
