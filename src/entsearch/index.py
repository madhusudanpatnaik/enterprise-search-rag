"""The two indexes: dense cosine (semantic) and BM25 (lexical).

`DenseIndex` is exact cosine over an L2-normalized matrix (a dot product), which is correct and
fast at enterprise-document scale; millions of chunks would want an approximate index (faiss).
`Bm25Index` is Okapi BM25 with Lucene's always-positive idf, written out — idf, term-frequency
saturation (k1) and length normalization (b) — not a library call, so the ranking is inspectable.
"""

from __future__ import annotations

import math
import re

import numpy as np
import numpy.typing as npt

from entsearch.domain import Chunk, Scored
from entsearch.encode import Encoder

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class DenseIndex:
    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self._matrix: npt.NDArray[np.float32] | None = None

    def build(self, chunks: list[Chunk], encoder: Encoder) -> None:
        self.chunk_ids = [c.id for c in chunks]
        self._matrix = encoder.encode([c.indexed_text for c in chunks]) if chunks else None

    def search(self, query_vector: npt.NDArray[np.float32], k: int) -> list[Scored]:
        if self._matrix is None or not self.chunk_ids:
            return []
        scores = self._matrix @ query_vector.reshape(-1)
        k = min(k, len(self.chunk_ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Scored(chunk_id=self.chunk_ids[i], score=float(scores[i])) for i in top]

    def search_text(self, encoder: Encoder, query: str, k: int) -> list[Scored]:
        return self.search(encoder.encode_queries([query])[0], k)


class Bm25Index:
    """Okapi BM25, Lucene idf. k1 saturates term frequency; b controls length normalization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self._tf: list[dict[str, int]] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def build(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.id for c in chunks]
        self._tf = []
        self._doc_len = []
        df: dict[str, int] = {}
        for chunk in chunks:
            tokens = tokenize(chunk.indexed_text)
            self._doc_len.append(len(tokens))
            freqs: dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self._tf.append(freqs)
            for term in freqs:
                df[term] = df.get(term, 0) + 1
        n = len(chunks)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        # Lucene's BM25 idf, log(1 + (N - df + 0.5) / (df + 0.5)). Unlike the classic Robertson
        # idf it is always positive, so a term in more than half the chunks still counts a little
        # instead of counting against the chunk.
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query: str, k: int) -> list[Scored]:
        if not self.chunk_ids:
            return []
        q_terms = tokenize(query)
        scores: list[Scored] = []
        for i, chunk_id in enumerate(self.chunk_ids):
            freqs = self._tf[i]
            dl = self._doc_len[i]
            length_norm = (1 - self.b + self.b * dl / self._avgdl) if self._avgdl else 1.0
            denom_norm = self.k1 * length_norm
            score = 0.0
            for term in q_terms:
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                score += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
            if score > 0:
                scores.append(Scored(chunk_id=chunk_id, score=score))
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[:k]
