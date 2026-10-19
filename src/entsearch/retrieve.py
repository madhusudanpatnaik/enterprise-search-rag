"""Hybrid retrieval: run both indexes and fuse with Reciprocal Rank Fusion.

RRF beats score-normalization tricks because it needs no calibration between a cosine score and a
BM25 score — it fuses *ranks*, not magnitudes. A chunk that both retrievers rank highly wins; a
chunk only one finds still gets a chance. `c` damps the contribution of deep ranks (the standard
value is 60).
"""

from __future__ import annotations

from entsearch.domain import Chunk, Retrieved, Scored
from entsearch.encode import Encoder
from entsearch.index import Bm25Index, DenseIndex


def rrf_fuse(rankings: list[list[Scored]], k_out: int, *, c: int = 60) -> list[Scored]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            fused[scored.chunk_id] = fused.get(scored.chunk_id, 0.0) + 1.0 / (c + rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [Scored(chunk_id=cid, score=score) for cid, score in ordered[:k_out]]


def hybrid_search(
    dense: DenseIndex,
    bm25: Bm25Index,
    encoder: Encoder,
    chunks_by_id: dict[str, Chunk],
    query: str,
    *,
    k_dense: int = 20,
    k_bm25: int = 20,
    k_out: int = 10,
) -> list[Retrieved]:
    dense_hits = dense.search_text(encoder, query, k_dense)
    bm25_hits = bm25.search(query, k_bm25)
    fused = rrf_fuse([dense_hits, bm25_hits], k_out)
    return [
        Retrieved(chunk=chunks_by_id[s.chunk_id], score=s.score, source="hybrid")
        for s in fused
        if s.chunk_id in chunks_by_id
    ]
