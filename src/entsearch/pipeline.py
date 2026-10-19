"""The engine: index a corpus, then answer with grounded citations — or abstain.

    chunk → index (dense + BM25) ─► hybrid retrieve (RRF) ─► rerank ─► relevance floor ─┐
                                                                                         │
       answer + citations ◄─ CITE-GATE ◄─ generate ◄─────────── relevant passages ◄──────┤
       abstention ◄────────────────────────────────────────────── none relevant ◄────────┘

`retrieve()` exposes each first-stage mode on its own (dense, BM25, hybrid; with or without
reranking), so retrieval can be evaluated — and ablated — separately from answering.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from typing import Literal

from entsearch import cite
from entsearch.answer import ExtractiveAnswerer, Generator
from entsearch.calibrate import Calibration
from entsearch.chunk import Strategy, chunk_document
from entsearch.domain import Answer, Chunk, Document, Query, Retrieved
from entsearch.encode import Encoder, HashingEncoder
from entsearch.index import Bm25Index, DenseIndex
from entsearch.rerank import LexicalReranker, Reranker
from entsearch.retrieve import hybrid_search

Mode = Literal["dense", "bm25", "hybrid"]


class SearchEngine:
    def __init__(
        self,
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        verifier: cite.Verifier | None = None,
        *,
        strategy: Strategy = Strategy.SENTENCE,
        chunk_size: int = 60,
        overlap: int | None = None,
        min_relevance: float | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        self.encoder: Encoder = encoder or HashingEncoder()
        self.reranker: Reranker = reranker or LexicalReranker()
        self.generator: Generator = generator or ExtractiveAnswerer()
        self.verifier: cite.Verifier = verifier or cite.LexicalVerifier()
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_relevance = (
            self.reranker.relevance_floor if min_relevance is None else min_relevance
        )
        if calibration is not None and calibration.reranker != self.reranker.name:
            raise ValueError(
                f"calibration was fitted for {calibration.reranker!r}, not {self.reranker.name!r}"
            )
        # with a calibration, a passage is relevant when its score's P(answerable) clears the
        # threshold; without one, when its raw score clears the reranker's floor
        self.calibration = calibration
        self.chunks: list[Chunk] = []
        self.chunks_by_id: dict[str, Chunk] = {}
        self._dense = DenseIndex()
        self._bm25 = Bm25Index()

    def index(self, documents: list[Document]) -> None:
        duplicates = sorted(d for d, n in Counter(doc.id for doc in documents).items() if n > 1)
        if duplicates:
            # A repeated id would make two chunks share one id: the index would score one text and
            # return the other. Fail loudly rather than serve the wrong passage.
            raise ValueError(f"duplicate document id(s): {', '.join(duplicates)}")
        self.chunks = [
            c
            for doc in documents
            for c in chunk_document(doc, self.strategy, size=self.chunk_size, overlap=self.overlap)
        ]
        self.chunks_by_id = {c.id: c for c in self.chunks}
        self._dense.build(self.chunks, self.encoder)
        self._bm25.build(self.chunks)

    def retrieve(
        self,
        query_text: str,
        *,
        mode: Mode = "hybrid",
        rerank: bool = True,
        k: int = 5,
        candidates: int = 20,
    ) -> list[Retrieved]:
        hits = self._first_stage(query_text, mode, candidates)
        if rerank:
            hits = self.reranker.rerank(Query(id="q", text=query_text), hits)
        return hits[:k]

    def _first_stage(self, query_text: str, mode: Mode, candidates: int) -> list[Retrieved]:
        if mode == "hybrid":
            return hybrid_search(
                self._dense, self._bm25, self.encoder, self.chunks_by_id, query_text,
                k_dense=candidates, k_bm25=candidates, k_out=candidates,
            )  # fmt: skip
        scored = (
            self._dense.search_text(self.encoder, query_text, candidates)
            if mode == "dense"
            else self._bm25.search(query_text, candidates)
        )
        return [Retrieved(chunk=self.chunks_by_id[s.chunk_id], score=s.score, source=mode)
                for s in scored]  # fmt: skip

    def search(
        self, query_text: str, *, k: int = 5, candidates: int = 20, query_id: str | None = None
    ) -> Answer:
        qid = query_id or hashlib.blake2b(query_text.encode(), digest_size=6).hexdigest()
        query = Query(id=qid, text=query_text)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        hits = self._first_stage(query_text, "hybrid", candidates)
        timings["retrieve"] = _ms(t0)

        t0 = time.perf_counter()
        ranked = self.reranker.rerank(query, hits)
        cal = self.calibration
        relevant = [
            r for r in ranked if (cal.answers(r.score) if cal else r.score >= self.min_relevance)
        ][:k]
        timings["rerank"] = _ms(t0)

        if not relevant:
            timings["total"] = round(sum(timings.values()), 3)
            return Answer(
                query_id=qid,
                text=cite.ABSTAIN,
                abstained=True,
                refusal_reason="no retrieved passage is relevant enough to answer from",
                timings_ms=timings,
            )

        t0 = time.perf_counter()
        answer_text = self.generator.generate(query, relevant)
        timings["generate"] = _ms(t0)

        t0 = time.perf_counter()
        gate = cite.enforce(answer_text, relevant, self.verifier)
        timings["cite"] = _ms(t0)
        timings["total"] = round(sum(timings.values()), 3)

        # The user gets only what the gate verified. An unsupported claim is dropped, not shown
        # with a warning; if nothing survives, the engine abstains.
        verified = " ".join(c.quote for c in gate.citations)
        abstained = gate.abstained or not verified
        return Answer(
            query_id=qid,
            text=cite.ABSTAIN if abstained else verified,
            draft=answer_text,
            citations=gate.citations,
            supported=not gate.unsupported,  # was the draft fully supported?
            abstained=abstained,
            refusal_reason="; ".join(gate.unsupported),
            retrieved=relevant,
            timings_ms=timings,
        )


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000.0, 3)
