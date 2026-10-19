"""Evaluation that can fail: retrieval, answering, abstention, the gate, and latency — separately.

  - **Retrieval** is scored on its own (hit@k, recall@k, MRR, nDCG@k) and ablated across dense,
    BM25, hybrid and hybrid+rerank, so a claim like "fusion helps" is a measured row, not prose.
  - **Answering** is split by whether the corpus covers the question: on answerable questions the
    engine should answer; on unanswerable ones the right behaviour is to abstain.
  - **The gate** is scored on claims known to be true and claims known to be false.
  - **Latency** percentiles are reported with their sample size; below ~20 samples a p95/p99 is
    an interpolation toward the maximum, not a tail measurement.

Cases are judged per chunk, or per document when the benchmark judges documents (BEIR qrels):
documents are then ranked by their best chunk. Cases with no gold at all (a gold phrase that
straddled a chunk boundary) are skipped and counted, never averaged in as a retrieval failure.
MRR is MRR@k, like the other metrics.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from entsearch import cite
from entsearch.domain import QueryCase
from entsearch.pipeline import Mode, SearchEngine

# Roughly 1 / (1 - p) samples before a percentile is more than an interpolation toward the max.
RELIABLE_N = {50: 2, 95: 20, 99: 100}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(values, p)), 3)


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined for an empty gold set")
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: list[str], relevant: set[str], k: int, grades: dict[str, int] | None = None
) -> float:
    """nDCG@k with linear gains, as trec_eval and BEIR compute it: a graded judgement's gain is
    its grade (0/1/2...), a binary one's is 1."""
    gain = {r: (grades or {}).get(r, 1) for r in relevant}
    dcg = sum(gain.get(cid, 0) / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]))
    best = sorted(gain.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(best))
    return dcg / ideal if ideal else 0.0


class RetrievalScore(BaseModel):
    setup: str
    queries: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def ranking(
    engine: SearchEngine, case: QueryCase, *, mode: Mode, rerank: bool, k: int
) -> tuple[list[str], set[str]]:
    """The top-k ranked units and the gold set, at the level the case is judged:

    - **documents** (BEIR qrels): documents ranked by their best chunk, retrieving deeper until k
      distinct documents are available (or the corpus is exhausted);
    - **gold spans** (a case with answer spans): a chunk counts for the first span it contains
      that no higher chunk already supplied — overlapping chunks holding the same sentence are one
      relevant unit, not two;
    - otherwise **chunks**."""
    if case.relevant_doc_ids:
        depth = max(20, 4 * k)
        while True:  # deepen until k distinct documents (long documents fill the top with chunks)
            hits = engine.retrieve(case.query.text, mode=mode, rerank=rerank, k=depth,
                                   candidates=depth)  # fmt: skip
            docs = list(dict.fromkeys(h.chunk.doc_id for h in hits))
            if len(docs) >= k or depth >= len(engine.chunks):
                return docs[:k], case.relevant_doc_ids
            depth *= 2
    hits = engine.retrieve(case.query.text, mode=mode, rerank=rerank, k=k)
    if case.answer_spans:
        spans = [s.lower() for s in case.answer_spans]
        seen: set[int] = set()
        units = []
        for h in hits:
            new = [i for i, s in enumerate(spans) if s in h.chunk.text.lower() and i not in seen]
            units.append(f"span:{new[0]}" if new else f"chunk:{h.chunk.id}")
            seen.update(new[:1])
        return units, {f"span:{i}" for i in range(len(spans))}
    return [h.chunk.id for h in hits], case.relevant_chunk_ids


def _has_gold(case: QueryCase) -> bool:
    return bool(case.relevant_doc_ids or case.relevant_chunk_ids)


def score_retrieval(
    engine: SearchEngine, cases: list[QueryCase], *, mode: Mode, rerank: bool, k: int
) -> RetrievalScore:
    scored = [c for c in cases if c.answerable and _has_gold(c)]
    hits, recalls, mrrs, ndcgs = [], [], [], []
    for case in scored:
        ranked, gold = ranking(engine, case, mode=mode, rerank=rerank, k=k)
        hits.append(hit_at_k(ranked, gold, k))
        recalls.append(recall_at_k(ranked, gold, k))
        mrrs.append(mrr(ranked[:k], gold))
        ndcgs.append(ndcg_at_k(ranked, gold, k, case.grades or None))
    n = len(scored) or 1
    return RetrievalScore(
        setup=f"{mode}{'+rerank' if rerank else ''}",
        queries=len(scored),
        hit_at_k=round(sum(hits) / n, 3),
        recall_at_k=round(sum(recalls) / n, 3),
        mrr=round(sum(mrrs) / n, 3),
        ndcg_at_k=round(sum(ndcgs) / n, 3),
    )


def ablation(engine: SearchEngine, cases: list[QueryCase], *, k: int = 3) -> list[RetrievalScore]:
    setups: list[tuple[Mode, bool]] = [
        ("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True),
    ]  # fmt: skip
    return [score_retrieval(engine, cases, mode=m, rerank=r, k=k) for m, r in setups]


class Rate(BaseModel):
    """A rate with its counts. `value` is None when there is nothing to divide by — "n/a", never
    a 0% that reads like a measurement."""

    hits: int
    of: int

    @property
    def value(self) -> float | None:
        return round(self.hits / self.of, 3) if self.of else None

    def __str__(self) -> str:
        return f"{self.value:.3f} ({self.hits}/{self.of})" if self.of else "n/a (0 cases)"


def _contains_gold(answer: str, spans: list[str]) -> bool:
    flat = " ".join(answer.lower().split())
    return any(" ".join(s.lower().split()) in flat for s in spans)


class EvalReport(BaseModel):
    k: int
    answerable: int
    unanswerable: int
    skipped_no_gold: int
    retrieval: RetrievalScore
    answered: Rate  # answerable questions the engine answered (did not abstain on)
    correct: Rate  # answered answerable questions whose delivered answer contains a gold span
    abstained: Rate  # unanswerable questions it abstained on
    unsupported_drafts: Rate  # answers whose draft carried a claim the gate refused
    cited: Rate  # answers with at least one verified citation
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_n: int

    @property
    def latency_unreliable(self) -> list[str]:
        return [f"p{p}" for p, need in RELIABLE_N.items() if self.latency_n < need]


def evaluate(engine: SearchEngine, cases: list[QueryCase], *, k: int = 3) -> EvalReport:
    """Answer-level evaluation. "Correct" needs gold answer spans (the synthetic set and JSONL
    cases carry them): the delivered answer must contain one. Groundedness alone is not
    correctness — an extractive answer is grounded by construction, right or wrong."""
    answerable = [c for c in cases if c.answerable]
    unanswerable = [c for c in cases if not c.answerable]
    latencies: list[float] = []
    answered = correct = with_spans = unsupported = cited = abstained_right = 0
    for case in cases:
        answer = engine.search(case.query.text, k=k, query_id=case.query.id)
        latencies.append(answer.timings_ms.get("total", 0.0))
        if case.answerable:
            if not answer.abstained:
                answered += 1
                unsupported += 0 if answer.supported else 1
                cited += 1 if answer.citations else 0
                if case.answer_spans:
                    with_spans += 1
                    correct += _contains_gold(answer.text, case.answer_spans)
        elif answer.abstained:
            abstained_right += 1
    return EvalReport(
        k=k,
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        skipped_no_gold=sum(1 for c in answerable if not _has_gold(c)),
        retrieval=score_retrieval(engine, cases, mode="hybrid", rerank=True, k=k),
        answered=Rate(hits=answered, of=len(answerable)),
        correct=Rate(hits=correct, of=with_spans),
        abstained=Rate(hits=abstained_right, of=len(unanswerable)),
        unsupported_drafts=Rate(hits=unsupported, of=answered),
        cited=Rate(hits=cited, of=answered),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        latency_n=len(latencies),
    )


class GateCase(BaseModel):
    claim: str
    true: bool  # does the corpus actually support it?
    note: str = ""


class GateOutcome(BaseModel):
    case: GateCase
    accepted: bool
    reason: str = ""

    @property
    def correct(self) -> bool:
        return self.accepted == self.case.true


def gate_eval(engine: SearchEngine, cases: list[GateCase], *, k: int = 3) -> list[GateOutcome]:
    """Run each claim through the gate against the passages retrieved for it."""
    outcomes = []
    for case in cases:
        result = cite.enforce(case.claim, engine.retrieve(case.claim, k=k), engine.verifier)
        accepted = not result.unsupported and not result.abstained
        outcomes.append(
            GateOutcome(case=case, accepted=accepted, reason="; ".join(result.unsupported))
        )
    return outcomes
