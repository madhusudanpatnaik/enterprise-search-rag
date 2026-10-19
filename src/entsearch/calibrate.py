"""Abstention from a calibrated reranker score, instead of a hand-set floor.

A reranker's raw score is not a probability: a cross-encoder's logit or the lexical reranker's
word fraction says "more relevant than", not "relevant enough to answer from". So the decision to
answer is learned on held-out questions labelled answerable or not:

  1. run each question through retrieval and reranking, keep the top reranked score;
  2. fit P(answerable | score) = sigmoid(a·score + b) by Platt scaling — logistic regression on one
     feature, with Platt's smoothed targets so a small set does not produce certainties;
  3. answer only when that probability reaches a threshold set by what the mistakes cost: with
     answering an unanswerable question `cost` times as bad as abstaining on an answerable one,
     the decision rule that minimizes expected cost is P ≥ cost / (1 + cost).

Calibrate on questions that are not in the evaluation, and per reranker and corpus — a calibration
says what one reranker's scores mean on one corpus, so it records both, and `--calibration` refuses
one fitted for another reranker or corpus. `entsearch calibrate` fits one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from entsearch.pipeline import SearchEngine


@dataclass(frozen=True)
class Calibration:
    reranker: str
    a: float
    b: float
    threshold: float
    n_answerable: int
    n_unanswerable: int
    corpus: str = ""  # what it was fitted on: "synthetic", "docs:<file>", "beir:<dataset>"

    def probability(self, score: float) -> float:
        z = self.a * score + self.b
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    def answers(self, score: float) -> bool:
        return self.probability(score) >= self.threshold

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls(**json.loads(Path(path).read_text()))


def fit_platt(
    scores: Sequence[float], labels: Sequence[bool], iters: int = 100
) -> tuple[float, float]:
    """(a, b) for P(label | s) = sigmoid(a·s + b): Newton's method with backtracking on Platt's
    smoothed targets ((N+ + 1)/(N+ + 2) for positives, 1/(N- + 2) for negatives)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        raise ValueError("calibration needs both answerable and unanswerable questions")
    t = np.where(y, (pos + 1) / (pos + 2), 1 / (neg + 2))

    def loss(w: np.ndarray) -> float:
        f = w[0] * s + w[1]
        return float(np.sum(np.logaddexp(0, f) - t * f))  # cross-entropy against the targets

    w = np.array([0.0, math.log((pos + 1) / (neg + 1))])  # start at the prior log-odds
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(w[0] * s + w[1])))
        g = np.array([np.sum((p - t) * s), np.sum(p - t)])
        h = p * (1 - p)
        hess = np.array([[np.sum(h * s * s), np.sum(h * s)], [np.sum(h * s), np.sum(h)]])
        hess += np.eye(2) * 1e-12
        step = np.linalg.solve(hess, g)
        lr = 1.0
        while loss(w - lr * step) > loss(w) and lr > 1e-10:
            lr /= 2
        w = w - lr * step
        if np.max(np.abs(lr * step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def threshold_for(cost: float) -> float:
    """The probability above which answering has lower expected cost than abstaining, when a wrong
    answer costs `cost` times an unnecessary abstention."""
    if cost <= 0:
        raise ValueError("cost must be positive")
    return cost / (1 + cost)


def top_scores(engine: SearchEngine, questions: Sequence[str], candidates: int = 20) -> list[float]:
    """The top reranked score each question gets (0 when nothing is retrieved)."""
    out = []
    for q in questions:
        hits = engine.retrieve(q, mode="hybrid", rerank=True, k=1, candidates=candidates)
        out.append(hits[0].score if hits else 0.0)
    return out


def calibrate(
    engine: SearchEngine,
    answerable: Sequence[str],
    unanswerable: Sequence[str],
    *,
    cost: float = 1.0,
    candidates: int = 20,
    corpus: str = "",
) -> Calibration:
    scores = top_scores(engine, [*answerable, *unanswerable], candidates)
    labels = [True] * len(answerable) + [False] * len(unanswerable)
    a, b = fit_platt(scores, labels)
    return Calibration(
        reranker=engine.reranker.name,
        a=a,
        b=b,
        threshold=threshold_for(cost),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        corpus=corpus,
    )
