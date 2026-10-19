"""THE GATE: no claim without a retrieved sentence that says it.

Every sentence of an answer is a claim, and each must be supported by a retrieved passage or the
answer is marked unsupported. "Supported" is decided by a `Verifier`:

  - `LexicalVerifier` (offline default) checks the claim against retrieved *sentences*, one at a
    time — never words pooled across a chunk, which let "rolls over up to 20 days" borrow its 20
    from "accrue 20 days" two sentences away. A sentence supports the claim when it (read with its
    document's title, so "Its certificates ..." has a subject) contains every content word of the
    claim, states every number the claim states with the same polarity, and has the same polarity
    on every shared word, negation scoped to its clause. Questions in a passage (an FAQ's "Can
    contractors take PTO?") are not evidence. `min_support` below 1.0 trades faithfulness for
    recall — one swapped word in five gets through at 0.8 — and is exposed, not hidden.
  - `MiniCheckVerifier` (production, `--verify minicheck`) asks MiniCheck, a model trained for
    exactly this check, whether a passage supports the claim; `NliVerifier` (`--verify nli`) asks
    a general entailment cross-encoder. Paraphrase needs one of them: the gate evaluation shows
    the lexical verifier refusing a true paraphrase.

Answer-level rules: citation markers and meta-phrasing ("according to the context") are not
content; a sentence with no content words is not a claim; an empty answer is unsupported; and the
abstention the generator is told to use is an abstention, not an ungrounded claim.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from entsearch import text
from entsearch.domain import Citation, Retrieved

ABSTAIN = "The documents do not cover this."


class GateResult(BaseModel):
    citations: list[Citation] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)  # "claim — reason"
    claims: int = 0
    abstained: bool = False


class Verifier(Protocol):
    name: str

    def verify(self, claim: str, retrieved: list[Retrieved]) -> tuple[str | None, str | None]:
        """(supporting chunk id, None) or (None, the reason the claim is unsupported)."""
        ...


_LEADING_PRONOUN = re.compile(r"^\s*(It|Its|They|Their|This|These|Those)\b")


def with_subject(sentence: str, title: str) -> str:
    """A sentence that opens with a pronoun ("Its certificates were rotated ...") is read with the
    document's title as its subject. Any other sentence is read as written: the title's words are
    never pooled in, or "Expense Policy" would let "First-class flights are not reimbursed" support
    "Expenses are not reimbursed"."""
    if title and (m := _LEADING_PRONOUN.match(sentence)):
        sub = f"{title}'s" if m.group(1) in ("Its", "Their") else title
        return sentence[: m.start(1)] + sub + sentence[m.end(1) :]
    return sentence


class LexicalVerifier:
    """One retrieved sentence must say the claim: `text.compare` clause by clause — every content
    word with the same negation and hypothetical state, every number by value, polarity and the
    word it is attached to. Questions in a passage are not evidence."""

    name = "lexical"

    def __init__(self, min_support: float = 1.0) -> None:
        if not 0 < min_support <= 1:
            raise ValueError("min_support must be in (0, 1]")
        self.min_support = min_support

    def verify(self, claim: str, retrieved: list[Retrieved]) -> tuple[str | None, str | None]:
        if not text.content_stems(claim) and not text.numbers(claim):
            return None, "states nothing checkable"
        candidates = [
            (r.chunk.id, with_subject(s, r.chunk.title))
            for r in retrieved
            for s in text.sentences(r.chunk.text)
            if not text.is_question(s)
        ]
        if not candidates:
            return None, "no retrieved passage states anything"
        best: tuple[float, str] | None = None
        for chunk_id, sentence in candidates:
            result = text.compare(claim, sentence, min_support=self.min_support)
            problem = result.problem(self.min_support)
            if problem is None:
                return chunk_id, None
            if best is None or result.coverage > best[0]:
                best = (result.coverage, problem)
        assert best is not None
        return None, best[1]


class NliVerifier:
    """Entailment with an NLI cross-encoder (default nli-deberta-v3-base). Lazy; never run here.
    Label order is read from the model config when it exposes one."""

    def __init__(
        self, model: str = "cross-encoder/nli-deberta-v3-base", *, min_entailment: float = 0.5
    ) -> None:
        self.name = model
        self._model_name = model
        self.min_entailment = min_entailment
        self._model: Any = None
        self._labels = ["contradiction", "entailment", "neutral"]  # the model card's order

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            id2label = getattr(getattr(self._model, "config", None), "id2label", None)
            if id2label:
                self._labels = [str(id2label[i]).lower() for i in range(len(id2label))]
        return self._model

    def verify(self, claim: str, retrieved: list[Retrieved]) -> tuple[str | None, str | None]:
        if not retrieved:
            return None, "no retrieved passage"
        logits = np.asarray(
            self._ensure().predict([(r.chunk.text, claim) for r in retrieved]), dtype=np.float64
        )
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        ent, con = self._labels.index("entailment"), self._labels.index("contradiction")
        i = int(np.argmax(probs[:, ent]))
        if probs[i, ent] >= self.min_entailment and probs[i, ent] > probs[i, con]:
            return retrieved[i].chunk.id, None
        return None, f"not entailed by any passage (best entailment {probs[i, ent]:.2f})"


class MiniCheckVerifier:
    """MiniCheck (Tang, Laban and Durrett, EMNLP 2024): a model trained to say whether a document
    supports a sentence. MiniCheck-Flan-T5-Large (770M) reaches GPT-4's accuracy on LLM-AggreFact
    at a fraction of the cost — the grounding checker the literature backs, where the lexical gate
    refuses every paraphrase and a general NLI model was not trained for the task.

    Each claim is scored against each retrieved passage (with a leading pronoun read as the
    passage's title, as the lexical gate does); the best-supporting passage is cited if MiniCheck's
    probability reaches `threshold`. Needs `requirements-minicheck.txt`; lazy, never run here."""

    def __init__(
        self,
        model_name: str = "flan-t5-large",
        *,
        threshold: float = 0.5,
        cache_dir: str | None = None,
        scorer: Any | None = None,
    ) -> None:
        self.name = f"minicheck-{model_name}"
        self.model_name = model_name
        self.threshold = threshold
        self.cache_dir = cache_dir
        self._scorer = scorer

    def _ensure(self) -> Any:
        if self._scorer is None:
            from minicheck.minicheck import MiniCheck

            kwargs: dict[str, Any] = {"model_name": self.model_name}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            self._scorer = MiniCheck(**kwargs)
        return self._scorer

    def verify(self, claim: str, retrieved: list[Retrieved]) -> tuple[str | None, str | None]:
        if not retrieved:
            return None, "no retrieved passage"
        docs = [with_subject(r.chunk.text, r.chunk.title) for r in retrieved]
        _labels, probs, _, _ = self._ensure().score(docs=docs, claims=[claim] * len(docs))
        i = int(np.argmax(probs))
        if float(probs[i]) >= self.threshold:
            return retrieved[i].chunk.id, None
        return None, f"no passage supports it (best MiniCheck support {float(probs[i]):.2f})"


def is_abstention(answer: str) -> bool:
    return text.words(answer) == text.words(ABSTAIN)


_PRONOUN_CLAIM = re.compile(r"^\s*(?:it|its|they|their|this|these|those)\b", re.I)


def enforce(answer: str, retrieved: list[Retrieved], verifier: Verifier) -> GateResult:
    if not answer.strip():
        return GateResult(unsupported=["(empty answer)"])
    if is_abstention(answer):
        return GateResult(abstained=True)
    result = GateResult()
    doc_of = {r.chunk.id: r.chunk.doc_id for r in retrieved}
    previous_doc: str | None = None
    for sentence in text.sentences(text.strip_markers(answer)):
        if not text.content_stems(sentence):
            continue  # a connective or a stray marker, not a claim
        result.claims += 1
        pool = retrieved
        if previous_doc is not None and _PRONOUN_CLAIM.match(sentence):
            # "Its certificates were rotated every 365 days" continues the previous claim: it must
            # be true of the document that claim came from, not of whichever one says it
            pool = [r for r in retrieved if r.chunk.doc_id == previous_doc]
        chunk_id, reason = verifier.verify(sentence, pool)
        if chunk_id is not None:
            result.citations.append(Citation(chunk_id=chunk_id, quote=sentence))
            previous_doc = doc_of[chunk_id]
        else:
            result.unsupported.append(f"{sentence} — {reason}")
    if result.claims == 0:
        result.unsupported.append("(no checkable claim)")
    return result
