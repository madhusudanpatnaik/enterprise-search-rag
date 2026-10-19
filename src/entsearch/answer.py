"""Answer generation. One interface; an extractive baseline and an LLM generator.

`ExtractiveAnswerer` returns the retrieved sentences that cover the most of the query's content
words — grounded by construction, a real baseline and the offline default. `AnthropicGenerator` is
the production path: Claude sees only the retrieved passages, is told to cite them as [n] and to
answer with the exact abstention sentence when they do not contain the answer, and the citation
gate checks its output anyway.
"""

from __future__ import annotations

from typing import Any, Protocol

from entsearch import text
from entsearch.cite import ABSTAIN
from entsearch.domain import Query, Retrieved

_SYSTEM = f"""\
You answer questions using only the numbered passages you are given. Cite each passage you rely on
as [n]. Write each sentence of your answer in the words of one passage sentence: an automatic check
compares every sentence you write with the passages and marks claims it cannot find there as
unsupported. Keep numbers, dates and negations exactly as the passages state them. If the passages
do not contain the answer, reply with exactly this sentence and nothing else: "{ABSTAIN}" """


class Generator(Protocol):
    name: str

    def generate(self, query: Query, retrieved: list[Retrieved]) -> str: ...


class ExtractiveAnswerer:
    """The retrieved sentences covering the most of the query's content words."""

    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, query: Query, retrieved: list[Retrieved]) -> str:
        q = set(text.content_stems(query.text))
        ranked: list[tuple[float, int, str]] = []
        for rank, r in enumerate(retrieved):
            for sentence in text.sentences(r.chunk.text):
                if text.is_question(sentence):
                    continue  # an FAQ's question is not an answer
                coverage = len(q & set(text.content_stems(sentence))) / len(q) if q else 0.0
                if coverage > 0:
                    ranked.append((coverage, -rank, sentence))
        ranked.sort(reverse=True)  # most coverage first; ties go to the higher-ranked passage
        chosen: list[str] = []
        for _, _, sentence in ranked:
            if sentence not in chosen:
                chosen.append(sentence)
            if len(chosen) == self.max_sentences:
                break
        return " ".join(chosen) if chosen else ABSTAIN


class ModelRefused(RuntimeError):
    """stop_reason 'refusal' — after the server-side fallback also declined."""


class AnthropicGenerator:
    """Claude with the server-side refusal fallback. Credentials resolve the SDK's way
    (ANTHROPIC_API_KEY, an `ant auth login` profile, ...). Never run in this repository."""

    def __init__(
        self, model: str = "claude-opus-5-5", *, effort: str = "high", client: Any = None
    ) -> None:
        self.name = model
        self.model = model
        self.effort = effort
        self._client = client

    def _ensure(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, query: Query, retrieved: list[Retrieved]) -> str:
        passages = "\n".join(
            f"[{i + 1}] " + (f"({r.chunk.title}) " if r.chunk.title else "") + r.chunk.text
            for i, r in enumerate(retrieved)
        )
        response = self._ensure().beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            system=_SYSTEM,
            messages=[
                {"role": "user", "content": f"Question: {query.text}\n\nPassages:\n{passages}"}
            ],
        )
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise ModelRefused(f"the generator declined (category: {category})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("answer truncated at max_tokens")
        return "".join(block.text for block in response.content if block.type == "text")
