"""Chunking strategies — the first lever the talk calls out.

Three strategies, each returning `Chunk`s whose `start`/`end` are exact character offsets
(`doc.text[chunk.start:chunk.end] == chunk.text`), so a citation can point at the span it used:

  - **fixed**: windows of `size` words overlapping by `overlap` words (default `size // 5`).
  - **sentence**: whole sentences packed up to `size` words, overlapping by `overlap` sentences
    (default 1). Keeps claims intact, which is what the citation gate needs.
  - **recursive**: paragraphs, then sentences, then words — the LangChain-style splitter.

Every strategy guarantees **no chunk exceeds `size` words**: a sentence longer than `size` (a log
line, a table row, unpunctuated text) is split into word windows rather than emitted whole and
silently truncated by an encoder with a token limit.

Size is counted in words, a deliberate approximation: a model-exact tokenizer is
provider-specific. Swap in the model's tokenizer when you pin a model.
"""

from __future__ import annotations

import re
from enum import StrEnum

from entsearch import text
from entsearch.domain import Chunk, Document

_WORD = re.compile(r"\S+")
_PARAGRAPH = re.compile(r"\n\s*\n")

Span = tuple[int, int]


class Strategy(StrEnum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"


def chunk_document(
    doc: Document,
    strategy: Strategy = Strategy.SENTENCE,
    *,
    size: int = 120,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk one document. `overlap` is in words for FIXED, in sentences for SENTENCE."""
    if size < 1:
        raise ValueError("size must be >= 1 word")
    if strategy is Strategy.FIXED:
        ov = size // 5 if overlap is None else overlap
        if not 0 <= ov < size:
            raise ValueError("FIXED overlap must be in [0, size) words")
        spans = _word_windows(doc.text, 0, len(doc.text), size=size, overlap=ov)
    elif strategy is Strategy.SENTENCE:
        ov = 1 if overlap is None else overlap
        if ov < 0:
            raise ValueError("SENTENCE overlap must be >= 0 sentences")
        spans = _pack(_units(doc.text, 0, len(doc.text), size), doc.text, size, overlap=ov)
    else:
        spans = _recursive(doc.text, size)
    chunks: list[Chunk] = []
    for s, e in spans:
        s, e = _trim(doc.text, s, e)
        if s < e:
            chunks.append(Chunk.make(doc.id, len(chunks), doc.text[s:e], s, e, doc.title))
    return chunks


def _word_windows(t: str, lo: int, hi: int, *, size: int, overlap: int) -> list[Span]:
    words = list(_WORD.finditer(t, lo, hi))
    spans: list[Span] = []
    step = size - overlap
    for i in range(0, len(words), step):
        window = words[i : i + size]
        spans.append((window[0].start(), window[-1].end()))
        if i + size >= len(words):
            break
    return spans


def _units(t: str, lo: int, hi: int, size: int) -> list[Span]:
    """Sentences of t[lo:hi], with any sentence longer than `size` words split into windows."""
    units: list[Span] = []
    for s, e in text.sentence_spans(t[lo:hi]):
        s, e = s + lo, e + lo
        if _count(t, s, e) > size:
            units.extend(_word_windows(t, s, e, size=size, overlap=0))
        else:
            units.append((s, e))
    return units


def _pack(units: list[Span], t: str, size: int, *, overlap: int) -> list[Span]:
    """Greedily pack consecutive units up to `size` words; the next chunk re-starts `overlap`
    units back, but always moves forward by at least one."""
    spans: list[Span] = []
    i = 0
    while i < len(units):
        j = i
        words = 0
        while j < len(units) and (j == i or words + _count(t, *units[j]) <= size):
            words += _count(t, *units[j])
            j += 1
        spans.append((units[i][0], units[j - 1][1]))
        if j >= len(units):
            break
        i = max(i + 1, j - overlap)
    return spans


def _recursive(t: str, size: int) -> list[Span]:
    units: list[Span] = []
    for s, e in _paragraphs(t):
        units.extend([(s, e)] if _count(t, s, e) <= size else _units(t, s, e, size))
    return _pack(units, t, size, overlap=0)


def _paragraphs(t: str) -> list[Span]:
    spans: list[Span] = []
    last = 0
    for m in _PARAGRAPH.finditer(t):
        if t[last : m.start()].strip():
            spans.append((last, m.start()))
        last = m.end()
    if t[last:].strip():
        spans.append((last, len(t)))
    return spans


def _trim(t: str, s: int, e: int) -> Span:
    while s < e and t[s].isspace():
        s += 1
    while e > s and t[e - 1].isspace():
        e -= 1
    return s, e


def _count(t: str, s: int, e: int) -> int:
    return len(_WORD.findall(t, s, e))
