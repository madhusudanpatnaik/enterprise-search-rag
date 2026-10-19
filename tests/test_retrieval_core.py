from __future__ import annotations

import numpy as np
import pytest

from entsearch import text
from entsearch.chunk import Strategy, chunk_document
from entsearch.domain import Chunk, Document, Scored
from entsearch.encode import HashingEncoder
from entsearch.index import Bm25Index, DenseIndex
from entsearch.pipeline import SearchEngine
from entsearch.retrieve import rrf_fuse

LONG = Document(
    id="d1",
    title="Handbook",
    text=(
        "The corporate VPN uses WireGuard. Certificates rotate every ninety days.\n\n"
        "Expense reimbursements are paid within thirty days. Receipts cost 3.5 dollars to file.\n\n"
        "Full-time staff accrue twenty days of paid time off each year."
    ),
)
RUN_ON = Document(id="log", text=" ".join(f"token{i}" for i in range(50)))  # no sentence breaks


def _words(c: Chunk) -> int:
    return len(c.text.split())


@pytest.mark.parametrize("strategy", list(Strategy))
@pytest.mark.parametrize("doc", [LONG, RUN_ON])
def test_chunks_are_exact_spans_and_never_exceed_size(strategy: Strategy, doc: Document) -> None:
    chunks = chunk_document(doc, strategy, size=12)
    assert chunks
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text
        assert c.id == f"{c.doc_id}:{c.index}" and c.title == doc.title
        assert _words(c) <= 12, (strategy, c.text)


def test_every_word_lands_in_some_chunk() -> None:
    for strategy in Strategy:
        covered = " ".join(c.text for c in chunk_document(LONG, strategy, size=8)).split()
        assert set(LONG.text.split()) <= set(covered), strategy


def test_fixed_windows_overlap_by_the_requested_words() -> None:
    a, b = chunk_document(RUN_ON, Strategy.FIXED, size=10, overlap=3)[:2]
    assert a.text.split()[-3:] == b.text.split()[:3]
    with pytest.raises(ValueError):
        chunk_document(RUN_ON, Strategy.FIXED, size=10, overlap=10)


def test_sentence_chunks_overlap_by_whole_sentences_and_keep_decimals() -> None:
    chunks = chunk_document(LONG, Strategy.SENTENCE, size=14, overlap=1)
    assert any("3.5 dollars" in c.text for c in chunks)  # "3.5" is not a sentence end
    for a, b in zip(chunks, chunks[1:], strict=False):
        if len(text.sentences(a.text)) > 1:  # a one-sentence chunk must still move forward
            assert text.sentences(b.text)[0] == text.sentences(a.text)[-1]


def test_hashing_encoder_returns_normalized_vectors() -> None:
    vecs = HashingEncoder(dim=256).encode(["hello world", "another sentence here"])
    assert vecs.shape == (2, 256)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_dense_index_finds_the_matching_chunk_first() -> None:
    chunks = chunk_document(LONG, Strategy.SENTENCE, size=12)
    enc = HashingEncoder()
    index = DenseIndex()
    index.build(chunks, enc)
    target = next(c for c in chunks if "WireGuard" in c.text)
    assert index.search_text(enc, target.indexed_text, k=3)[0].chunk_id == target.id


def test_bm25_ranks_the_chunk_with_the_query_terms_and_keeps_common_terms_positive() -> None:
    chunks = chunk_document(LONG, Strategy.SENTENCE, size=12)
    index = Bm25Index()
    index.build(chunks)
    hits = index.search("WireGuard certificates rotate", k=3)
    assert "WireGuard" in next(c for c in chunks if c.id == hits[0].chunk_id).text
    # "handbook" (the title) is in every chunk: Lucene's idf keeps it slightly positive, where
    # the classic Robertson idf would make matching it count *against* a chunk.
    assert all(h.score > 0 for h in index.search("handbook", k=10))


def test_rrf_prefers_what_both_rankers_rank_high() -> None:
    dense = [Scored(chunk_id="a", score=0.9), Scored(chunk_id="b", score=0.8)]
    bm25 = [Scored(chunk_id="b", score=5.0), Scored(chunk_id="c", score=4.0)]
    assert rrf_fuse([dense, bm25], k_out=3)[0].chunk_id == "b"  # top-2 in both


def test_the_title_makes_a_subjectless_chunk_findable() -> None:
    doc = Document(
        id="legacy",
        title="Legacy OpenVPN service",
        text="Its certificates were rotated every 365 days.",
    )
    other = Document(id="cafe", title="Cafeteria", text="Lunch is served from noon.")
    engine = SearchEngine()
    engine.index([doc, other])
    top = engine.retrieve("OpenVPN certificate rotation", mode="bm25", rerank=False, k=1)
    assert top[0].chunk.doc_id == "legacy"


def test_duplicate_document_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate document id"):
        SearchEngine().index([LONG, LONG.model_copy(update={"text": "different"})])


# --------------------------------------------------------------------------- hand-computed scores


def _bm25(texts: list[str]) -> tuple[Bm25Index, list[Chunk]]:
    chunks = [Chunk.make("d", i, t, 0, len(t)) for i, t in enumerate(texts)]
    index = Bm25Index()
    index.build(chunks)
    return index, chunks


def test_bm25_score_by_hand() -> None:
    import math

    index, _ = _bm25(["a b", "a c c", "d"])
    [hit] = index.search("c", k=3)
    # N=3, df(c)=1, tf=2, |chunk|=3, avgdl=2, k1=1.5, b=0.75 (Lucene's idf)
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    tf_part = 2 * (1.5 + 1) / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2))
    assert hit.chunk_id == "d:1" and hit.score == pytest.approx(idf * tf_part, rel=1e-9)


def test_a_term_in_every_chunk_still_scores_positive() -> None:
    index, _ = _bm25(["x a", "x b", "x c"])
    hits = index.search("x", k=10)
    assert len(hits) == 3 and all(h.score > 0 for h in hits)  # non-empty first: not vacuous


def test_rrf_fuses_ranks_by_hand_and_ignores_magnitudes() -> None:
    dense = [Scored(chunk_id="a", score=0.9), Scored(chunk_id="b", score=0.8)]
    bm25 = [Scored(chunk_id="b", score=5.0), Scored(chunk_id="c", score=4.0)]
    fused = {s.chunk_id: s.score for s in rrf_fuse([dense, bm25], k_out=3)}
    assert fused == pytest.approx({"b": 1 / 62 + 1 / 61, "a": 1 / 61, "c": 1 / 62})
    loud = [Scored(chunk_id="a", score=1e9), Scored(chunk_id="b", score=1e-9)]
    assert rrf_fuse([loud, bm25], k_out=3) == rrf_fuse([dense, bm25], k_out=3)
