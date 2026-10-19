"""Calibrated abstention, the MiniCheck verifier and the encoder's query instruction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from entsearch import cite
from entsearch.calibrate import Calibration, calibrate, fit_platt, threshold_for
from entsearch.cli import main
from entsearch.datasets import synthetic
from entsearch.domain import Chunk, Retrieved
from entsearch.encode import BGE_QUERY_INSTRUCTION, HashingEncoder, SentenceTransformerEncoder
from entsearch.metrics import evaluate
from entsearch.pipeline import SearchEngine
from entsearch.rerank import CrossEncoderReranker

# ------------------------------------------------------------------ Platt scaling


def test_platt_recovers_a_known_logistic_curve() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(-3, 3, 4000)
    y = rng.random(4000) < 1 / (1 + np.exp(-(2.0 * s - 1.0)))
    a, b = fit_platt(s.tolist(), y.tolist())
    assert a == pytest.approx(2.0, abs=0.15) and b == pytest.approx(-1.0, abs=0.15)


def test_platt_stays_finite_on_separable_data_and_needs_both_classes() -> None:
    a, b = fit_platt([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert math.isfinite(a) and math.isfinite(b) and a > 0
    with pytest.raises(ValueError, match="both"):
        fit_platt([0.5, 0.6], [True, True])


def test_the_threshold_follows_the_cost_of_a_wrong_answer() -> None:
    assert threshold_for(1.0) == 0.5 and threshold_for(3.0) == 0.75
    with pytest.raises(ValueError):
        threshold_for(0)


def test_calibration_round_trips_and_is_numerically_stable(tmp_path: Path) -> None:
    cal = Calibration("lexical", a=4.0, b=-2.0, threshold=0.5, n_answerable=3, n_unanswerable=3)
    cal.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json") == cal
    assert cal.probability(0.5) == 0.5 and cal.answers(0.5) and not cal.answers(0.49)
    assert cal.probability(1e6) == 1.0 and cal.probability(-1e6) == 0.0  # no overflow


# ------------------------------------------------------------------ in the engine


def _engine(calibration: Calibration | None = None) -> SearchEngine:
    engine = SearchEngine(chunk_size=synthetic.CHUNK_SIZE, calibration=calibration)
    engine.index(synthetic.corpus())
    return engine


def test_calibration_questions_are_held_out_from_the_evaluation() -> None:
    held_out = set(synthetic.CALIBRATION_ANSWERABLE) | set(synthetic.CALIBRATION_UNANSWERABLE)
    evaluated = {q for q, _ in synthetic.ANSWERABLE} | set(synthetic.UNANSWERABLE)
    assert not held_out & evaluated


def test_calibrated_abstention_on_held_out_questions() -> None:
    base = _engine()
    cal = calibrate(base, synthetic.CALIBRATION_ANSWERABLE, synthetic.CALIBRATION_UNANSWERABLE)
    cases = synthetic.cases(base)
    floor, calibrated = evaluate(base, cases), evaluate(_engine(cal), cases)
    assert (floor.abstained.hits, calibrated.abstained.hits) == (6, 8)  # of 10 unanswerable
    assert floor.correct.hits == calibrated.correct.hits == 12  # no answerable question lost


def test_a_calibration_belongs_to_one_reranker() -> None:
    cal = Calibration("lexical", 1.0, 0.0, 0.5, 1, 1)
    with pytest.raises(ValueError, match="fitted for 'lexical'"):
        SearchEngine(reranker=CrossEncoderReranker(), calibration=cal)


def test_calibrate_cli_writes_a_calibration_search_can_use(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = tmp_path / "q.jsonl"
    rows = [{"text": q, "answerable": True} for q in synthetic.CALIBRATION_ANSWERABLE]
    rows += [{"text": q, "answerable": False} for q in synthetic.CALIBRATION_UNANSWERABLE]
    questions.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "cal.json"
    assert main(["calibrate", "--questions", str(questions), "--out", str(out), "--cost", "3"]) == 0
    assert Calibration.load(out).threshold == 0.75
    assert main(["search", "What is the parental leave policy?", "--calibration", str(out)]) == 0
    assert "The documents do not cover this." in capsys.readouterr().out


# ------------------------------------------------------------------ MiniCheck


def _hit(cid: str, text: str, title: str = "") -> Retrieved:
    return Retrieved(chunk=Chunk.make(cid.split(":")[0], 0, text, 0, len(text), title), score=1.0)


class _Scorer:
    def __init__(self, probs: list[float]) -> None:
        self.probs = probs
        self.seen: list[tuple[list[str], list[str]]] = []

    def score(self, docs: list[str], claims: list[str]) -> Any:
        self.seen.append((docs, claims))
        return [int(p >= 0.5) for p in self.probs], self.probs, None, None


def test_minicheck_cites_the_best_supporting_passage() -> None:
    scorer = _Scorer([0.2, 0.91])
    v = cite.MiniCheckVerifier(scorer=scorer)
    hits = [_hit("a:0", "Unrelated text."), _hit("vpn:0", "Its certificates rotate.", "VPN")]
    assert v.verify("The VPN certificates are rotated.", hits) == ("vpn:0", None)
    docs, claims = scorer.seen[0]
    assert docs[1] == "VPN's certificates rotate."  # a leading pronoun read as the title
    assert claims == ["The VPN certificates are rotated."] * 2


def test_minicheck_refuses_below_its_threshold() -> None:
    v = cite.MiniCheckVerifier(scorer=_Scorer([0.3]), threshold=0.5)
    chunk_id, reason = v.verify("claim", [_hit("a:0", "text")])
    assert chunk_id is None and reason == "no passage supports it (best MiniCheck support 0.30)"
    assert v.verify("claim", []) == (None, "no retrieved passage")
    assert v.name == "minicheck-flan-t5-large"


# ------------------------------------------------------------------ the encoder


class _FakeST:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kw: Any) -> Any:
        self.texts += texts
        return np.ones((len(texts), 3), dtype=np.float32) / math.sqrt(3)


def test_bge_queries_get_the_retrieval_instruction_and_passages_do_not() -> None:
    enc = SentenceTransformerEncoder()
    assert enc.name == "BAAI/bge-small-en-v1.5" and enc.query_prefix == BGE_QUERY_INSTRUCTION
    fake = _FakeST()
    enc._model = fake
    enc.encode(["a passage"])
    enc.encode_queries(["a question"])
    assert fake.texts == ["a passage", BGE_QUERY_INSTRUCTION + "a question"]
    assert SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2").query_prefix == ""


def test_the_hashing_encoder_encodes_queries_like_passages() -> None:
    enc = HashingEncoder()
    assert np.array_equal(enc.encode_queries(["vpn"]), enc.encode(["vpn"]))
