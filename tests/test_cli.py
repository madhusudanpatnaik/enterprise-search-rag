from __future__ import annotations

import json
from pathlib import Path

import pytest

from entsearch.cli import main
from entsearch.datasets import loader, synthetic


def test_cli_demo_runs_offline_and_reports_every_section(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Answer [SUPPORTED]" in out and "Answer [ABSTAINED]" in out
    assert "retrieval ablation — answerable questions" in out
    assert "hybrid+rerank" in out
    assert "latency (n=22)" in out and "p99 from too few samples" in out
    misses = len(synthetic.LEXICAL_EXPECTED_MISSES)
    assert f"{len(synthetic.GATE_CASES) - misses}/{len(synthetic.GATE_CASES)} claims" in out


def test_cli_search_on_synthetic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "How much unused PTO rolls over into the next year?"]) == 0
    assert "maximum of 5 days" in capsys.readouterr().out


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _beir(root: Path) -> Path:
    (root / "qrels").mkdir(parents=True)
    _write_jsonl(
        root / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Remote Access", "text": "The current VPN uses WireGuard."},
            {"_id": "d2", "title": "Expenses", "text": "Receipts are required over 50 dollars."},
            {"_id": "d3", "title": "Office", "text": "The office opens at 8 am."},
        ],
    )
    _write_jsonl(
        root / "queries.jsonl",
        [
            {"_id": "q1", "text": "Which VPN software is used?"},
            {"_id": "q2", "text": "When are receipts required?"},
            {"_id": "q3", "text": "A query with no positive judgement"},
        ],
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\nq3\td3\t0\n"
    )
    return root


def test_beir_loader_keeps_positive_judgements_only(tmp_path: Path) -> None:
    docs, cases = loader.load_beir(_beir(tmp_path))
    assert [d.id for d in docs] == ["d1", "d2", "d3"] and docs[0].title == "Remote Access"
    assert [(c.query.id, c.relevant_doc_ids) for c in cases] == [("q1", {"d1"}), ("q2", {"d2"})]


def test_cli_eval_on_a_beir_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--beir", str(_beir(tmp_path)), "--k", "2", "--retrieval-only"]) == 0
    out = capsys.readouterr().out
    assert "3 documents into 3 chunks; 2 queries" in out
    bm25 = next(line for line in out.splitlines() if line.strip().startswith("bm25"))
    assert bm25.split()[1:3] == ["2", "1.000"]  # n=2, hit@2 = 1.0


def test_cli_eval_on_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs, queries = tmp_path / "docs.jsonl", tmp_path / "queries.jsonl"
    _write_jsonl(docs, [{"id": d.id, "title": d.title, "text": d.text} for d in synthetic.corpus()])
    _write_jsonl(
        queries,
        [
            {"id": "q1", "text": "How long must passwords be?", "relevant_doc_ids": ["mfa"]},
            {"id": "u1", "text": "Who won the 2022 World Cup?", "answerable": False},
        ],
    )
    assert main(["eval", "--docs", str(docs), "--queries", str(queries), "--k", "3"]) == 0
    out = capsys.readouterr().out
    assert "1 answerable, 1 unanswerable" in out
    assert "abstained (of unanswerable)           1.000 (1/1)" in out


def test_cli_eval_needs_a_dataset() -> None:
    with pytest.raises(SystemExit):
        main(["eval"])


def test_download_lists_without_fetching(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["download", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "cross-encoder/nli-deberta-v3-base" in out and "nothing downloaded" in out


def test_download_fetches_each_model_into_the_cache() -> None:
    from entsearch import assets

    calls: list[dict[str, object]] = []

    def fetch(**kw: object) -> str:
        calls.append(kw)
        return "/c/mini"

    lines = assets.download(assets.find(["encoder"]), cache_dir="/c", fetch_model=fetch)
    assert calls == [
        {
            "repo_id": "BAAI/bge-small-en-v1.5",
            "repo_type": "model",
            "cache_dir": "/c",
        }
    ] and lines == ["encoder: /c/mini"]


def _zip(path: Path, members: dict[str, str]) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)


def test_beir_download_unpacks_once_and_refuses_escaping_members(tmp_path: Path) -> None:
    from entsearch import assets

    fetched: list[str] = []

    def fake_fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        _zip(
            dest,
            {
                "scifact/corpus.jsonl": "{}\n",
                "scifact/queries.jsonl": "{}\n",
                "scifact/qrels/test.tsv": "query-id\tcorpus-id\tscore\n",
            },
        )

    out = assets.download_beir("scifact", tmp_path, fetch=fake_fetch)
    assert (out / "corpus.jsonl").exists() and fetched[0].endswith("/scifact.zip")
    assets.download_beir("scifact", tmp_path, fetch=fake_fetch)
    assert len(fetched) == 1  # already there: not fetched again
    with pytest.raises(ValueError, match="outside"):
        assets.download_beir("evil", tmp_path, fetch=lambda u, d: _zip(d, {"../escaped.txt": "x"}))
    assert not (tmp_path.parent / "escaped.txt").exists()
    with pytest.raises(ValueError, match="not a BEIR dataset name"):
        assets.download_beir("../etc", tmp_path, fetch=fake_fetch)


# ------------------------------------------------------------------ calibration belongs to a corpus


def _questions(path: Path) -> Path:
    rows = [{"text": q, "answerable": True} for q in synthetic.CALIBRATION_ANSWERABLE]
    rows += [{"text": q, "answerable": False} for q in synthetic.CALIBRATION_UNANSWERABLE]
    _write_jsonl(path, rows)
    return path


def test_calibrating_a_real_corpus_needs_questions_about_it(tmp_path: Path) -> None:
    # regression: without --questions the synthetic wiki's questions were fitted on any corpus
    with pytest.raises(SystemExit, match="needs --questions"):
        main(["calibrate", "--beir", str(_beir(tmp_path)), "--out", str(tmp_path / "c.json")])


def test_a_calibration_is_refused_on_a_corpus_it_was_not_fitted_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    beir = _beir(tmp_path / "scifact")
    out = tmp_path / "cal.json"
    questions = _questions(tmp_path / "q.jsonl")
    assert main(["calibrate", "--beir", str(beir), "--questions", str(questions),
                 "--out", str(out)]) == 0  # fmt: skip
    assert "lexical on beir:scifact" in capsys.readouterr().out
    assert main(["eval", "--beir", str(beir), "--retrieval-only", "--calibration", str(out)]) == 0
    with pytest.raises(SystemExit, match="fitted on beir:scifact, not synthetic"):
        main(["search", "How long must passwords be?", "--calibration", str(out)])


def test_a_missing_model_library_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert main(["search", "vpn?", "--rerank", "cross"]) == 2
    assert "pip install -r requirements-models.txt" in capsys.readouterr().err
