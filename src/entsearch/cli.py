"""`entsearch` — demo the engine, search a corpus, or evaluate retrieval, answers and latency.

entsearch demo                                    # synthetic wiki, offline, fully scored
entsearch search "what vpn do we use?" [--docs docs.jsonl]
entsearch eval --beir path/to/scifact [--split test] [--k 10]
entsearch eval --docs docs.jsonl --queries queries.jsonl [--k 5]
# production models (need the extras / a key):
entsearch demo --encoder st --rerank cross --verify minicheck
entsearch search "..." --docs docs.jsonl --generate anthropic
# abstention fitted on held-out questions about the same corpus, with the same reranker:
entsearch calibrate --beir path/to/scifact --questions held_out.jsonl --rerank cross --out cal.json
entsearch eval --beir path/to/scifact --rerank cross --calibration cal.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entsearch import cite
from entsearch.answer import AnthropicGenerator, ExtractiveAnswerer, Generator
from entsearch.calibrate import Calibration
from entsearch.datasets import loader, synthetic
from entsearch.domain import Answer, Document, QueryCase
from entsearch.encode import Encoder, HashingEncoder, SentenceTransformerEncoder
from entsearch.metrics import (
    EvalReport,
    RetrievalScore,
    ablation,
    evaluate,
    gate_eval,
)
from entsearch.pipeline import SearchEngine
from entsearch.rerank import CrossEncoderReranker, LexicalReranker, Reranker


def _build_engine(
    args: argparse.Namespace, *, chunk_size: int = 60, calibration: Calibration | None = None
) -> SearchEngine:
    encoder: Encoder = (
        SentenceTransformerEncoder(args.encoder_model) if args.encoder == "st" else HashingEncoder()
    )
    reranker: Reranker = CrossEncoderReranker() if args.rerank == "cross" else LexicalReranker()
    generator: Generator = (
        AnthropicGenerator() if args.generate == "anthropic" else ExtractiveAnswerer()
    )
    verifier: cite.Verifier = cite.LexicalVerifier()
    if args.verify == "nli":
        verifier = cite.NliVerifier()
    elif args.verify == "minicheck":
        verifier = cite.MiniCheckVerifier()
    return SearchEngine(
        encoder, reranker, generator, verifier, chunk_size=chunk_size, calibration=calibration
    )


def _corpus(args: argparse.Namespace) -> tuple[list[Document], str, int]:
    """The documents a command runs on, a label naming them, and the chunk size: BEIR, a JSONL
    file, or the synthetic wiki."""
    if getattr(args, "beir", None):
        return loader.load_beir(args.beir, args.split)[0], f"beir:{Path(args.beir).name}", 60
    if getattr(args, "docs", None):
        return loader.load_documents(args.docs), f"docs:{Path(args.docs).name}", 60
    return synthetic.corpus(), "synthetic", synthetic.CHUNK_SIZE


def _engine_for(args: argparse.Namespace, corpus: str, chunk_size: int) -> SearchEngine:
    """The engine, with `--calibration` applied only if it was fitted on this corpus."""
    calibration = Calibration.load(args.calibration) if args.calibration else None
    if calibration is not None and calibration.corpus not in ("", corpus):
        raise SystemExit(
            f"{args.calibration} was fitted on {calibration.corpus}, not {corpus}: a calibration "
            "says what scores mean on one corpus; fit one here with `entsearch calibrate`"
        )
    return _build_engine(args, chunk_size=chunk_size, calibration=calibration)


def _print_answer(answer: Answer) -> None:
    if answer.abstained:
        status = "ABSTAINED"
    elif answer.supported:
        status = "SUPPORTED"
    else:
        status = f"UNSUPPORTED — {answer.refusal_reason}"
    print(f"Answer [{status}]:\n  {answer.text}")
    for c in answer.citations:
        print(f"  cites [{c.chunk_id}] {c.quote}")
    for r in answer.retrieved[:3]:
        print(f"  retrieved {r.chunk.id:<18} score={r.score:.3f}  {r.chunk.text[:70]!r}")
    print("  latency: " + "  ".join(f"{k}={v:.2f}ms" for k, v in answer.timings_ms.items()))


def _print_ablation(rows: list[RetrievalScore], k: int, label: str) -> None:
    print(f"\n{label}")
    header = f"{f'hit@{k}':>9}{f'recall@{k}':>11}{f'MRR@{k}':>9}{f'nDCG@{k}':>9}"
    print(f"  {'setup':<15}{'n':>4}{header}")
    for r in rows:
        print(f"  {r.setup:<15}{r.queries:>4}{r.hit_at_k:>9.3f}{r.recall_at_k:>11.3f}"
              f"{r.mrr:>9.3f}{r.ndcg_at_k:>9.3f}")  # fmt: skip


def _print_report(report: EvalReport) -> None:
    print(
        f"\nanswers (k={report.k}): {report.answerable} answerable, "
        f"{report.unanswerable} unanswerable, {report.skipped_no_gold} skipped (no gold)"
    )
    print(f"  answered (of answerable)              {report.answered}")
    print(f"  correct: contains a gold answer span  {report.correct}")
    print(f"  abstained (of unanswerable)           {report.abstained}")
    print(f"  drafts with a refused claim           {report.unsupported_drafts}")
    print(f"  answers with a verified citation      {report.cited}")
    flags = report.latency_unreliable
    caveat = f" — {', '.join(flags)} from too few samples to be a tail measurement" if flags else ""
    print(
        f"  latency (n={report.latency_n}) p50={report.latency_p50_ms:.2f}ms "
        f"p95={report.latency_p95_ms:.2f}ms p99={report.latency_p99_ms:.2f}ms{caveat}"
    )


def _cmd_demo(args: argparse.Namespace) -> int:
    engine = _engine_for(args, "synthetic", synthetic.CHUNK_SIZE)
    engine.index(synthetic.corpus())
    print(f"indexed {len(synthetic.corpus())} documents into {len(engine.chunks)} chunks")
    for question in [
        "What does the current VPN use?",
        "How much unused PTO rolls over into the next year?",
        "Who is the CEO of the company?",
    ]:
        print(f"\n=== {question} ===")
        _print_answer(engine.search(question, k=3))

    cases = synthetic.cases(engine)
    _print_ablation(ablation(engine, cases, k=3), 3, "retrieval ablation — answerable questions")
    _print_ablation(
        ablation(engine, synthetic.cases(engine, paraphrased=True), k=3),
        3,
        "retrieval ablation — paraphrased questions (little shared vocabulary)",
    )
    _print_report(evaluate(engine, cases, k=3))

    from entsearch.calibrate import calibrate

    cal = calibrate(
        engine,
        synthetic.CALIBRATION_ANSWERABLE,
        synthetic.CALIBRATION_UNANSWERABLE,
        corpus="synthetic",
    )
    floor = engine.min_relevance
    calibrated = _build_engine(args, chunk_size=synthetic.CHUNK_SIZE, calibration=cal)
    calibrated.index(synthetic.corpus())
    print(
        f"\nabstention calibrated on {cal.n_answerable}+{cal.n_unanswerable} held-out "
        f"questions (P(answerable) = sigmoid({cal.a:.2f}*score {cal.b:+.2f}), answer when "
        f">= {cal.threshold}); the uncalibrated floor was a raw score of {floor}:"
    )
    _print_report(evaluate(calibrated, cases, k=3))

    outcomes = gate_eval(engine, synthetic.GATE_CASES, k=3)
    right = sum(o.correct for o in outcomes)
    print(f"\ncitation gate ({engine.verifier.name}): {right}/{len(outcomes)} claims judged right")
    for o in outcomes:
        mark = "ok  " if o.correct else "MISS"
        verdict = "accepted" if o.accepted else "refused "
        truth = "true " if o.case.true else "false"
        print(f"  {mark} {truth} {verdict} {o.case.claim}")
        if o.case.note:
            print(f"         {o.case.note}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    docs, label, chunk_size = _corpus(args)
    engine = _engine_for(args, label, chunk_size)
    engine.index(docs)
    _print_answer(engine.search(args.query, k=args.k))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    cases: list[QueryCase]
    if args.beir:
        docs, cases = loader.load_beir(args.beir, args.split)
        label = f"beir:{Path(args.beir).name}"
    elif args.docs and args.queries:
        docs, cases = loader.load_documents(args.docs), loader.load_cases(args.queries)
        label = f"docs:{Path(args.docs).name}"
    else:
        raise SystemExit("eval needs --beir DIR, or --docs and --queries")
    engine = _engine_for(args, label, 60)
    engine.index(docs)
    print(f"indexed {len(docs)} documents into {len(engine.chunks)} chunks; {len(cases)} queries")
    _print_ablation(ablation(engine, cases, k=args.k), args.k, "retrieval ablation")
    if not args.retrieval_only:
        _print_report(evaluate(engine, cases, k=args.k))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import json

    from entsearch.calibrate import calibrate

    docs, label, chunk_size = _corpus(args)
    if label != "synthetic" and not args.questions:
        raise SystemExit(
            f"calibrating on {label} needs --questions about it, answerable and not; the built-in "
            "held-out questions are about the synthetic wiki"
        )
    engine = _build_engine(args, chunk_size=chunk_size)
    engine.index(docs)
    if args.questions:
        rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line]
        answerable = [r["text"] for r in rows if r["answerable"]]
        unanswerable = [r["text"] for r in rows if not r["answerable"]]
    else:
        answerable = synthetic.CALIBRATION_ANSWERABLE
        unanswerable = synthetic.CALIBRATION_UNANSWERABLE
    cal = calibrate(engine, answerable, unanswerable, cost=args.cost, corpus=label)
    cal.save(args.out)
    print(
        f"{cal.reranker} on {label}: P(answerable) = sigmoid({cal.a:.3f}*score {cal.b:+.3f}); "
        f"answer when >= {cal.threshold:.3f} ({len(answerable)} answerable, "
        f"{len(unanswerable)} not) -> {args.out}"
    )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from entsearch import assets

    if args.beir:
        print(f"{args.beir}: {assets.download_beir(args.beir, args.to)}")
        return 0
    chosen = assets.find(args.names)
    if args.list or args.dry_run:
        print("\n".join(assets.describe(chosen)))
        if args.dry_run:
            print("\n(dry run: nothing downloaded)")
        return 0
    for line in assets.download(chosen, cache_dir=args.cache_dir):
        print(line)
    return 0


def _add_model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--encoder", choices=["hashing", "st"], default="hashing")
    p.add_argument("--encoder-model", default="BAAI/bge-small-en-v1.5", help="with --encoder st")
    p.add_argument("--rerank", choices=["lexical", "cross"], default="lexical")
    p.add_argument("--generate", choices=["extractive", "anthropic"], default="extractive")
    p.add_argument("--verify", choices=["lexical", "nli", "minicheck"], default="lexical")
    p.add_argument("--calibration", help="abstention calibration JSON from `entsearch calibrate`")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="entsearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="offline demo on the synthetic wiki")
    _add_model_flags(d)
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("search", help="answer one query against a corpus")
    s.add_argument("query")
    s.add_argument("--docs", help="documents.jsonl (defaults to the synthetic wiki)")
    s.add_argument("--k", type=int, default=5)
    _add_model_flags(s)
    s.set_defaults(func=_cmd_search)

    c = sub.add_parser("calibrate", help="fit abstention on held-out labelled questions")
    c.add_argument("--docs", help="documents.jsonl (defaults to the synthetic wiki)")
    c.add_argument("--beir", help="a BEIR dataset directory: its corpus (bring your own questions)")
    c.add_argument("--split", default="test", help=argparse.SUPPRESS)
    c.add_argument(
        "--questions",
        help='questions.jsonl, one {"text": ..., "answerable": true|false} per line, about this '
        "corpus (defaults to the synthetic held-out set); never the evaluation questions",
    )
    c.add_argument(
        "--cost", type=float, default=1.0, help="a wrong answer's cost / an unneeded abstention's"
    )
    c.add_argument("--out", required=True)
    _add_model_flags(c)
    c.set_defaults(func=_cmd_calibrate)

    dl = sub.add_parser("download", help="fetch models / a BEIR set on the machine that runs this")
    dl.add_argument("names", nargs="*", help="models to fetch (default: all; see --list)")
    dl.add_argument("--list", action="store_true", help="show models, sizes and what needs them")
    dl.add_argument("--dry-run", action="store_true", help="show what would be fetched")
    dl.add_argument("--cache-dir", help="instead of the Hugging Face cache (HF_HOME)")
    dl.add_argument("--beir", help="a BEIR dataset to fetch instead, e.g. scifact, nfcorpus, fiqa")
    dl.add_argument("--to", default="data/beir", help="where BEIR datasets are unpacked")
    dl.set_defaults(func=_cmd_download)

    e = sub.add_parser("eval", help="retrieval ablation, answer metrics and latency")
    e.add_argument("--beir", help="a BEIR dataset directory (corpus.jsonl, queries.jsonl, qrels/)")
    e.add_argument("--split", default="test")
    e.add_argument("--docs", help="documents.jsonl")
    e.add_argument("--queries", help="queries.jsonl")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--retrieval-only", action="store_true", help="skip answering every query")
    _add_model_flags(e)
    e.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ModuleNotFoundError as e:
        hint = _install_hint(e.name)
        if hint is None:
            raise
        print(f"error: {e.name} is not installed; run: pip install -r {hint}", file=sys.stderr)
        return 2
    except TypeError as e:  # the SDK found no credentials when the request was made
        if "Could not resolve authentication method" not in str(e):
            raise
        print("error: no Anthropic credentials; set ANTHROPIC_API_KEY (see .env.example)",
              file=sys.stderr)  # fmt: skip
        return 2


# optional module -> the requirements file that installs it
_OPTIONAL = {
    "sentence_transformers": "requirements-models.txt",
    "minicheck": "requirements-minicheck.txt",
    "anthropic": "requirements-generate.txt",
}


def _install_hint(module: str | None) -> str | None:
    name = module or ""
    return next((f for m, f in _OPTIONAL.items() if name == m or name.startswith(m + ".")), None)


if __name__ == "__main__":
    raise SystemExit(main())
