from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from entsearch import cite, text
from entsearch.answer import AnthropicGenerator, ModelRefused
from entsearch.datasets import synthetic
from entsearch.domain import Chunk, Query, QueryCase, Retrieved
from entsearch.metrics import (
    Rate,
    evaluate,
    gate_eval,
    hit_at_k,
    mrr,
    ndcg_at_k,
    percentile,
    ranking,
    recall_at_k,
    score_retrieval,
)
from entsearch.pipeline import SearchEngine
from entsearch.rerank import LexicalReranker


def _retrieved(body: str, cid: str = "d:0", title: str = "") -> Retrieved:
    doc_id, _, idx = cid.partition(":")
    chunk = Chunk(id=cid, doc_id=doc_id, index=int(idx or 0), text=body, start=0, end=len(body),
                  title=title)  # fmt: skip
    return Retrieved(chunk=chunk, score=1.0)


def _engine() -> SearchEngine:
    engine = SearchEngine(chunk_size=synthetic.CHUNK_SIZE)
    engine.index(synthetic.corpus())
    return engine


# --------------------------------------------------------------------------- text


def test_numbers_are_canonical_however_written() -> None:
    assert text.numbers("paid $1,200 within 30 days") == {"1200", "30"}
    assert text.numbers("at least two weeks") == {"2"}
    assert text.numbers("the fifteenth") == text.numbers("the 15th") == {"15"}


def test_negation_is_scoped_to_its_clause() -> None:
    assert text.negated_stems("paid within 30 days, never 45") == set()
    assert text.signed_numbers("paid within 30 days, never 45") == {("30", False), ("45", True)}
    assert text.negated_stems("PTO is not paid out on departure") == {
        "pay",
        "out",
        "on",
        "departur",
    }
    # a negation with nothing after it denies what came before; coordination resets the scope
    assert text.negated_stems("Contractors are eligible; employees are not.") == {"employ"}
    assert text.negated_stems("Contractors do not accrue days and are not covered.") == {
        "accru",
        "day",
        "cover",
    }


# --------------------------------------------------------------------------- the lexical verifier

VERIFY = cite.LexicalVerifier()


def test_support_comes_from_one_sentence_never_a_pool() -> None:
    passage = _retrieved(
        "Full-time employees accrue 20 days of paid time off per year. "
        "Unused PTO rolls over up to a maximum of 5 days."
    )
    cid, reason = VERIFY.verify("Unused PTO rolls over up to a maximum of 20 days.", [passage])
    assert cid is None and reason is not None and "number(s) 20" in reason


def test_numbers_in_a_negated_clause_do_not_support() -> None:
    passage = _retrieved("Reimbursements are paid within 30 days, never 45.")
    assert VERIFY.verify("Reimbursements are paid within 30 days.", [passage])[0] == "d:0"
    cid, reason = VERIFY.verify("Reimbursements are paid within 45 days.", [passage])
    assert cid is None and reason is not None  # 45 is only in the "never" clause


def test_a_question_in_a_passage_is_not_evidence() -> None:
    faq = _retrieved("Can contractors take paid time off? No.")
    cid, reason = VERIFY.verify("Contractors can take paid time off.", [faq])
    assert cid is None and reason is not None


def test_the_document_title_gives_a_sentence_its_subject() -> None:
    passage = _retrieved("Its certificates were rotated every 365 days.", title="Legacy VPN")
    assert VERIFY.verify("Legacy VPN certificates were rotated every 365 days.", [passage])[0]
    wrong = "Legacy VPN certificates were rotated every 90 days."
    assert VERIFY.verify(wrong, [passage])[0] is None


def test_lowering_min_support_lets_one_swapped_word_in_five_through() -> None:
    passage = _retrieved("Domestic reimbursements are paid within 30 days of approval.")
    claim = "Domestic reimbursements are paid within 30 days of rejection."
    assert VERIFY.verify(claim, [passage])[0] is None
    assert cite.LexicalVerifier(min_support=0.8).verify(claim, [passage])[0] == "d:0"
    with pytest.raises(ValueError):
        cite.LexicalVerifier(min_support=0)


def test_gate_eval_misses_exactly_the_documented_lexical_limits() -> None:
    outcomes = gate_eval(_engine(), synthetic.GATE_CASES, k=3)
    wrong = {o.case.claim for o in outcomes if not o.correct}
    assert wrong == synthetic.LEXICAL_EXPECTED_MISSES


# --------------------------------------------------------------------------- the NLI verifier


class _FakeNli:
    """Stands in for the cross-encoder: fixed logits, and a config whose label order differs
    from the model card's — the verifier must read the order, not assume it."""

    class config:  # noqa: N801 - mirrors the attribute the real model exposes
        id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

    def __init__(self, logits: list[list[float]]) -> None:
        self._logits = logits

    def predict(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        return self._logits[: len(pairs)]


def test_nli_verifier_reads_the_label_order_from_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real _ensure() path runs: sentence_transformers.CrossEncoder is replaced by a stand-in
    # whose config lists the labels in a different order from the model card's default.
    import sys
    import types

    logits = [[-2.0, 0.0, 3.0], [4.0, 0.0, -1.0]]  # x contradicts, y entails (in id2label order)
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = lambda name: _FakeNli(logits)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    verifier = cite.NliVerifier()
    passages = [_retrieved("a", "x:0"), _retrieved("b", "y:0")]
    assert verifier.verify("claim", passages) == ("y:0", None)
    assert verifier._labels == ["entailment", "neutral", "contradiction"]
    module.CrossEncoder = lambda name: _FakeNli([[0.0, 3.0, 0.0]])  # type: ignore[attr-defined]
    cid, reason = cite.NliVerifier().verify("claim", passages[:1])  # neutral wins
    assert cid is None and reason is not None and "not entailed" in reason


# --------------------------------------------------------------------------- answer-level gate


def test_enforce_handles_markers_meta_phrasing_empty_and_abstention() -> None:
    passage = [_retrieved("The current VPN uses WireGuard.", "vpn:0")]
    ok = cite.enforce("According to the documents, the current VPN uses WireGuard [1].", passage,
                      VERIFY)  # fmt: skip
    assert ok.claims == 1 and not ok.unsupported and ok.citations[0].chunk_id == "vpn:0"
    assert cite.enforce("", passage, VERIFY).unsupported == ["(empty answer)"]
    assert cite.enforce(cite.ABSTAIN, passage, VERIFY).abstained
    assert cite.enforce("[1] [2].", passage, VERIFY).unsupported == ["(no checkable claim)"]


# --------------------------------------------------------------------------- pipeline + metrics


def test_pipeline_answers_with_grounded_citations_and_abstains_off_topic() -> None:
    engine = _engine()
    answer = engine.search("What does the current VPN use?", k=3)
    assert answer.supported and not answer.abstained
    assert "The current VPN uses WireGuard" in answer.text
    assert any(c.chunk_id.startswith("vpn:") and "WireGuard" in c.quote for c in answer.citations)
    off_topic = engine.search("Who won the 2022 World Cup?", k=3)
    assert off_topic.abstained and off_topic.text == cite.ABSTAIN
    assert "generate" not in off_topic.timings_ms  # abstained before generating


def test_answer_level_metrics_on_the_synthetic_wiki() -> None:
    engine = _engine()
    report = evaluate(engine, synthetic.cases(engine), k=3)
    assert report.skipped_no_gold == 0
    assert (report.answered.hits, report.answered.of) == (12, 12)
    assert (report.correct.hits, report.correct.of) == (12, 12)
    # The lexical relevance floor sees topic words: near-domain unanswerable questions ("How
    # many sick days ...?") get answered. Measured, not hidden.
    assert report.abstained.of == 10 and report.abstained.hits == 6
    # One draft carried the legacy VPN's "Its certificates were rotated every 365 days" into an
    # answer about the current VPN; the gate refused it and the user never saw it.
    assert (report.unsupported_drafts.hits, report.cited.hits) == (1, 12)
    assert report.latency_n == 22 and report.latency_unreliable == ["p99"]


def test_the_user_gets_only_verified_claims() -> None:
    answer = _engine().search("How often are the current VPN certificates rotated?", k=3)
    assert "365" in answer.draft and "365" not in answer.text
    assert "rotated every 90 days" in answer.text and not answer.supported
    assert "365" in answer.refusal_reason


def test_a_pronoun_claim_must_hold_for_the_previous_claims_document() -> None:
    passages = [
        _retrieved("The current VPN uses WireGuard.", "vpn:0", title="Remote Access"),
        _retrieved("Its certificates were rotated every 365 days.", "old:0", title="Legacy VPN"),
    ]
    answer = "The current VPN uses WireGuard. Its certificates were rotated every 365 days."
    gate = cite.enforce(answer, passages, VERIFY)
    assert [c.chunk_id for c in gate.citations] == ["vpn:0"] and len(gate.unsupported) == 1


def test_rates_with_nothing_to_divide_by_are_not_zero() -> None:
    assert Rate(hits=0, of=0).value is None and str(Rate(hits=0, of=0)) == "n/a (0 cases)"
    assert str(Rate(hits=3, of=4)) == "0.750 (3/4)"


def test_metrics_by_hand() -> None:
    assert hit_at_k(["a", "b"], {"b"}, 1) == 0.0 and hit_at_k(["a", "b"], {"b"}, 2) == 1.0
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 3) == pytest.approx(0.5)
    assert mrr(["x", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert ndcg_at_k(["x", "b"], {"b"}, 2) == pytest.approx(1 / 1.5849625, abs=1e-6)
    # more relevant items than k: the ideal ranking is cut at k
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "x"], {"a", "b", "c"}, 2) == pytest.approx(1 / (1 + 1 / 1.5849625))
    # graded gains, as trec_eval: a grade-1 document ranked above a grade-2 one is not perfect
    assert ndcg_at_k(["d1", "d2"], {"d1", "d2"}, 2, {"d1": 1, "d2": 2}) == pytest.approx(
        (1 + 2 / 1.5849625) / (2 + 1 / 1.5849625), abs=1e-6
    )
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25.0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


class _IdentityReranker(LexicalReranker):
    name = "identity"

    def rerank(self, query: Query, candidates: list[Retrieved]) -> list[Retrieved]:
        return list(candidates)


class _ReversedReranker(LexicalReranker):
    name = "reversed"

    def rerank(self, query: Query, candidates: list[Retrieved]) -> list[Retrieved]:
        return list(reversed(super().rerank(query, candidates)))


def test_reranking_helps_and_a_worse_reranker_scores_worse() -> None:
    # The metric must be able to fail in both directions: the lexical reranker beats keeping the
    # fused order, and reversing it is worse than both.
    def mrr_with(reranker: LexicalReranker) -> float:
        engine = SearchEngine(reranker=reranker, chunk_size=synthetic.CHUNK_SIZE)
        engine.index(synthetic.corpus())
        cases = synthetic.cases(engine) + synthetic.cases(engine, paraphrased=True)
        return score_retrieval(engine, cases, mode="hybrid", rerank=True, k=3).mrr

    lexical, identity, reversed_ = (
        mrr_with(LexicalReranker()),
        mrr_with(_IdentityReranker()),
        mrr_with(_ReversedReranker()),
    )
    assert lexical > identity > reversed_


def test_recall_counts_a_gold_sentence_once_however_many_chunks_hold_it() -> None:
    engine = _engine()
    case = next(c for c in synthetic.cases(engine) if "Split tunneling" in c.answer_spans[0])
    assert len(case.relevant_chunk_ids) == 2  # the overlap puts the sentence in two chunks
    ranked, gold = ranking(engine, case, mode="hybrid", rerank=True, k=1)
    assert gold == {"span:0"} and ranked == ["span:0"]  # one chunk with it is full recall


def test_document_ranking_reaches_k_distinct_documents() -> None:
    from entsearch.domain import Document

    docs = [Document(id=f"d{i}", title=f"Doc {i}",
                     text=" ".join(f"Topic shared {i} sentence {j}." for j in range(12)))
            for i in range(30)]  # fmt: skip
    engine = SearchEngine(chunk_size=8)
    engine.index(docs)
    case = QueryCase(query=Query(id="q", text="topic shared sentence"),
                     relevant_doc_ids={d.id for d in docs})  # fmt: skip
    ranked, _ = ranking(engine, case, mode="bm25", rerank=False, k=10)
    assert len(ranked) == 10 == len(set(ranked))


def test_document_level_cases_rank_documents_by_their_best_chunk() -> None:
    engine = _engine()
    case = QueryCase(query=Query(id="q", text="How long must passwords be?"),
                     relevant_doc_ids={"mfa"})  # fmt: skip
    ranked, gold = ranking(engine, case, mode="hybrid", rerank=True, k=3)
    assert gold == {"mfa"} and ranked[0] == "mfa"
    assert len(ranked) == len(set(ranked))  # documents, not chunks: no repeats


# --------------------------------------------------------------------------- the real SDK, mocked


def _client(response: httpx2.Response) -> tuple[anthropic.Anthropic, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append({"url": str(request.url), "beta": request.headers.get("anthropic-beta"),
                     "body": json.loads(request.content)})  # fmt: skip
        return response

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    return anthropic.Anthropic(api_key="test-key", http_client=http, max_retries=0), sent


def _message(answer: str | None, stop_reason: str = "end_turn", **extra: Any) -> httpx2.Response:
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "...", "signature": "s"}]
    if answer is not None:
        content.append({"type": "text", "text": answer})
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            **extra,
        },
    )


def test_generator_request_through_the_real_sdk() -> None:
    client, sent = _client(_message("The current VPN uses WireGuard [1]."))
    passage = _retrieved("The current VPN uses WireGuard.", "vpn:0", title="Remote Access")
    out = AnthropicGenerator(client=client).generate(Query(id="q", text="What VPN?"), [passage])
    assert out == "The current VPN uses WireGuard [1]."  # the thinking block is skipped
    [call] = sent
    body = call["body"]
    assert call["url"].endswith("/v1/messages?beta=true")
    assert call["beta"] == "server-side-fallback-2026-07-01"
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "high"}
    assert "temperature" not in body
    assert cite.ABSTAIN in body["system"]
    assert "[1] (Remote Access) The current VPN uses WireGuard." in body["messages"][0]["content"]


@pytest.mark.parametrize(
    ("response", "error", "match"),
    [
        (
            _message(
                "The VPN",
                "refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": None},
            ),
            ModelRefused,
            "category: cyber",
        ),
        (_message("The VPN uses", "max_tokens"), RuntimeError, "truncated"),
    ],
)
def test_generator_refusal_and_truncation_are_errors_not_answers(
    response: httpx2.Response, error: type[Exception], match: str
) -> None:
    client, _ = _client(response)
    with pytest.raises(error, match=match):
        AnthropicGenerator(client=client).generate(Query(id="q", text="?"), [_retrieved("x")])


def test_bullets_and_headings_are_separate_sentences() -> None:
    page = (
        "## Accrual\n- Full-time employees accrue 20 days of paid time off per year\n"
        "- Accrual starts on the first day of employment\n\n## Rollover\n"
        "- Unused PTO rolls over up to a maximum of 5 days into the next year"
    )
    passage = [_retrieved(page, "pto:0", title="PTO")]
    borrowed = cite.enforce(
        "Unused PTO rolls over up to a maximum of 20 days [1].", passage, VERIFY
    )
    assert borrowed.unsupported and not borrowed.citations  # the 20 is in another bullet
    bulleted = cite.enforce(
        "- Unused PTO rolls over up to a maximum of 5 days [1]", passage, VERIFY
    )
    assert not bulleted.unsupported and bulleted.citations  # markers are not words
