"""A small fictional enterprise wiki built so the evaluation can fail.

Everything here is invented. At `CHUNK_SIZE` the corpus chunks into ~30 passages, so k=3 retrieves
a tenth of it. It is built around the things that make retrieval and grounding hard:

  - **Hard negatives** that share vocabulary but not facts: contractor vs employee time off, the
    retired OpenVPN service vs the current WireGuard one, international vs domestic reimbursements.
  - **Paraphrased** questions with little lexical overlap ("vacation" for "paid time off") — the
    case a learned encoder exists for, and where the offline hashing encoder should do poorly.
  - **Unanswerable** questions the corpus never covers: the right answer is to abstain.
  - **An FAQ** whose question ("Can contractors take paid time off?") shares every word with a
    false claim — a question in a passage is not evidence.
  - **Gate claims**, true and false: wrong numbers, numbers borrowed from a neighbouring sentence,
    flipped negations, a swapped entity, an antonym swap, a negation scoped to a second predicate
    — and one the lexical verifier gets wrong on purpose, so its limit is measured rather than
    claimed: a true paraphrase it refuses, which is what MiniCheck is for.

Gold relevance is derived, not hand-labelled: a question's relevant chunks are those containing its
gold phrase under the engine's actual chunking, so the eval stays correct if the chunker changes.

Fourteen questions cannot rank retrievers — the ablation on them swings with the chunk size. They
check that the harness measures what it claims to; numbers that rank setups come from a real
benchmark (`entsearch eval --beir ...`).
"""

from __future__ import annotations

from entsearch.domain import Document, Query, QueryCase
from entsearch.metrics import GateCase
from entsearch.pipeline import SearchEngine

CHUNK_SIZE = 20  # words; small enough that k=3 is a small fraction of the corpus

_DOCS: list[tuple[str, str, str]] = [
    (
        "pto",
        "Paid Time Off Policy",
        "Full-time employees accrue 20 days of paid time off per year. Accrual starts on the first "
        "day of employment and is prorated for partial years.\n\n"
        "PTO requests must be submitted at least two weeks in advance through the HR portal. "
        "Managers approve or decline requests within three business days.\n\n"
        "Unused PTO rolls over up to a maximum of 5 days into the next year. PTO is not paid out on "
        "departure except where local law requires it.",
    ),
    (
        "pto-contractors",
        "Contractor Time Away",
        "Contractors are not eligible for paid time off. Contractor time away is unpaid and must be "
        "agreed with the engagement manager in advance. Contractors do not accrue vacation days and "
        "are not covered by the PTO rollover rule.",
    ),
    (
        "expense",
        "Expense Policy",
        "Expenses over 50 dollars require an itemized receipt. Receipts must be uploaded within 10 "
        "days of the purchase.\n\n"
        "Domestic reimbursements are paid within 30 days of approval. International reimbursements "
        "are paid within 45 days of approval because of currency conversion.\n\n"
        "All travel must be booked through the corporate travel portal. First-class flights are not "
        "reimbursed.",
    ),
    (
        "vpn",
        "Remote Access",
        "Remote access to internal systems requires the corporate VPN. The current VPN uses "
        "WireGuard and its certificates are rotated every 90 days.\n\n"
        "Split tunneling is disabled by policy. Devices must pass a posture check before the VPN "
        "connects.",
    ),
    (
        "vpn-legacy",
        "Legacy VPN (retired)",
        "The legacy VPN used OpenVPN and was retired in 2024. Its certificates were rotated every 365 "
        "days. Legacy VPN profiles no longer connect and should be deleted from laptops.",
    ),
    (
        "mfa",
        "Account Security",
        "All employees must enable multi-factor authentication on their accounts. Hardware security "
        "keys are required for engineers with production access.\n\n"
        "Passwords must be at least 16 characters long. Passwords are rotated only after a suspected "
        "compromise.",
    ),
    (
        "onboarding",
        "Onboarding",
        "New hires complete security training in their first week. Laptops are provisioned by IT "
        "before the start date. Access to systems is granted based on the employee's role and "
        "reviewed every quarter.",
    ),
    (
        "faq",
        "Time Off FAQ",
        "Can contractors take paid time off? No. Contractor time away is unpaid and is agreed with "
        "the engagement manager.",
    ),
    (
        "office",
        "Office Information",
        "The office is open from 8 am to 7 pm on weekdays. Visitors must sign in at reception and "
        "wear a badge. Parking is available in the north garage on a first-come basis.",
    ),
]

# (question, gold phrase present verbatim in the relevant passage)
ANSWERABLE: list[tuple[str, str]] = [
    (
        "How many days of paid time off do full-time employees accrue?",
        "accrue 20 days of paid time off",
    ),
    ("How much unused PTO rolls over into the next year?", "maximum of 5 days"),
    ("How far in advance must PTO requests be submitted?", "at least two weeks in advance"),
    ("When are domestic reimbursements paid?", "Domestic reimbursements are paid within 30 days"),
    (
        "When are international reimbursements paid?",
        "International reimbursements are paid within 45 days",
    ),
    ("What does the current VPN use?", "current VPN uses WireGuard"),
    ("How often are the current VPN certificates rotated?", "rotated every 90 days"),
    ("Do engineers need hardware security keys?", "keys are required for engineers"),
    ("How long must passwords be?", "at least 16 characters"),
    ("When do new hires complete security training?", "security training in their first week"),
    (
        "Are contractors eligible for paid time off?",
        "Contractors are not eligible for paid time off",
    ),
    ("Is split tunneling allowed on the VPN?", "Split tunneling is disabled"),
]

# Same facts, little shared vocabulary: what a learned encoder is for.
PARAPHRASED: list[tuple[str, str]] = [
    ("How much vacation does a full-timer earn annually?", "accrue 20 days of paid time off"),
    ("Which tunnel software secures remote connections today?", "current VPN uses WireGuard"),
]

# Off-topic questions, then near-domain ones: about the wiki's subjects (leave, pay, VPN, laptops)
# but not answered by it — the hard case for a relevance floor, which sees the topic words.
UNANSWERABLE: list[str] = [
    "Who is the CEO of the company?",
    "What is on the cafeteria menu on Friday?",
    "Who won the 2022 World Cup?",
    "What is the parental leave policy?",
    "How many sick days do full-time employees get?",
    "What is the parental leave policy for employees?",
    "When are annual bonuses paid?",
    "Which VPN do contractors use on their phones?",
    "How long is the probation period for new hires?",
    "How much is the home office stipend?",
]

GATE_CASES: list[GateCase] = [
    GateCase(
        claim="According to the policy, full-time employees accrue 20 days of paid time off "
        "per year [1].",
        true=True,
        note="LLM-style phrasing with a citation marker",
    ),
    GateCase(claim="The current VPN uses WireGuard [2].", true=True),
    GateCase(claim="Passwords must be at least 16 characters long.", true=True),
    GateCase(claim="International reimbursements are paid within 45 days of approval.", true=True),
    GateCase(
        claim="Full-time employees accrue 50 days of paid time off per year.",
        true=False,
        note="wrong number",
    ),
    GateCase(
        claim="Unused PTO rolls over up to a maximum of 20 days.",
        true=False,
        note="number borrowed from a neighbouring sentence",
    ),
    GateCase(claim="Split tunneling is not disabled.", true=False, note="flipped negation"),
    GateCase(
        claim="Contractors are eligible for paid time off.", true=False, note="flipped negation"
    ),
    GateCase(
        claim="The current VPN uses OpenVPN.",
        true=False,
        note="entity swapped in from the retired service",
    ),
    GateCase(
        claim="Domestic reimbursements are paid within 30 days of rejection.",
        true=False,
        note="antonym swap — caught because 'rejection' is in no retrieved sentence",
    ),
    GateCase(
        claim="Contractors can take paid time off.",
        true=False,
        note="every word is in the FAQ's question, which is not evidence",
    ),
    GateCase(
        claim="Full-time employees get 20 days of vacation a year.",
        true=True,
        note="true paraphrase — the lexical verifier refuses it (expected miss; NLI's job)",
    ),
    GateCase(
        claim="Contractors do not accrue vacation days and are covered by the PTO rollover rule.",
        true=False,
        note="negation of the second predicate flipped",
    ),
    GateCase(
        claim="Expenses are not reimbursed.",
        true=False,
        note="the document title must not lend its subject to 'first-class flights'",
    ),
    GateCase(
        claim="Remote access is disabled by policy.",
        true=False,
        note="title pooling again: it is split tunneling that is disabled",
    ),
    GateCase(
        claim="The office is open from 7 pm to 8 am on weekdays.",
        true=False,
        note="times swapped: values bound to their words",
    ),
    GateCase(
        claim="IT reviews access every quarter.",
        true=False,
        note="an acronym that lowercases into a stopword is still a word",
    ),
    GateCase(
        claim="Hardware security keys may be used by engineers with production access.",
        true=False,
        note="'may' (permission) is not 'are required' (obligation)",
    ),
    GateCase(
        claim="The legacy VPN used OpenVPN and was retired in 2024.",
        true=True,
        note="a true statement about the hard negative",
    ),
    GateCase(
        claim="The current VPN uses WireGuard and is rotated every 365 days.",
        true=False,
        note="the 365 belongs to the legacy VPN's sentence",
    ),
]

# The gate cases the lexical verifier is expected to get wrong — its measured limits.
LEXICAL_EXPECTED_MISSES = {
    "Full-time employees get 20 days of vacation a year.",
}


# Held out from evaluation: what `calibrate` fits abstention on. Answerable questions in other
# words than ANSWERABLE's; unanswerable ones near the wiki's topics, none in UNANSWERABLE.
CALIBRATION_ANSWERABLE: list[str] = [
    "What is the minimum length for a password?",
    "How soon after a purchase must receipts be uploaded?",
    "Which protocol does the current VPN run on?",
    "What do new hires do in their first week?",
    "How many PTO days can be carried into the next year?",
    "Who needs hardware security keys?",
    "What hours is the office open on weekdays?",
    "Where can visitors park?",
    "How quickly do managers respond to PTO requests?",
    "Are first-class flights reimbursed?",
]
CALIBRATION_UNANSWERABLE: list[str] = [
    "What is the bereavement leave policy?",
    "How are overtime hours paid?",
    "What is the dress code at the office?",
    "How much is the retirement plan match?",
    "How do I reset a forgotten VPN password?",
    "When is the next company offsite?",
    "What is the mileage reimbursement rate?",
    "Is there a gym in the office?",
    "How many weeks of sabbatical do employees get?",
    "Which laptop models does IT provision?",
]


def corpus() -> list[Document]:
    return [Document(id=i, title=t, text=body, source="wiki") for i, t, body in _DOCS]


def cases(engine: SearchEngine, *, paraphrased: bool = False) -> list[QueryCase]:
    """Eval cases against an already-indexed engine. Paraphrased questions are opt-in so they are
    reported separately, not blended into the headline numbers."""
    pairs = PARAPHRASED if paraphrased else ANSWERABLE
    out = [
        QueryCase(
            query=Query(id=f"{'p' if paraphrased else 'q'}{n}", text=question),
            relevant_chunk_ids={c.id for c in engine.chunks if phrase.lower() in c.text.lower()},
            answer_spans=[phrase],
        )
        for n, (question, phrase) in enumerate(pairs)
    ]
    if not paraphrased:
        out += [
            QueryCase(query=Query(id=f"u{n}", text=q), answerable=False)
            for n, q in enumerate(UNANSWERABLE)
        ]
    return out
