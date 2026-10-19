# Enterprise Search System

[![CI](https://github.com/madhusudanpatnaik/enterprise-search-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/madhusudanpatnaik/enterprise-search-rag/actions/workflows/ci.yml)

Enterprise RAG done the way the talk argues for: not "embed → top-k → prompt", but **chunk
deliberately → retrieve two ways → fuse → re-rank → refuse anything the sources don't support**,
measured on retrieval quality *and* tail latency.

> The wiki in `datasets/synthetic.py` is fictional. The retrieval math (BM25, RRF, cosine), the
> citation gate, and the metrics are real and run offline; the production models
> (sentence-transformers encoder, cross-encoder reranker, MiniCheck, an LLM generator) are real
> code behind interfaces, swapped in with flags and **not yet run** — so there is no measured
> result on a real corpus yet; `entsearch eval --beir ...` produces one.

```bash
uv venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
uv run pytest -q                                   # 67 passed, offline, deterministic
uv run entsearch demo                              # synthetic wiki: answers + citations + eval
uv run entsearch search "what vpn do we use?" --docs docs.jsonl
uv run entsearch eval --beir path/to/scifact --k 10     # or --docs/--queries JSONL
# production models (install the extras, set a key):
uv run entsearch demo --encoder st --rerank cross --verify minicheck
uv run entsearch search "..." --docs docs.jsonl --generate anthropic
# abstention, fitted on held-out questions about the same corpus (yours: BEIR has no unanswerable ones)
uv run entsearch calibrate --beir path/to/scifact --questions held_out.jsonl --rerank cross --out cal.json
uv run entsearch eval --beir path/to/scifact --rerank cross --calibration cal.json
```

## Run it on another machine

Building and testing never downloads a model. Each machine installs from the
requirements files and fetches its own models; nothing large lives in the repository.

```bash
python3.12 -m venv .venv && source .venv/bin/activate    # or: uv venv --python 3.12
pip install -r requirements.txt                          # core: the offline demo and the evaluation
pip install -r requirements-models.txt                   # sentence-transformers: encoder, reranker, NLI
pip install -r requirements-generate.txt                 # the Claude answer generator
pip install -r requirements-minicheck.txt                # MiniCheck, the grounding verifier
pip install -r requirements-dev.txt                      # tests, mypy, ruff
cp .env.example .env    # fill in the keys you need, then:  set -a; . ./.env; set +a
entsearch download --list           # encoder ~90 MB, reranker ~90 MB, NLI ~0.7 GB
entsearch download                  # into the Hugging Face cache (HF_HOME)
entsearch download --beir scifact   # a BEIR benchmark into data/beir/
pytest -q && entsearch eval --beir data/beir/scifact --encoder st --rerank cross --k 10
```


## The pipeline

```
documents ─► chunk ─► ┌ dense index (cosine) ┐         ┌─ rerank ─┐
                      │                       ├─ RRF ──►│ (cross-  ├─► generate ─► CITE-GATE ─► answer
                      └ BM25 index (lexical)  ┘         │ encoder) │                 │          + citations
                                                        └──────────┘       refuse unsupported ◄┘
```

- **Chunking** (`chunk.py`): fixed / sentence / recursive, each returning real character offsets so
  a citation points at an exact span. Sentence chunking keeps claims intact for the gate.
- **Two retrievers** (`index.py`): exact cosine over L2-normalized vectors, and **Okapi BM25 written
  out** (idf, k1 saturation, b length-normalization) — not a library call, so the ranking is
  inspectable.
- **Fusion** (`retrieve.py`): Reciprocal Rank Fusion, which needs no score calibration between a
  cosine and a BM25 score because it fuses ranks, not magnitudes.
- **Re-ranking** (`rerank.py`): a cross-encoder reads (query, passage) together — the accuracy pass
  enterprise RAG relies on. A deterministic lexical reranker stands in offline.
- **The gate** (`cite.py`, `text.py`): every sentence of an answer is a claim, and one retrieved
  sentence must say it — compared clause by clause: every content word present with the same
  negation and hypothetical state (negation is a per-clause parity that verbs like "reject" and
  "cancel" flip too), every number by value, polarity and the word it is attached to. A document's
  title supplies only a leading pronoun's subject ("Its certificates ..."); its words are never
  pooled in. A claim that opens with a pronoun must hold for the previous claim's document. Bullets,
  headings and blank lines separate sentences; questions in a passage are not evidence. It does
  **not** trust the model's `[n]` markers. **The user receives only verified claims** — an
  unsupported one is dropped (the draft is kept for inspection), and an answer with nothing
  verified becomes an abstention. `--verify minicheck` swaps in MiniCheck, a model trained for
  exactly this check (a claim against a passage); `--verify nli` a general entailment model.
- **Abstention** (`rerank.py`, `calibrate.py`, `pipeline.py`): when no reranked passage is
  relevant enough, the engine answers "The documents do not cover this." without calling the
  generator. "Relevant enough" is learned, not set by hand: `entsearch calibrate` fits
  P(answerable | top reranked score) by Platt scaling on held-out questions labelled answerable or
  not, and the engine answers only when that probability reaches cost / (1 + cost), where `cost`
  is how much worse a wrong answer is than an unneeded abstention. A calibration belongs to one
  reranker on one corpus: it records both, and `--calibration` refuses it anywhere else. Without
  one, the reranker's hand-set floor applies.
- **Encoder** (`encode.py`): `BAAI/bge-small-en-v1.5` by default for `--encoder st` (the same size
  as the earlier `all-MiniLM-L6-v2`, higher on MTEB retrieval), with the retrieval instruction its
  model card recommends in front of queries and never in front of passages.
- **Metrics** (`metrics.py`): hit@k / recall@k / MRR@k / nDCG@k per retrieval setup — judged per
  gold answer span (overlapping chunks holding the same sentence count once), per chunk, or per
  document (BEIR, deepening retrieval until k documents, graded gains as trec_eval). Answer-level:
  answered, **correct** (the delivered answer contains a gold span — groundedness alone is not
  correctness), abstained on unanswerable, drafts with a refused claim, cited; every rate printed
  with its counts and "n/a" when there is nothing to divide by. Latency p50/p95/p99 with n, each
  percentile flagged when n is too small for it (p95 under 20, p99 under 100).

## What has been verified, and how

| | State |
|---|---|
| Chunking, BM25, cosine, RRF, rerank + relevance floor, calibration, the citation gate, metrics, CLI | **67 tests, strict mypy over `src` and `tests`, ruff clean**, fully offline. Hand-computed checks: an exact BM25 score, RRF's rank-only arithmetic, nDCG with more relevant items than k and with graded gains; the lexical reranker beats keeping the fused order, which beats reversing it |
| Synthetic wiki (`entsearch demo`, 9 docs → 29 chunks, k=3) | answered 12/12 answerable and **12/12 correct**; with the hand-set floor, **abstained on only 6 of 10 unanswerable questions** — all 4 off-topic ones, but only 2 of the 6 near-domain ones ("How many sick days ...?" gets the PTO sentence). **Calibrated on 20 held-out questions** (10 answerable, 10 not, none in the evaluation), it abstains on **8 of 10** and still answers all 12 correctly. Twenty questions is a small calibration set and the lexical reranker a blunt score, so treat the gain as a direction, not a rate. One draft had a refused claim (the legacy VPN's "Its certificates were rotated every 365 days" attached to the current VPN); the user's answer dropped it. Latency n=22, p99 flagged |
| Citation gate (lexical) | **19/20** labelled claims judged right — wrong numbers, borrowed numbers, flipped negations, a swapped entity, an antonym swap, an FAQ question as evidence, a flipped second predicate, two title-pooling claims, a time swap, IT vs HR, "may" for "are required", a number from another document. **Missed, on purpose and asserted by a test:** a true paraphrase ("get 20 days of vacation") — the NLI verifier's job |
| Retrieval ablation | Measured per setup. On paraphrased questions, fusing BM25 with the no-semantics hashing encoder *hurt* (hit@3 1.0 → 0.5) — on two questions, at one chunk size: a hypothesis (fusion pays only with a learned encoder), not a result. **14 questions cannot rank retrievers** — real numbers come from `entsearch eval --beir <dir>` |
| `AnthropicGenerator` | **Real SDK over mock HTTP**: request shape (`claude-opus-5-5`, fallbacks, beta header, effort, no `temperature`), thinking block skipped, refusal and max_tokens raised. Never called the real API |
| `SentenceTransformerEncoder`, `CrossEncoderReranker`, `NliVerifier`, `MiniCheckVerifier` | Real code, not yet run (no model is downloaded on the build machine); MiniCheck is tested through a stand-in scorer; the NLI verifier's label-order reading runs through its real loading path against a stand-in `sentence_transformers` module |

## Review: what an adversarial review found, and what changed

| Finding | Fix |
|---|---|
| Gate was bag-of-words over the whole chunk: passed "5 days" (corpus: 20), "not disabled", "never uses WireGuard", and "It is." | one supporting sentence, full content coverage, every number with its polarity, clause-scoped negation; no-content sentences are not claims |
| Gate refused correct `[n]`-cited answers, the mandated abstention, and passed an empty answer | markers and "according to / based on the context" preambles stripped; abstention recognised; empty answer unsupported |
| Never abstained: "Who won the 2022 World Cup?" came back SUPPORTED with citations | reranker relevance floor → abstain before generating |
| Eval was a tautology: 5 one-chunk docs, k=5 → recall 1.0 even with a reversed reranker | 29 chunks, hard negatives, paraphrased and unanswerable questions, per-setup ablation, a test that a reversed reranker scores worse, BEIR loader with document-level scoring |
| FIXED overlap hard-coded at 20 words; no word fallback (500-word chunks); decimals split; offsets off by whitespace; duplicate ids served the wrong text | overlap in words (default size/5, validated); every strategy bounded by `size`; abbreviation/decimal-safe splitter; exact offsets; duplicate ids refused |
| Retired default model; BM25 idf dead code; p95/p99 from 5 samples | `claude-opus-5-5` with refusal fallback; Lucene idf documented; percentiles carry n and a caveat |

### Second review

A second adversarial round against the reworked code found the gate held for the tested strings
but not their neighbours, and that the evaluation still could not fail. Fixed:

| Finding | Fix |
|---|---|
| Unpunctuated bullets were one "sentence": the borrowed-number defect came back on a bulleted page | blank lines, bullets and headings end sentences; markers stripped |
| The title was pooled into every sentence: "Expenses are not reimbursed" and "Remote access is disabled" were supported | the title only replaces a leading pronoun |
| Negation compared as one set per text; elliptical negation ("employees are not") ignored; IT/US/AM, must/may, option letters, units and times invisible | the shared `text.py` clause analysis (see [meeting-intelligence](https://github.com/madhusudanpatnaik/meeting-intelligence)'s README): per-clause polarity, elliptical negation, acronyms, modals, single letters, units, clock times, numbers bound to words |
| Answer metrics measured groundedness only: a wrong answer scored perfect | a correctness rate against gold spans; only verified claims delivered |
| "4/4 abstained" rested on a floor 0.007 above one question; near-domain questions were answered | six near-domain unanswerables added; the result (6/10) reported as measured |
| BEIR document scoring capped at 20 chunks; graded qrels collapsed to binary; duplicate chunks of one gold sentence counted as two | deepen until k documents; graded gains; span-level recall |
| BM25, RRF, nDCG cut-off, the NLI label reading and the reranker comparison could be broken with tests green | hand-computed and real-path tests |
| p99 unflagged from n=20; "0.000" printed for rates with no cases | per-percentile flags; "n/a (0 cases)" |

### Review before publishing (2026-10-17)

| Finding | Fix |
|---|---|
| `calibrate` without `--docs` fitted the synthetic wiki and its questions, and this README then applied the result to BEIR: a calibration of the wrong corpus | `calibrate --beir`/`--docs` requires your own `--questions`; a calibration records its corpus and `--calibration` refuses it on another |
| An install group (`faiss`) that no code used | removed |
| MiniCheck installed from its repository's moving `main` | pinned to a commit |
| A missing optional library or API key raised a traceback | one line saying which requirements file or variable is needed |

## Why it's built this way — checked against the literature (2026-10)

| Decision | Evidence | Verdict |
|---|---|---|
| **Hybrid retrieval: BM25 + dense, fused by RRF (c = 60)** | BEIR (Thakur et al., 2021): BM25 is a strong zero-shot baseline that many dense retrievers fail to beat out of domain, so lexical and semantic signals are complementary. RRF (Cormack et al., SIGIR 2009) fuses ranks, so a cosine and a BM25 score never need calibrating; c = 60 is the paper's constant, used here unchanged | **Holds.** Caveat: tuned weighted fusion can beat plain RRF on a given corpus, so the per-setup ablation (`entsearch eval`) is the arbiter, not the default |
| **Cross-encoder re-ranking after fusion** | Reading query and passage together (Nogueira & Cho, 2019) is the standard precision stage after first-stage retrieval | **Holds.** `ms-marco-MiniLM-L-6-v2` is a small, fast reranker; larger ones trade latency for accuracy, measured by the same harness |
| **Encoder: `bge-small-en-v1.5`** (was `all-MiniLM-L6-v2`) | MiniLM is small, fast, ubiquitous — and from 2021. Newer models of the same size score higher on MTEB retrieval (Muennighoff et al., 2023); bge-small-en-v1.5 reports 51.68 on its card. Aggregate MTEB scores can mislead, so the choice should be confirmed on the target corpus | **Fixed** — the default, with the query instruction its card recommends; `--encoder-model` picks another. Chosen on published MTEB numbers: confirm on BEIR with `entsearch eval --beir ... --encoder st` |
| **Every answer sentence is a claim checked against one retrieved sentence; only verified claims are delivered** | ALCE (Gao et al., 2023) evaluates citations exactly this way — does the cited passage entail the statement — with an NLI model as the judge. Delivering only verified claims, and abstaining when none survive, puts precision first | **Holds as a design** |
| **The verifier: lexical by default, an NLI cross-encoder optionally** | MiniCheck (Tang et al., EMNLP 2024): a 770M model trained for grounding reaches GPT-4 accuracy on LLM-AggreFact at a fraction of the cost. A general NLI model (`nli-deberta-v3-base`) is not trained for that task, and the lexical gate refuses every paraphrase | **Fixed:** `--verify minicheck` (`MiniCheckVerifier`) is the research-backed verifier. The lexical gate stays the default for the offline build: a precision-first floor (19/20 on its own gate set, the miss a designed paraphrase) |
| **Metrics: nDCG with linear gains, recall per gold span, answer correctness separate from groundedness, latency percentiles with their n** | Linear gains are what trec_eval and BEIR compute; a grounded answer can still be wrong, so correctness is scored on its own | **Holds** |

**Done after this review (2026-10):** the MiniCheck verifier; the bge-small encoder with its query
instruction; abstention from a calibrated reranker score (6/10 → 8/10 on the synthetic held-out
test). **Still open:** running the learned models and confirming the encoder on BEIR — they need
the model downloads this build does not make.

**References**
- Thakur et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
  Models.* NeurIPS 2021 Datasets and Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
- Cormack, Clarke and Büttcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
  Learning Methods.* SIGIR 2009. [doi:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
- Nogueira and Cho. *Passage Re-ranking with BERT.* 2019. [arXiv:1901.04085](https://arxiv.org/abs/1901.04085)
- Muennighoff et al. *MTEB: Massive Text Embedding Benchmark.* EACL 2023.
  [arXiv:2210.07316](https://arxiv.org/abs/2210.07316)
- Gao et al. *Enabling Large Language Models to Generate Text with Citations* (ALCE). EMNLP 2023.
  [arXiv:2305.14627](https://arxiv.org/abs/2305.14627)
- Tang, Laban and Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.*
  EMNLP 2024. [arXiv:2404.10774](https://arxiv.org/abs/2404.10774)

## Known gaps / to run for real

- **The lexical gate is strict on wording.** A model's paraphrase ("get 20 days of PTO") is refused;
  the generator is told to use the passage's words, and `--verify minicheck` is the production
  path (wired to MiniCheck's published API and tested through a stand-in; not yet run).
  Meaning carried only by word order (no number to bind) or by verbs outside its lists is not seen
  ([meeting-intelligence](https://github.com/madhusudanpatnaik/meeting-intelligence)'s README lists these limits, each asserted by a test there).
- **Abstention is only as good as the reranker's score.** Calibration turns a score into a
  probability; it cannot separate questions the score does not separate. Two near-domain questions
  still pass with the lexical reranker; calibrate the cross-encoder on your own held-out questions.
- **Run on a real benchmark** (`entsearch eval --beir scifact --encoder st --rerank cross`) — the
  synthetic numbers check the harness, not the retrievers.
- **No ACL filtering is wired** though `Document.metadata` carries the hook — filter by the caller's
  permissions *before* ranking.
- **Exact cosine over a numpy matrix** is fine to ~1e5 chunks; beyond that it needs an
  approximate index (faiss, HNSW), which is not wired.

## More projects

One of five enterprise AI builds, each a standalone package with its own tests and CI:

- [guarded-support-agent](https://github.com/madhusudanpatnaik/guarded-support-agent): a tool-using agent with a gate on every call, human review of irreversible actions and rollback
- [document-intelligence](https://github.com/madhusudanpatnaik/document-intelligence): contracts and invoices to validated JSON: OCR vs. layout parsing vs. a vision model
- [enterprise-search-rag](https://github.com/madhusudanpatnaik/enterprise-search-rag) (this repository): hybrid retrieval with re-ranking, verified citations and calibrated abstention
- [meeting-intelligence](https://github.com/madhusudanpatnaik/meeting-intelligence): transcription, diarization and summaries whose action items are checked against the transcript
- [sovereign-llm](https://github.com/madhusudanpatnaik/sovereign-llm): fine-tune, quantize and serve a small open model, measured against a frontier API

MIT licensed: see [LICENSE](LICENSE).
