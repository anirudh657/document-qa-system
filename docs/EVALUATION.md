# Evaluation Methodology

Retrieval and generation fail differently and must be measured separately. A wrong answer
means either the retriever fetched the wrong passages or the generator ignored the right
ones, and a single end-to-end score cannot tell you which.

## Golden dataset

Stored as JSONL in `eval/datasets/`, versioned with the repo.

```json
{
  "id": "q-019",
  "question": "How do I select every column from a table?",
  "source": "MySQL Handbook.pdf",
  "expected_pages": [14],
  "tags": ["mysql", "literal"],
  "answerable": true
}
```

**Questions point at pages, not chunk IDs.** Chunk IDs derive from the document hash and
the chunk's position, so changing chunk size — the single most important thing to ablate —
renumbers every chunk and invalidates the whole dataset. A golden set you must rewrite to
run the experiment is not a golden set. Pages are stable: a fact on page 5 stays on page 5
whatever the chunker does, so one dataset scores a 128-token run and a 512-token run and
the comparison is real.

**Rules that keep the dataset honest**

1. Write questions from the source documents, before running the system. Questions
   written after seeing model output measure whether the system agrees with itself.
2. Include unanswerable questions (`"answerable": false`). A system that never refuses is
   not grounded, it is guessing politely.
3. Cover the hard cases deliberately: multi-hop questions, questions whose wording does
   not match the source, questions about rare identifiers, and near-duplicate passages.
4. Target 100+ questions. See **Statistical power** below for why 34 is not enough — this
   is the project's binding constraint, not a nicety.

## Statistical power

With 32 answerable questions, one question is worth **0.031** of hit@5, and hit@5 = 0.656
carries a 95% confidence interval of **[0.49, 0.82]**. Reporting it to three decimals
implies a precision that is not there.

Two configurations are compared on the same questions, so the right test is **McNemar's
exact test** on the questions where they disagree. Only discordant pairs carry information:

| Fixed | Broken | p | Verdict |
| --- | --- | --- | --- |
| 3 | 0 | 0.250 | indistinguishable from chance |
| 4 | 0 | 0.125 | indistinguishable from chance |
| 5 | 0 | 0.062 | indistinguishable from chance |
| 6 | 0 | 0.031 | significant |
| 8 | 0 | 0.008 | significant |
| 7 | 1 | 0.070 | indistinguishable from chance |

**The minimum detectable win at this sample size is 6 clean fixes with zero regressions.**
A technique whose addressable pool is smaller than 6 cannot be proven here no matter how
well it works. Report the fixed/broken counts and the p-value alongside any metric delta;
a table of three-decimal numbers without them invites conclusions the data cannot support.

Required sample size to detect a 10-point real improvement: ~120 questions.

## Retrieval metrics

| Metric | What it answers |
| --- | --- |
| **Hit rate @k** | Did any correct page appear in the top k at all? |
| **MRR** | How high up was the first correct page? |
| **NDCG@10** | Is the whole ordering good, not just the first hit? |
| **Recall@k** | What fraction of the expected pages were retrieved? |
| **chunk_precision@k** | How many of the k retrieved *chunks* covered an expected page? |
| ~~precision@k~~ | **Do not use.** See below. |

Hit rate is the floor: if the correct passage is not retrieved, no amount of prompt work
saves the answer. Reranking should move MRR and NDCG far more than it moves hit rate,
which is exactly how you confirm it is doing its job.

**Why `precision@k` is not trustworthy on this corpus.** It divides by the number of
distinct retrieved *pages*, not by k. Smaller chunks land several pieces on the same page,
the page list collapses, the divisor shrinks, and the score rises with no change in what
was found. Demonstrated: identical retrieval with the correct page at rank 1 both times
scored 0.200 with large chunks and 0.500 with small ones. `chunk_precision@k` divides by
the number of chunks actually sent — a fixed k — and measures the thing precision is for,
which is how much of the context window was wasted.

## Generation metrics

| Metric | What it answers |
| --- | --- |
| **Faithfulness** | Is every claim supported by the retrieved context? |
| **Answer relevance** | Does it actually answer the question asked? |
| **Correct refusal rate** | Does it decline when the corpus has no answer? |
| **Citation validity** | Do all cited IDs exist in the retrieved set? |

Citation validity is a hard pass/fail, not a score. A citation to a chunk that was never
retrieved is a bug.

## LLM-as-judge calibration

Judges are themselves unreliable, so calibrate before trusting them: hand-label 30
questions, run the judge on the same 30, and report agreement. Below ~85% agreement the
judge's scores are not evidence and the prompt needs work.

## Running

```bash
rag eval                                     # full golden set against current build
rag eval --tags multi-hop                    # one slice
rag eval --baseline eval/runs/<run>.json     # diff against a previous run
rag warm-rewrites                            # populate the rewrite cache first
```

An eval run must make no live API calls: those are slow, rate-limited and
non-deterministic, and a measurement must be none of the three. Query rewrites are
pre-computed into a cache by `rag warm-rewrites`, and `rag eval` reads from it. When
rewrites are missing, the run prints a warning and the numbers do **not** measure
rewriting — read that warning before recording anything.

Every run is stored with the git SHA, config snapshot and timestamp. The snapshot records
which retrieval stages were active (`hybrid`, `rerank`, `query_rewrite`), which model wrote
the rewrites, and `page_spans` — see below for why that last flag divides the results into
two incomparable eras.

## Regression gates

CI fails when a metric drops more than 2% below the recorded baseline. The point is that
a change which improves one metric while quietly destroying another cannot ship unnoticed.

---

# Results

## Current

34-question golden set (32 answerable, 2 refusal), 3 documents, 148 chunks.
`BAAI/bge-base-en-v1.5`, chunk_size 160, overlap 64, top_k 5.

| Run | Date | Hit@5 | MRR | NDCG@10 | Recall@5 | chunk_prec@5 | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| page-span attribution | 2026-10-31 | **0.906** | **0.677** | **0.714** | 0.880 | 0.406 | 3 |
| single-page attribution | 2026-09-28 | 0.656 | 0.579 | 0.583 | 0.630 | — | 11 |

**8 questions fixed, 0 broken. McNemar exact p = 0.0078.** The first statistically
significant result in this project.

Fixed: `q-008`, `q-017`, `q-020`, `q-022`, `q-023`, `q-031`, `q-032`, `q-033`.

### What the fix was

A chunk was attributed to the page its first unit *started* in, and nothing else. This
corpus averages ~132 tokens of text per page against a chunk_size of 160, so a chunk
routinely spans two or three pages — and a page shorter than the chunk stride gets no
chunk starting inside it at all. Ten pages had no chunk carrying their label, and seven
golden questions were scored as retrieval failures while their text sat in the index under
the previous page's number. Every mislabel was off by exactly one page.

`Chunk.page_end` now records the span and `retrieved_keys` credits every page a chunk
covers. This corrects a label rather than loosening a metric: if a chunk's text contains
page 15, retrieving that chunk did surface page 15.

Index coverage went from 96 to 106 distinct pages; `MySQL Handbook` from 74 of 80 pages to
80 of 80. Unreachable golden questions: 7 → 0.

### Caveats on this result

**Two variables moved, not one.** The 0.656 baseline was dense-only; this run had hybrid
retrieval enabled. Hybrid previously measured as *identical to three decimal places* at
chunk 200 (see below), so it is very unlikely to explain any of the +0.250 — but that is an
inference, not a measurement. **Outstanding: one run at chunk 160 with
`enable_hybrid=False` to attribute the gain cleanly.**

**Query rewriting was not measured.** 25 of 32 rewrites failed (the pinned model
`gemini-2.5-flash-lite` was retired mid-run, HTTP 404), so these numbers are effectively
retrieval without rewriting. That is what makes 0.906 a clean attribution-only result.

**Span credit is slightly generous at page boundaries.** A chunk spanning pages 14–15 is
credited with page 15 even if it caught only its first line. The retrieval did surface part
of that page, so the credit is not wrong, but whether that part contains the *answer* is a
question retrieval metrics cannot answer — that is faithfulness, and needs a calibrated
judge. MRR and NDCG rising alongside hit rate is reassuring: boundary luck would lift hit
rate while leaving ordering flat.

### Remaining failures

All three are vocabulary gaps — the query states an intent without using the document's
term. This is exactly what query rewriting targets.

| ID | Question | Expected | Missing term |
| --- | --- | --- | --- |
| `q-019` | How do I select every column from a table? | p14 | `SELECT *` |
| `q-021` | How do I find rows where a column has no value at all? | p16 | `NULL` |
| `q-026` | How do I count only the rows matching a condition? | p31 | `COUNT` |

**Query rewriting is now unprovable on this dataset.** Its entire addressable pool is 3
questions, and 3 clean fixes gives p = 0.250. Growing the golden set is a prerequisite for
the experiment, not an improvement to it.

---

## Invalidated results

Everything below was measured under **single-page attribution**, when seven answerable
questions could not be scored correctly regardless of retrieval quality. The runs are kept
as history; the numbers are not evidence and must not be compared against anything above.
Runs recorded after the fix carry `page_spans: true` in their config snapshot.

### Phase 2 techniques — both measured as negative

34-question golden set, chunk_size 200, top_k 5.

| Run | Hit@5 | MRR | NDCG@10 | Precision@5 |
| --- | --- | --- | --- | --- |
| dense only | 0.656 | 0.600 | 0.571 | 0.218 |
| + hybrid BM25 + RRF | 0.656 | 0.600 | 0.571 | 0.218 |
| + cross-encoder rerank | 0.625 | 0.467 | 0.473 | 0.194 |

Hybrid changed nothing to three decimal places. Reranking made every metric worse and was
disabled.

**Both now have a simpler explanation than the mechanisms originally proposed.** The
correct page was absent from the index at page granularity for 7 of 32 questions, so BM25
had no absent page to match and the reranker had no absent page to promote — it rescored
plausible neighbours of a hole, confidently. Neither technique was given a fair test. Both
need re-measuring against the corrected baseline before any conclusion stands.

### Chunk size ablation — invalid by construction

34-question golden set, dense retrieval only, overlap 64, top_k 5.

| chunk_size | Hit@5 | MRR | NDCG@10 | Precision@5 | Recall@5 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.625 | 0.594 | 0.572 | 0.278 | 0.583 |
| 160 | 0.656 | 0.579 | 0.583 | 0.258 | 0.630 |
| 200 | 0.656 | 0.600 | 0.571 | 0.218 | 0.604 |
| 256 | 0.625 | 0.534 | 0.524 | 0.193 | 0.573 |

This is the worst-affected result, for two independent reasons.

**Chunk size controlled the bug.** How many pages get orphaned is a direct function of the
chunk stride against page length, so these four rows compared configurations whose page
labels were differently wrong — not configurations that retrieved differently.

**The precision column was an artifact and its interpretation was backwards.** The original
note read: *"Precision decreases monotonically as chunks grow, which is the expected
mechanism: a larger chunk carries more irrelevant text alongside the relevant passage."*
That reasoning is wrong. `precision@5` divides by the distinct-page count, so larger chunks
spread the top-k over more distinct pages and inflate the divisor. The monotonicity is the
divisor moving, not relevance. 160 was chosen partly on a +0.040 precision margin that did
not measure anything.

The honest caveat already recorded — that a 0.02 difference is not resolvable at 32
questions — was correct and understated. Every gap in this table is one question or less.

**Re-run this ablation after the fix, with `chunk_precision@5`, before trusting any chunk
size.** 160 remains in use as the incumbent, not as a measured optimum.

---

## Outstanding

| # | Work | Why |
| --- | --- | --- |
| 1 | Re-run at chunk 160 with `enable_hybrid=False` | Attribute the +0.250 to the fix alone |
| 2 | Pin `LLM_MODEL=gemini-3.5-flash-lite` | 2.5-flash-lite was retired mid-run |
| 3 | Grow golden set toward 120 questions | Nothing smaller than 6 clean fixes is detectable |
| 4 | Re-measure hybrid and reranking | Neither got a fair test |
| 5 | Re-run the chunk-size ablation | Invalid by construction |
| 6 | Add a payload schema version to `index_fingerprint()` | A payload change does not currently invalidate the index, so a stale index loads silently |
| 7 | Phase 3 generation metrics | Faithfulness is the only thing that can confirm a retrieved span actually contains the answer |
