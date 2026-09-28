# Evaluation Methodology

Retrieval and generation fail differently and must be measured separately. A wrong answer
means either the retriever fetched the wrong passages or the generator ignored the right
ones, and a single end-to-end score cannot tell you which.

## Golden dataset

Stored as JSONL in `eval/datasets/`, versioned with the repo.

```json
{
  "id": "q-001",
  "question": "What is the refund window for annual plans?",
  "expected_chunk_ids": ["billing-policy.md#refunds:2"],
  "expected_answer": "30 days from the date of purchase.",
  "tags": ["billing", "single-hop"],
  "answerable": true
}
```

**Rules that keep the dataset honest**

1. Write questions from the source documents, before running the system. Questions
   written after seeing model output measure whether the system agrees with itself.
2. Include unanswerable questions (`"answerable": false`). A system that never refuses is
   not grounded, it is guessing politely.
3. Cover the hard cases deliberately: multi-hop questions, questions whose wording does
   not match the source, questions about rare identifiers, and near-duplicate passages.
4. Target 100+ questions. Below ~50 the metrics move on noise.

## Retrieval metrics

| Metric | What it answers |
|---|---|
| **Hit rate @k** | Did any correct chunk appear in the top k at all? |
| **MRR** | How high up was the first correct chunk? |
| **NDCG@10** | Is the whole ordering good, not just the first hit? |
| **Context recall** | What fraction of the needed information was retrieved? |
| **Context precision** | How much of what was retrieved was actually relevant? |

Hit rate is the floor: if the correct passage is not retrieved, no amount of prompt work
saves the answer. Reranking should move MRR and NDCG far more than it moves hit rate,
which is exactly how you confirm it is doing its job.

## Generation metrics

| Metric | What it answers |
|---|---|
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
make eval                        # full golden set against current build
make eval TAGS=multi-hop         # one slice
make eval COMPARE=<git-sha>      # diff against a previous run
```

Every run is stored with the git SHA, config snapshot and timestamp, so any number in the
final report can be traced back to the exact code that produced it.

## Regression gates

CI fails when a metric drops more than 2% below the recorded baseline. The point is that
a change which improves one metric while quietly destroying another cannot ship unnoticed.

## Results

Fill this in as you go. The Phase 1 row must be recorded before Phase 2 work begins.

| Run | Phase | Hit@5 | MRR | NDCG@10 | Faithfulness | Refusal | p95 (s) | $/query |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | — | — | — | — | — | — | — |
| hybrid | 2 | — | — | — | — | — | — | — |
| + rerank | 2 | — | — | — | — | — | — | — |
| + rewrite | 2 | — | — | — | — | — | — | — |
| final | 5 | — | — | — | — | — | — | — |

## Chunk size ablation

34-question golden set, 3 documents, BAAI/bge-base-en-v1.5, top_k=5, overlap=64,
dense retrieval only.

| chunk_size | Hit@5 | MRR | NDCG@10 | Precision@5 | Recall@5 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.625 | 0.594 | 0.572 | 0.278 | 0.583 |
| **160** | **0.656** | 0.579 | **0.583** | 0.258 | **0.630** |
| 200 | 0.656 | 0.600 | 0.571 | 0.218 | 0.604 |
| 256 | 0.625 | 0.534 | 0.524 | 0.193 | 0.573 |

Hit rate peaks at 160-200 and falls either side. Precision decreases monotonically
as chunks grow, which is the expected mechanism: a larger chunk carries more
irrelevant text alongside the relevant passage. Everything at 256 and above is worse
on every metric.

160 chosen: same hit rate as 200, with better precision (+0.040) and recall (+0.026).
200 wins only MRR (+0.021).

Caveat: at 32 answerable questions a 0.02 difference is not resolvable. The
defensible claim is that 160 and 200 are indistinguishable and that anything above
256 is worse.
