# Product Requirements Document — Document QA System

| | |
|---|---|
| **Owner** | anirudh657 |
| **Status** | Draft |
| **Last updated** | _fill in when you edit this_ |

---

## 1. Problem

People sitting on a body of documents — internal wikis, contracts, research papers,
course notes, product manuals — cannot get answers out of them quickly. Keyword search
returns documents when what was wanted was an answer, and it fails whenever the asker's
vocabulary differs from the author's. Reading the documents does not scale.

General-purpose language models cannot fill the gap on their own. They have never seen
the private corpus, and when asked about it they produce fluent, confident, wrong
answers. Any usable system must ground every claim in a retrievable passage and show
that passage to the user.

## 2. Goals

**G1 — Grounded answers.** Every answer is derived from retrieved passages and carries
citations that link back to the source document and location.

**G2 — Measurable retrieval quality.** Retrieval performance is tracked as a number on a
fixed question set, so any change can be shown to help or hurt.

**G3 — Honest refusal.** When the corpus does not contain the answer, the system says so
instead of inventing one.

**G4 — Production behaviour.** Bounded latency, bounded cost per query, structured logs,
traces, and a deployment that survives restart.

## 3. Non-goals

- Fine-tuning or training a language model. This is a retrieval and systems project.
- Multi-tenant SaaS with billing, orgs, and seat management.
- A polished consumer UI. The demo interface exists to exercise the API, nothing more.
- Real-time collaborative editing of documents.
- Agentic tool use beyond retrieval.

## 4. Users

| User | Needs | Success looks like |
|---|---|---|
| **Primary — knowledge worker** | Fast, cited answers over a corpus they did not write | Finds the answer in one query and trusts the citation enough not to open the source |
| **Secondary — corpus owner** | Confidence the system is not fabricating | Can inspect what was retrieved for any answer |
| **Operator (me)** | To know when quality regresses | An eval run fails CI before a bad change ships |

## 5. Functional requirements

### 5.1 Ingestion
- **FR-1** Accept PDF, Markdown, HTML, TXT and DOCX inputs.
- **FR-2** Extract text with structure preserved (headings, sections, page numbers).
- **FR-3** Chunk documents so that a chunk is semantically self-contained, and attach
  parent-document context to each chunk.
- **FR-4** Store per-chunk metadata: source path, title, section heading, page number,
  document date, content hash.
- **FR-5** Re-index incrementally — an unchanged document is not re-embedded. Detection
  is by content hash.
- **FR-6** Ingestion is idempotent. Running it twice on the same corpus produces the same
  index and no duplicates.

### 5.2 Retrieval
- **FR-7** Hybrid retrieval combining sparse (BM25) and dense (embedding) search.
- **FR-8** Rerank the fused candidate set with a cross-encoder before generation.
- **FR-9** Support metadata filters (date range, source, document type) at query time.
- **FR-10** Rewrite or expand the query before retrieval when the raw question is short
  or underspecified.

### 5.3 Generation
- **FR-11** Answers cite the specific chunks used, identified by document and location.
- **FR-12** When no retrieved passage supports an answer, return an explicit "not found
  in the corpus" response.
- **FR-13** Stream tokens to the client.
- **FR-14** Never include a citation to a passage that was not in the retrieved set.

### 5.4 API
- **FR-15** `POST /v1/query` — question in, streamed answer plus citations out.
- **FR-16** `POST /v1/ingest` — enqueue a document or directory for indexing.
- **FR-17** `GET /v1/health` — liveness and dependency status.
- **FR-18** `GET /v1/query/{id}/trace` — what was retrieved, reranked and sent to the model.

### 5.5 Evaluation
- **FR-19** A versioned golden set of questions with known correct source passages.
- **FR-20** Retrieval metrics: hit rate @k, MRR, NDCG@10.
- **FR-21** Generation metrics: faithfulness, answer relevance, context precision,
  context recall.
- **FR-22** An eval run in CI that fails the build when a metric drops past a threshold.

## 6. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | End-to-end p95 latency, cached | < 500 ms |
| NFR-2 | End-to-end p95 latency, uncached | < 3.5 s to first token |
| NFR-3 | Cost per query | < $0.01 average |
| NFR-4 | Ingestion throughput | ≥ 100 pages/minute |
| NFR-5 | Availability under load test | 99% over a 30-minute run at 20 rps |
| NFR-6 | Every request carries a trace ID through all components | 100% |
| NFR-7 | Secrets never appear in logs or traces | 100% |

## 7. Success metrics

The project is done when all of the following are true and demonstrable:

| Metric | Baseline (Phase 1) | Target |
|---|---|---|
| Hit rate @5 | measure it | ≥ 0.90 |
| NDCG@10 | measure it | ≥ 0.75 |
| Faithfulness | measure it | ≥ 0.95 |
| Correct refusal rate on unanswerable questions | measure it | ≥ 0.90 |
| p95 latency to first token | measure it | < 3.5 s |
| Cost per query | measure it | < $0.01 |

The Phase 1 baseline is not a guess — it is recorded by running the eval harness against
the naive implementation before any Phase 2 work begins. The before/after table is the
main deliverable of this project.

## 8. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| Golden dataset is too small to be meaningful | Metrics become noise | Target 100+ questions; report confidence intervals |
| Golden dataset is written after seeing model output | Metrics are self-fulfilling | Write questions from the source documents, before running the system |
| Reranker latency blows the p95 budget | NFR-2 fails | Cap candidates at 50; benchmark a smaller reranker; consider GPU or a hosted rerank API |
| LLM-as-judge is unreliable | Generation metrics are untrustworthy | Hand-label a 30-question subset and check judge agreement |
| Embedding model change invalidates the index | Full re-index required | Version the index by embedding model; store model name in collection metadata |
| Chunking strategy is the real bottleneck | Time spent on reranking is wasted | Ablate chunking before optimising retrieval |

**Open questions**
- What is the corpus? The choice drives chunking strategy and how hard the golden set is
  to build. _Decide before Phase 1 ends._
- Self-hosted embeddings or a hosted API? Affects NFR-3 and NFR-4 directly.
- Is a UI in scope at all, or is the API plus eval report the whole deliverable?

## 9. Out of scope for v1

Multi-language corpora, image and table extraction from PDFs, user accounts,
conversation memory across turns, and fine-tuned rerankers. Each is a reasonable v2.
