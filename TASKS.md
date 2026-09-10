# Task Board

The single source of truth for what is done and what is left. Claude Code reads and
updates this file directly — no integration, no API, just a file in the repo.

**Conventions**
- `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked
- A task is only `[x]` when it is committed, tested and pushed.
- Blocked tasks get a `→ reason` on the same line.

**Progress:** 0 / 78 tasks · Phase 1

---

## Phase 0 — Setup

- [x] Create repository
- [x] Write PRD
- [x] Write architecture document
- [ ] Decide the corpus — this blocks meaningful chunking and eval work
- [ ] Set up local dev environment and confirm `make services` comes up clean
- [ ] Collect 20–50 seed documents into `data/`
- [ ] Confirm LLM provider and get an API key into `.env`

## Phase 1 — Basic RAG

Goal: an end-to-end answer, however crude, plus a recorded baseline score.

- [ ] `Document` and `Chunk` data models
- [ ] PDF loader with page-number retention
- [ ] Markdown and plain text loaders
- [ ] HTML loader
- [ ] Fixed-size chunker with overlap
- [ ] Embedding client with batching
- [ ] Qdrant collection setup and schema
- [ ] Chunk upsert with deterministic IDs
- [ ] Dense top-k retrieval
- [ ] Base prompt template with citation instructions
- [ ] LLM client with streaming
- [ ] `POST /v1/query` endpoint
- [ ] `GET /v1/health` endpoint
- [ ] Ingestion CLI (`make ingest`)
- [ ] End-to-end smoke test
- [ ] **Record the Phase 1 baseline in `docs/EVALUATION.md`** — do not skip this, the
      whole before/after story depends on it

## Phase 2 — Advanced Retrieval

Goal: measurably better retrieval than Phase 1, with each change ablated.

- [ ] Structure-aware chunker (split on headings first)
- [ ] Contextual chunk headers (title + section path prepended before embedding)
- [ ] Parent-document retrieval (embed small, return large)
- [ ] BM25 sparse index
- [ ] Reciprocal rank fusion over dense + sparse
- [ ] Cross-encoder reranker
- [ ] Rerank candidate cap and latency benchmark
- [ ] Query rewriting
- [ ] HyDE behind a feature flag
- [ ] Metadata filtering at query time
- [ ] Ablation table: which of the above actually helped, with numbers

## Phase 3 — Evaluation System

Goal: nothing ships without a score. This is the phase that makes the project credible.

- [ ] Golden dataset schema (question, expected source chunks, expected answer, tags)
- [ ] Write 100+ golden questions from the source documents
- [ ] Include unanswerable questions to test refusal behaviour
- [ ] Hit rate @k
- [ ] MRR
- [ ] NDCG@10
- [ ] Faithfulness (LLM-as-judge)
- [ ] Answer relevance
- [ ] Context precision and context recall
- [ ] Judge calibration — hand-label 30 questions and check agreement
- [ ] Eval runner CLI (`make eval`)
- [ ] Store eval runs in Postgres with a git SHA
- [ ] Regression thresholds that fail CI
- [ ] Results summary rendered to Markdown

## Phase 4 — Production Features

Goal: it behaves like a service, not a script.

- [ ] Streaming responses end to end
- [ ] Semantic cache in Redis
- [ ] Cache invalidation on re-index
- [ ] Rate limiting
- [ ] API key auth
- [ ] Structured JSON logging with trace IDs
- [ ] OpenTelemetry tracing across all components
- [ ] Token and cost accounting per request
- [ ] `GET /v1/query/{id}/trace` endpoint
- [ ] Async ingestion via a worker queue
- [ ] Incremental re-indexing by content hash
- [ ] `POST /v1/ingest` endpoint
- [ ] Citation validation against the retrieved set
- [ ] PII detection and redaction at ingest
- [ ] Graceful degradation when the reranker fails
- [ ] Retry with backoff on LLM rate limits
- [ ] Dockerfile
- [ ] docker compose for the full stack
- [ ] GitHub Actions CI: lint, type-check, test
- [ ] GitHub Actions: eval on PR with threshold gate

## Phase 5 — Final Polish

Goal: the thing that gets shown to someone else.

- [ ] Load test at 20 rps for 30 minutes
- [ ] Profile and fix the p95 latency hotspot
- [ ] Cost-per-query optimisation pass
- [ ] Architecture diagram (rendered, not ASCII)
- [ ] README final pass with real numbers
- [ ] API documentation
- [ ] Minimal demo UI
- [ ] Deploy somewhere public
- [ ] Technical write-up: what worked, what did not, what the numbers say
- [ ] Demo video, 2–3 minutes
- [ ] Final before/after benchmark table

---

## Final Deliverables

- [ ] Deployed, working demo
- [ ] Public repo with a clean README
- [ ] Architecture diagram
- [ ] Evaluation report with before/after numbers
- [ ] Benchmark table: latency, cost, accuracy
- [ ] Technical write-up
- [ ] Demo video
