# Architecture — Document QA System

## 1. System overview

Two independent paths share one index.

**Ingestion (offline, asynchronous).** Documents enter through a loader, are parsed and
chunked, embedded in batches, and written to the vector store with their metadata. This
path is throughput-sensitive and latency-insensitive.

**Query (online, synchronous).** A question is normalised and optionally rewritten, then
searched two ways in parallel, fused, reranked, packed into a prompt, and streamed to the
model. This path is latency-sensitive and its budget is spent carefully.

```
INGESTION                                QUERY
──────────                               ─────
  documents                                question
      │                                        │
      ▼                                        ▼
  ┌────────┐                             ┌──────────────┐
  │ loader │  pdf/md/html/docx           │ query        │  rewrite, expand,
  └────────┘                             │ transformer  │  extract filters
      │                                  └──────────────┘
      ▼                                        │
  ┌────────┐                        ┌──────────┴──────────┐
  │chunker │  structure-aware       ▼                     ▼
  └────────┘                  ┌─────────┐          ┌────────────┐
      │                       │  BM25   │          │   dense    │
      ▼                       │ sparse  │          │  vector    │
  ┌──────────┐                └─────────┘          └────────────┘
  │ embedder │  batched             │                     │
  └──────────┘                      └──────────┬──────────┘
      │                                        ▼
      ▼                                  ┌───────────┐
  ┌──────────────────┐                   │    RRF    │  reciprocal rank fusion
  │  VECTOR STORE    │◄──────────────────│  fusion   │
  │  (Qdrant)        │──────────────────►└───────────┘
  │  vectors +       │                         │
  │  payload/metadata│                         ▼
  └──────────────────┘                   ┌───────────┐
      ▲                                  │ reranker  │  cross-encoder, top 50 → 5
      │                                  └───────────┘
  ┌──────────────────┐                         │
  │  POSTGRES        │                         ▼
  │  doc registry,   │                   ┌───────────┐
  │  hashes, runs    │                   │  prompt   │  context packing + citations
  └──────────────────┘                   │  builder  │
                                         └───────────┘
                                               │
  ┌──────────────────┐                         ▼
  │  REDIS           │◄───────────────────┌──────────┐
  │  semantic cache, │  cache check       │   LLM    │  streamed
  │  rate limits     │  before retrieval  └──────────┘
  └──────────────────┘                         │
                                               ▼
                                      answer + citations + trace id
```

## 2. Components

### 2.1 Ingestion

**Loaders** (`src/rag/ingestion/loaders.py`) normalise every input format into a common
`Document` with raw text plus structural markers. PDF extraction keeps page numbers;
Markdown and HTML keep heading hierarchy. Structure is not decoration — it is what makes
the chunker good.

**Chunker** (`src/rag/ingestion/chunking.py`) splits on structural boundaries first
(headings, sections), then by token budget with overlap, never mid-sentence. Each chunk
carries a *contextual header*: the document title and its section path, prepended before
embedding. A chunk that reads "The window is 30 days" is useless in isolation; the same
chunk headed "Refund Policy > Timelines" is retrievable.

Parent-document retrieval is supported: small chunks are embedded for precision, but the
larger parent section is what gets sent to the model.

**Indexing pipeline** (`src/rag/ingestion/pipeline.py`) batches embedding calls, computes
a content hash per document, skips unchanged documents, and upserts by deterministic
chunk ID so re-runs do not duplicate.

### 2.2 Retrieval

**Dense** — embedding similarity over the vector store. Good at paraphrase and concept
matching, bad at exact identifiers, product codes and rare proper nouns.

**Sparse** — BM25. Exactly the opposite failure profile. This is why both exist.

**Fusion** — reciprocal rank fusion over the two ranked lists. RRF is chosen over score
interpolation because it needs no score normalisation between two scoring schemes that
are not comparable.

**Reranking** — a cross-encoder scores each `(query, passage)` pair jointly rather than
comparing two independently-computed vectors. This is far more accurate and far more
expensive, which is why it runs on ~50 candidates rather than the whole corpus. It is
usually the single largest quality win in the whole system.

**Query transformation** — short or ambiguous questions are rewritten before retrieval.
HyDE (generate a hypothetical answer, embed *that*) is included behind a flag because it
helps on some corpora and hurts on others; the eval harness decides.

### 2.3 Generation

The prompt builder packs reranked passages into a fixed token budget, each tagged with an
ID the model is instructed to cite. Passages are ordered with the strongest at the start
and end of the context window, since attention degrades in the middle.

The system prompt requires that unsupported claims be omitted and that the model state
plainly when the context does not contain the answer. Citation IDs are validated against
the retrieved set after generation; a hallucinated citation is a bug, not a warning.

### 2.4 Observability

Every request gets a trace ID that flows through retrieval, reranking and generation.
Spans record what was retrieved, what the reranker did to the ordering, the exact prompt
sent, token counts and cost. `GET /v1/query/{id}/trace` exposes this, which is what makes
a bad answer debuggable instead of mysterious.

## 3. Data flow — one query, step by step

1. Request arrives; trace ID assigned; rate limit checked in Redis.
2. Semantic cache lookup — embed the question, search cached question embeddings above a
   similarity threshold. On hit, return immediately.
3. Query transformation: normalise, optionally rewrite, extract any metadata filters.
4. Parallel dense and sparse retrieval, ~25 candidates each.
5. RRF fusion into a single ranked list of ~50.
6. Cross-encoder rerank; keep top 5.
7. Prompt assembly within the token budget.
8. Stream from the LLM.
9. Validate citations against the retrieved set.
10. Write the trace, token counts and cost; populate the cache.

## 4. Technology choices

| Layer | Choice | Reasoning |
|---|---|---|
| Language | Python 3.11 | The retrieval and eval ecosystem lives here |
| API | FastAPI | Async, native streaming, Pydantic validation, generated OpenAPI |
| Vector store | Qdrant | Runs locally in Docker, native hybrid search and metadata filtering, no managed-service dependency |
| Embeddings | `bge-base-en-v1.5` (swappable) | Strong open baseline; local means no per-chunk API cost during development |
| Sparse | BM25 | Complements dense retrieval where it is weakest |
| Reranker | `bge-reranker-base` cross-encoder | Largest quality gain per unit of effort |
| Metadata store | Postgres | Document registry, content hashes, eval run history |
| Cache / limits | Redis | Semantic cache and rate limiting in one dependency |
| Tracing | OpenTelemetry | Vendor-neutral; export anywhere later |
| Eval | Ragas + custom harness | Ragas for standard metrics, custom for retrieval metrics and thresholds |
| Orchestration | None (direct calls) | A framework here would hide the parts of the system this project exists to demonstrate |
| Container | Docker + docker compose | One command to bring up the full dependency set |
| CI | GitHub Actions | Lint, type-check, test, and run eval on every PR |

The orchestration row is a deliberate decision worth defending in a write-up: LangChain
or LlamaIndex would make Phase 1 faster and Phases 2–3 harder to reason about, because
the retrieval internals this project is about would sit behind an abstraction.

## 5. Failure modes and handling

| Failure | Behaviour |
|---|---|
| Vector store unreachable | 503 with a clear message; health endpoint reports the dependency down |
| Reranker times out | Fall back to fusion order, flag it in the trace, serve the answer |
| LLM rate-limited | Exponential backoff with jitter, then a typed error the client can act on |
| Empty retrieval result | Explicit "not found in corpus" — never a free-form guess |
| Malformed or corrupt document at ingest | Skip, log with the file path, continue the batch |
| Embedding model changed | Index version mismatch detected at startup; refuse to serve stale vectors |

## 6. Scaling notes

Nothing here is built for scale yet, deliberately. The path if it were needed: move
ingestion to a worker queue, shard the Qdrant collection, put the reranker behind its own
GPU service, and move the semantic cache to a shared cluster. None of this belongs in v1
and none of it should be built before the load test in Phase 5 shows where the actual
limit is.
