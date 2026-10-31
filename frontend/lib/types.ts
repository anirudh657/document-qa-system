/**
 * Wire types, mirroring `src/rag/api/schemas.py`.
 *
 * These are the API's contract, not the backend's domain models — schemas.py keeps them
 * separate on purpose so a pipeline refactor is not a breaking API change. Keep this
 * file in step with that one and nothing else.
 *
 * Verified against the backend on 2026-10-31. If a field here stops matching, the
 * symptom is a silent `undefined` at runtime rather than a type error, so check
 * /docs (the generated OpenAPI page) when something renders blank.
 */

/** From `rag.models.Citation`. */
export interface Citation {
  chunk_id: string;
  source_path: string;
  section: string | null;
  page_number: number | null;
  quote: string;
}

export interface QueryRequest {
  question: string;
  /** 1–20, rejected by the API outside that range. */
  top_k?: number;
  filters?: Record<string, unknown> | null;
  stream?: boolean;
}

export interface QueryResponse {
  query_id: string;
  answer: string;
  citations: Citation[];
  /**
   * False when the system declined because nothing retrieved supported an answer.
   * This is a correct outcome, not an error — render it as one.
   */
  answered: boolean;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  cache_hit: boolean;
}

export interface SearchHit {
  chunk_id: string;
  text: string;
  score: number;
  page_number: number | null;
  source: string;
}

export interface SearchResponse {
  query: string;
  hits: SearchHit[];
}

export interface HealthResponse {
  /** "ok" or "degraded" — degraded still returns 200, so check this field, not the status code. */
  status: string;
  dependencies: Record<string, string>;
  index_fingerprint: string;
}

export interface DocumentSummary {
  source: string;
  chunks: number;
  /** Distinct pages covered, counting the full span of every chunk. */
  pages: number;
}

export interface DocumentList {
  documents: DocumentSummary[];
  total_chunks: number;
}

export interface UploadResponse {
  source: string;
  chunks: number;
  pages: number;
  /** True when a document of this name was already indexed and has been rebuilt. */
  replaced: boolean;
  warning: string;
}
