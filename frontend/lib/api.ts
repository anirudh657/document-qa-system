/**
 * The only module that talks to the API.
 *
 * Every call goes to this app's own origin at /api/*, which next.config.mjs proxies to
 * FastAPI. Nothing here knows the backend's address, so deploying the API elsewhere is
 * one environment variable and no code change.
 */

import type {
  DocumentList,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  SearchResponse,
  UploadResponse,
} from "./types";

/**
 * Carries the HTTP status so callers can tell the two failures apart that matter:
 * 503 means the service is up but has no language model (search still works), while
 * anything else is a real fault.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail || `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** The API is reachable but cannot generate answers. Searching is still possible. */
  get isDegraded(): boolean {
    return this.status === 503;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    // fetch only rejects when the request never got an HTTP response at all.
    throw new ApiError(0, "Could not reach the API. Is it running on port 8000?");
  }

  if (!response.ok) {
    // FastAPI puts the message in `detail`, which is a string for HTTPException and a
    // list of field errors for a 422. Flatten both rather than rendering [object Object].
    let detail = "";
    try {
      const body = await response.json();
      detail = Array.isArray(body?.detail)
        ? body.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ")
        : String(body?.detail ?? "");
    } catch {
      detail = await response.text().catch(() => "");
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

export function ask(body: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify({ top_k: 5, ...body }),
  });
}

/**
 * `filters` narrows retrieval to a subset of the index — `{ source: "MySQL Handbook.pdf" }`
 * searches only that document. The API passes it straight to Qdrant as a payload filter,
 * so the keys are payload field names, not a query language.
 */
export function search(
  question: string,
  topK = 5,
  filters: Record<string, unknown> | null = null,
): Promise<SearchResponse> {
  return request<SearchResponse>("/search", {
    method: "POST",
    body: JSON.stringify({ question, top_k: topK, filters }),
  });
}

export function health(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function listDocuments(): Promise<DocumentList> {
  return request<DocumentList>("/documents");
}

/**
 * Upload one file and index it.
 *
 * No Content-Type header: the browser has to set it itself, because multipart needs a
 * boundary parameter that only it knows. Setting "application/json" here — which the
 * shared `request` helper does — makes the server reject the body as malformed, so this
 * one call goes direct rather than through it.
 */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);

  let response: Response;
  try {
    response = await fetch("/api/documents", { method: "POST", body: form });
  } catch {
    throw new ApiError(0, "Could not reach the API. Is it running on port 8000?");
  }

  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = Array.isArray(body?.detail)
        ? body.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ")
        : String(body?.detail ?? "");
    } catch {
      detail = "";
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as UploadResponse;
}
