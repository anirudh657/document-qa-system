"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, listDocuments, uploadDocument } from "@/lib/api";
import type { DocumentSummary } from "@/lib/types";

type Upload =
  | { kind: "none" }
  | { kind: "busy"; name: string }
  | { kind: "done"; name: string; chunks: number; pages: number; replaced: boolean }
  | { kind: "failed"; message: string };

/**
 * The corpus rail: what is indexed, and how to add to it.
 *
 * Present on every screen because it is the answer to the question a reader actually
 * has first — "what can I ask about?" — and because a document tool that does not show
 * its documents is asking people to guess.
 */
export default function Corpus({ onChanged }: { onChanged?: () => void }) {
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null);
  const [total, setTotal] = useState(0);
  const [upload, setUpload] = useState<Upload>({ kind: "none" });
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await listDocuments();
      setDocuments(list.documents);
      setTotal(list.total_chunks);
    } catch {
      // The rail is context, not the task. A failed listing must not take the page down.
      setDocuments([]);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const send = useCallback(
    async (file: File) => {
      setUpload({ kind: "busy", name: file.name });
      try {
        const result = await uploadDocument(file);
        setUpload({
          kind: "done",
          name: result.source,
          chunks: result.chunks,
          pages: result.pages,
          replaced: result.replaced,
        });
        await refresh();
        onChanged?.();
      } catch (error) {
        setUpload({
          kind: "failed",
          message:
            error instanceof ApiError
              ? error.detail || error.message
              : "That upload failed.",
        });
      }
    },
    [refresh, onChanged],
  );

  const busy = upload.kind === "busy";

  return (
    <aside className="rail">
      <div className="rail-head">
        <h2 className="rail-title">Corpus</h2>
        {documents !== null && (
          <span className="mono rail-count">
            {documents.length} {documents.length === 1 ? "doc" : "docs"} · {total} passages
          </span>
        )}
      </div>

      <ul className="doc-list">
        {documents === null && <li className="doc-empty">Loading…</li>}
        {documents?.length === 0 && (
          <li className="doc-empty">Nothing indexed yet. Add a document below.</li>
        )}
        {documents?.map((doc) => (
          <li key={doc.source} className="doc">
            <span className="doc-name" title={doc.source}>
              {doc.source}
            </span>
            <span className="mono doc-meta">
              {doc.pages}p · {doc.chunks}
            </span>
          </li>
        ))}
      </ul>

      {/* A real <input type="file"> does the work; the drop zone is a label for it, so
          keyboard and click both reach it without a synthetic handler. */}
      <label
        className={`drop ${dragging ? "is-over" : ""} ${busy ? "is-busy" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file && !busy) void send(file);
        }}
      >
        <input
          ref={input}
          type="file"
          className="sr-only"
          accept=".pdf,.txt,.md,.markdown,.html,.htm,.docx"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void send(file);
            event.target.value = "";
          }}
        />
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path
            d="M8 11V3m0 0L5 6m3-3l3 3M2.5 11v1.5A1.5 1.5 0 004 14h8a1.5 1.5 0 001.5-1.5V11"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span>{busy ? `Indexing ${upload.name}…` : "Add a document"}</span>
      </label>

      <p className="drop-hint">PDF, Word, Markdown, HTML or text. Up to 32 MB.</p>

      <div aria-live="polite">
        {upload.kind === "done" && (
          <div className="note note-good">
            <strong>{upload.name}</strong> {upload.replaced ? "re-indexed" : "indexed"} —{" "}
            {upload.pages} {upload.pages === 1 ? "page" : "pages"}, {upload.chunks} passages.
            <span className="note-sub">
              The corpus changed, so recorded evaluations are no longer a like-for-like
              baseline.
            </span>
          </div>
        )}
        {upload.kind === "failed" && (
          <div className="note note-bad">{upload.message}</div>
        )}
      </div>
    </aside>
  );
}
