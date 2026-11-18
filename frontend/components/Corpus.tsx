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
 * The corpus rail: what is indexed, how to add to it, and which part of it to search.
 *
 * Present on every screen because it is the answer to the question a reader actually
 * has first — "what can I ask about?" — and because a document tool that does not show
 * its documents is asking people to guess.
 *
 * Passing `onSelect` turns the list into a scope picker: one document, or all of them.
 * Without it the list is read-only, so a screen that has no use for scoping does not
 * grow a control that does nothing.
 */
export default function Corpus({
  onChanged,
  selected = null,
  onSelect,
}: {
  onChanged?: () => void;
  /** The source currently scoped to, or null for the whole corpus. */
  selected?: string | null;
  onSelect?: (source: string | null) => void;
}) {
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null);
  const [total, setTotal] = useState(0);
  const [upload, setUpload] = useState<Upload>({ kind: "none" });
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  // Read through refs inside `refresh` so neither prop joins its dependency list.
  // `onSelect` is usually an inline arrow, so a new identity on every parent render
  // would re-create `refresh`, re-fire the effect below, and refetch forever.
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  const refresh = useCallback(async () => {
    try {
      const list = await listDocuments();
      setDocuments(list.documents);
      setTotal(list.total_chunks);

      // A scope pointing at a document that is no longer indexed filters every query
      // down to nothing, and the screen just stops answering with no reason given.
      // Drop it instead.
      const scope = selectedRef.current;
      if (scope !== null && !list.documents.some((doc) => doc.source === scope)) {
        onSelectRef.current?.(null);
      }
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
  const pickable = typeof onSelect === "function";
  const hasDocuments = documents !== null && documents.length > 0;

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

        {/* An explicit "all" row rather than a hidden toggle: the scope is always
            visible, and clearing it is the same gesture as setting it. */}
        {pickable && hasDocuments && (
          <li>
            <button
              type="button"
              className={`doc-pick ${selected === null ? "is-selected" : ""}`}
              aria-pressed={selected === null}
              onClick={() => onSelect?.(null)}
            >
              <span className="doc-name">All documents</span>
              <span className="mono doc-meta">{total}</span>
            </button>
          </li>
        )}

        {documents?.map((doc) =>
          pickable ? (
            <li key={doc.source}>
              <button
                type="button"
                className={`doc-pick ${selected === doc.source ? "is-selected" : ""}`}
                aria-pressed={selected === doc.source}
                title={doc.source}
                onClick={() => onSelect?.(selected === doc.source ? null : doc.source)}
              >
                <span className="doc-name">{doc.source}</span>
                <span className="mono doc-meta">
                  {doc.pages}p · {doc.chunks}
                </span>
              </button>
            </li>
          ) : (
            <li key={doc.source} className="doc">
              <span className="doc-name" title={doc.source}>
                {doc.source}
              </span>
              <span className="mono doc-meta">
                {doc.pages}p · {doc.chunks}
              </span>
            </li>
          ),
        )}
      </ul>

      {pickable && hasDocuments && (
        <p className="pick-hint">Pick one to search just that document.</p>
      )}

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
