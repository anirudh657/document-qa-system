"use client";

import { useState } from "react";

import Chevron from "@/components/Chevron";
import Corpus from "@/components/Corpus";
import { pageLabel } from "@/components/PassageRows";
import { ApiError, search } from "@/lib/api";
import type { SearchHit } from "@/lib/types";

const STRONG = 0.65;

type State =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "done"; hits: SearchHit[] }
  | { kind: "failed"; message: string };

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || state.kind === "working") return;

    setState({ kind: "working" });
    try {
      const response = await search(trimmed, 10);
      setState({ kind: "done", hits: response.hits });
    } catch (error) {
      setState({
        kind: "failed",
        message: error instanceof ApiError ? error.detail || error.message : "Search failed.",
      });
    }
  }

  return (
    <div className="layout">
      <Corpus />
      <div className="work">
      <form onSubmit={submit}>
        <div className="ask-field">
          <label htmlFor="query" className="sr-only">
            Search the documents
          </label>
          <input
            id="query"
            type="text"
            value={query}
            maxLength={2000}
            placeholder="Search the documents…"
            onChange={(event) => setQuery(event.target.value)}
            autoComplete="off"
          />
          <button
            type="submit"
            className={`icon-button ${state.kind === "working" ? "is-working" : ""}`}
            disabled={state.kind === "working" || query.trim() === ""}
            aria-label="Search"
          >
            <svg width="17" height="17" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <circle
                cx="8"
                cy="8"
                r="5.2"
                stroke={query.trim() === "" ? "var(--faint)" : "#fff"}
                strokeWidth="1.9"
              />
              <path
                d="M12 12l3 3"
                stroke={query.trim() === "" ? "var(--faint)" : "#fff"}
                strokeWidth="1.9"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
      </form>

      <p className="lede">
        Passages only, no answer written. This is exactly what the model would have been
        given — useful when an answer looks wrong, because it separates &ldquo;never found
        it&rdquo; from &ldquo;found it and ranked it badly&rdquo;.
      </p>

      <div aria-live="polite" aria-busy={state.kind === "working"}>
        {state.kind === "failed" && (
          <p style={{ margin: 0, fontSize: 14, color: "var(--ink-soft)" }}>{state.message}</p>
        )}

        {state.kind === "done" && <Results hits={state.hits} />}
      </div>
      </div>
    </div>
  );
}

function Results({ hits }: { hits: SearchHit[] }) {
  if (hits.length === 0) {
    return (
      <p style={{ margin: 0, fontSize: 14, color: "var(--ink-soft)" }}>
        No passage matched. Nothing in the corpus is close to this.
      </p>
    );
  }

  return (
    <>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 12,
          marginBottom: 4,
        }}
      >
        <h2 className="label">
          {hits.length} {hits.length === 1 ? "passage" : "passages"}, best first
        </h2>
      </div>

      {hits.map((hit) => {
        const strong = hit.score >= STRONG;
        return (
          <details key={hit.chunk_id} className="disclosure">
            <summary>
              <span
                className="mono"
                style={{
                  flex: "none",
                  width: 40,
                  fontSize: 13,
                  fontWeight: 500,
                  color: strong ? "var(--ink)" : "#a5a5ae",
                }}
              >
                {hit.score.toFixed(2)}
              </span>
              <div className="track" style={{ flex: "none", width: 64 }}>
                <span
                  style={{
                    width: `${Math.round(Math.max(0, Math.min(1, hit.score)) * 100)}%`,
                    background: strong ? "var(--ink)" : "#a5a5ae",
                  }}
                />
              </div>
              <span
                style={{
                  flex: 1,
                  minWidth: 0,
                  fontSize: 13.5,
                  color: "var(--ink-soft)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {hit.text}
              </span>
              <span className="mono" style={{ flex: "none", fontSize: 12, color: "var(--muted)" }}>
                {pageLabel(hit)}
              </span>
              <Chevron />
            </summary>
            <div style={{ padding: "0 2px 20px 54px" }}>
              <p style={{ margin: "0 0 10px", fontSize: 14, lineHeight: 1.68, color: "var(--ink-soft)" }}>
                {hit.text}
              </p>
              <span
                className="mono"
                style={{
                  fontSize: 11,
                  color: "var(--muted)",
                  background: "var(--surface)",
                  borderRadius: 5,
                  padding: "4px 8px",
                }}
              >
                {hit.source}
              </span>
            </div>
          </details>
        );
      })}
    </>
  );
}
