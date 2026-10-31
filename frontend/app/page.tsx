"use client";

import { useCallback, useState } from "react";

import AnswerText from "@/components/AnswerText";
import Chevron from "@/components/Chevron";
import Corpus from "@/components/Corpus";
import PassageRows from "@/components/PassageRows";
import { ApiError, ask, search } from "@/lib/api";
import type { QueryResponse, SearchHit } from "@/lib/types";

type State =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "answered"; result: QueryResponse; hits: SearchHit[] }
  | { kind: "failed"; error: ApiError };

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  const submit = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;

      setQuestion(trimmed);
      setState({ kind: "working" });

      try {
        // Both at once: /query returns only the passages the model cited, so showing
        // what it rejected — the part worth seeing — needs /search as well.
        const [result, searched] = await Promise.all([
          ask({ question: trimmed }),
          search(trimmed).catch(() => ({ query: trimmed, hits: [] })),
        ]);
        setState({ kind: "answered", result, hits: searched.hits });
      } catch (error) {
        setState({
          kind: "failed",
          error: error instanceof ApiError ? error : new ApiError(0, "Something went wrong."),
        });
      }
    },
    [],
  );

  return (
    <div className="layout">
      <Corpus />

      <div className="work">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (state.kind !== "working") void submit(question);
          }}
        >
          <div className="ask-field">
            <label htmlFor="question" className="sr-only">
              Your question
            </label>
            <input
              id="question"
              type="text"
              value={question}
              maxLength={2000}
              placeholder="What do the documents say about…"
              onChange={(event) => setQuestion(event.target.value)}
              autoComplete="off"
            />
            <button
              type="submit"
              className={`icon-button ${state.kind === "working" ? "is-working" : ""}`}
              disabled={state.kind === "working" || question.trim() === ""}
              aria-label={state.kind === "working" ? "Asking" : "Ask"}
            >
              <svg width="16" height="16" viewBox="0 0 18 18" fill="none" aria-hidden="true">
                <path
                  d="M4 9h10M9.5 4.5L14 9l-4.5 4.5"
                  stroke={question.trim() === "" ? "var(--faint)" : "var(--bg)"}
                  strokeWidth="1.9"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </form>

        <p className="lede">
          Answers are drawn only from the indexed documents, and every claim names the page
          it came from. When nothing supports an answer, this says so rather than guessing.
        </p>

        <div aria-live="polite" aria-busy={state.kind === "working"}>
          {state.kind === "working" && <Working />}
          {state.kind === "failed" && <Failed error={state.error} />}
          {state.kind === "answered" && <Answered result={state.result} hits={state.hits} />}
        </div>
      </div>
    </div>
  );
}

function Working() {
  return (
    <div aria-hidden="true" style={{ display: "flex", flexDirection: "column", gap: 13 }}>
      {["100%", "94%", "58%"].map((width, i) => (
        <div
          key={i}
          className="is-working"
          style={{ height: 19, width, background: "var(--surface-soft)", borderRadius: 5 }}
        />
      ))}
    </div>
  );
}

function Failed({ error }: { error: ApiError }) {
  // 503 is worth its own treatment: the service is up, it just has no language model,
  // and search still works. "Something went wrong" would hide a usable half.
  if (error.isDegraded) {
    return (
      <section>
        <h1 className="heading">Can&rsquo;t write answers right now</h1>
        <p style={{ margin: "0 0 20px", fontSize: 14, lineHeight: 1.65, color: "var(--ink-soft)", maxWidth: "58ch" }}>
          The language model isn&rsquo;t reachable. Searching the documents still works, so you
          can read the passages yourself.
        </p>
        <a className="button" href="/search">
          Search the documents instead
        </a>
      </section>
    );
  }

  return (
    <section>
      <h1 className="heading">That didn&rsquo;t work</h1>
      <p style={{ margin: 0, fontSize: 14, lineHeight: 1.65, color: "var(--ink-soft)", maxWidth: "58ch" }}>
        {error.detail || error.message}
      </p>
    </section>
  );
}

function Answered({ result, hits }: { result: QueryResponse; hits: SearchHit[] }) {
  // A refusal is a correct outcome on a grounded system, so it carries the same weight
  // as an answer rather than an error treatment.
  if (!result.answered) {
    return (
      <section>
        <p className="answer">That isn&rsquo;t in these documents.</p>
        <p style={{ margin: "0 0 22px", fontSize: 14, lineHeight: 1.7, color: "var(--ink-soft)", maxWidth: "60ch" }}>
          Nothing in the indexed corpus supports an answer, so this won&rsquo;t guess.
        </p>
        {hits.length > 0 && (
          <details>
            <summary className="quiet-summary">
              <Chevron size={12} />
              What it looked at anyway
            </summary>
            <div style={{ paddingTop: 10 }}>
              <PassageRows hits={hits} />
            </div>
          </details>
        )}
      </section>
    );
  }

  return (
    <section>
      <AnswerText text={result.answer} citations={result.citations} />

      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", marginBottom: 32 }}>
        <span className="chip chip-amber">
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path
              d="M7 1.4l1.6 3.4 3.7.5-2.7 2.6.7 3.7L7 9.9l-3.3 1.7.7-3.7L1.7 5.3l3.7-.5L7 1.4z"
              stroke="var(--amber)"
              strokeWidth="1.3"
              strokeLinejoin="round"
            />
          </svg>
          {result.citations.length} {result.citations.length === 1 ? "source" : "sources"}
        </span>
        <span className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
          {result.latency_ms} ms{result.cache_hit ? " · cached" : ""}
        </span>
      </div>

      {result.citations.length > 0 && (
        <>
          <p className="label">Grounded in</p>
          {result.citations.map((citation, index) => (
            <details key={citation.chunk_id} className="disclosure">
              <summary>
                <span
                  className="mono"
                  style={{
                    flex: "none",
                    width: 20,
                    fontSize: 11,
                    fontWeight: 600,
                    color: "var(--faint)",
                  }}
                >
                  {index + 1}
                </span>
                <span style={{ flex: 1, minWidth: 0, fontSize: 13.5, fontWeight: 500 }}>
                  {citation.source_path}
                </span>
                {citation.page_number !== null && (
                  <span className="mono" style={{ flex: "none", fontSize: 11.5, color: "var(--muted)" }}>
                    p. {citation.page_number}
                  </span>
                )}
                <Chevron />
              </summary>
              <p
                className="serif"
                style={{
                  margin: 0,
                  padding: "0 2px 18px 32px",
                  fontSize: 15,
                  lineHeight: 1.66,
                  color: "var(--ink-soft)",
                  maxWidth: "62ch",
                }}
              >
                {citation.quote}
              </p>
            </details>
          ))}
        </>
      )}

      {hits.length > 0 && (
        <details style={{ marginTop: 22 }}>
          <summary className="quiet-summary">
            <Chevron size={12} />
            All {hits.length} passages the search considered
          </summary>
          <div style={{ paddingTop: 10 }}>
            <PassageRows hits={hits} />
            <a href="/search" style={{ display: "inline-block", marginTop: 12, fontSize: 12.5 }}>
              Open the search inspector →
            </a>
          </div>
        </details>
      )}
    </section>
  );
}
