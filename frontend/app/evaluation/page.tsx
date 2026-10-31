import Chevron from "@/components/Chevron";
import Corpus from "@/components/Corpus";
import { compare, loadRuns, pickBaseline, type EvalRun } from "@/lib/runs";

// Runs are files that change outside the build, so never cache this page.
export const dynamic = "force-dynamic";

function metric(run: EvalRun, name: string): number | null {
  const value = run.metrics[name];
  return typeof value === "number" ? value : null;
}

function hitRate(run: EvalRun): number | null {
  const key = Object.keys(run.metrics).find((k) => k.startsWith("hit_rate@"));
  return key ? run.metrics[key] : null;
}

function show(value: number | null, digits = 3): string {
  return value === null ? "—" : value.toFixed(digits);
}

export default async function EvaluationPage() {
  const runs = await loadRuns();

  if (runs.length === 0) {
    return (
      <div className="layout">
      <Corpus />
      <div className="work">
        <h1 className="heading">
          How good is retrieval?
        </h1>
        <p style={{ margin: 0, fontSize: 14, color: "var(--muted)", lineHeight: 1.6, maxWidth: "60ch" }}>
          No evaluation runs found in <span className="mono">eval/runs/</span>. Run{" "}
          <span className="mono">rag eval</span> from the repo root and reload.
        </p>
        </div>
      </div>
    );
  }

  const latest = runs[runs.length - 1];
  // Not runs[0]: the oldest file may have been scored on a different dataset.
  const first = pickBaseline(runs, latest);
  const delta = first ? compare(first, latest) : null;
  const failures = latest.results.filter((r) => r.hit === 0);
  const answerable = latest.results.length;

  return (
    <div className="layout">
      <Corpus />
      <div className="work">
      <h1 className="heading">
        How good is retrieval?
      </h1>
      <p style={{ margin: "0 0 36px", fontSize: 14, color: "var(--muted)", lineHeight: 1.6, maxWidth: "60ch" }}>
        {answerable} hand-written questions, each pointing at the page that answers it. Scored
        without a language model, so a run is free and gives the same number every time.
      </p>

      {/* headline: first run vs latest */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 24,
          padding: "24px 0",
          borderTop: "1px solid var(--border)",
          borderBottom: "1px solid var(--border)",
          marginBottom: 36,
        }}
      >
        {first && (
          <>
            <div>
              <div className="mono" style={{ fontSize: 15, color: "var(--faint)", textDecoration: "line-through" }}>
                {show(hitRate(first))}
              </div>
              <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 3 }}>
                {first.started_at.slice(0, 10)}
              </div>
            </div>
            <svg width="22" height="14" viewBox="0 0 22 14" fill="none" aria-hidden="true" style={{ flex: "none" }}>
              <path
                d="M1 7h18M14.5 2L20 7l-5.5 5"
                stroke="var(--faint)"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </>
        )}

        <div>
          <div
            className="mono"
            style={{ fontSize: 40, fontWeight: 600, letterSpacing: "-0.03em", lineHeight: 1 }}
          >
            {show(hitRate(latest))}
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 5 }}>
            of questions found their page
          </div>
        </div>

        {delta && (delta.fixed.length > 0 || delta.broke.length > 0) && (
          <div style={{ flex: "1 1 180px", minWidth: 0 }}>
            <span className={`chip ${delta.p !== null && delta.p < 0.05 ? "chip-green" : "chip-amber"}`}>
              {delta.fixed.length} fixed, {delta.broke.length} broken
            </span>
            <p style={{ margin: "8px 0 0", fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5 }}>
              {delta.p !== null && delta.p < 0.05
                ? `p = ${delta.p.toFixed(4)} — too consistent to be luck.`
                : `p = ${delta.p?.toFixed(3)} — not enough to tell it from luck.`}
            </p>
          </div>
        )}
      </div>

      {/* every run, newest first */}
      <h2 className="label">Runs</h2>
      <p style={{ margin: "0 0 8px", fontSize: 12, color: "var(--muted)" }}>
        Newest first. Open one to see the settings behind it.
      </p>

      {[...runs].reverse().map((run) => (
        <details key={run.file} className="disclosure">
          <summary>
            <span style={{ flex: 1, minWidth: 0, fontSize: 13.5 }}>
              {run.started_at.slice(0, 16).replace("T", " ")}
            </span>
            {run.config.page_spans === true && (
              <span
                style={{
                  flex: "none",
                  fontSize: 11,
                  color: "var(--muted)",
                  background: "var(--surface)",
                  borderRadius: 5,
                  padding: "3px 7px",
                }}
              >
                page spans
              </span>
            )}
            <span className="mono" style={{ flex: "none", fontSize: 14, fontWeight: 600, width: 48, textAlign: "right" }}>
              {show(hitRate(run))}
            </span>
            <Chevron />
          </summary>

          <div style={{ padding: "0 2px 20px 2px" }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 26, marginBottom: 16 }}>
              {[
                ["MRR", metric(run, "mrr")],
                ["NDCG@10", metric(run, "ndcg@10")],
                ["Recall@5", metric(run, "recall@5")],
                ["Chunk prec.", metric(run, "chunk_precision@5")],
              ].map(([label, value]) => (
                <div key={label as string}>
                  <div
                    style={{
                      fontSize: 10.5,
                      color: "var(--faint)",
                      letterSpacing: "0.04em",
                      textTransform: "uppercase",
                      marginBottom: 4,
                    }}
                  >
                    {label}
                  </div>
                  <div className="mono" style={{ fontSize: 14 }}>
                    {show(value as number | null)}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {Object.entries(run.config).map(([key, value]) => (
                <span
                  key={key}
                  className="mono"
                  style={{
                    fontSize: 10.5,
                    color: "var(--muted)",
                    background: "var(--surface)",
                    borderRadius: 5,
                    padding: "3px 7px",
                  }}
                >
                  {key}={String(value)}
                </span>
              ))}
            </div>
          </div>
        </details>
      ))}

      {/* failures in the latest run */}
      <h2 className="label" style={{ marginTop: 36 }}>Still failing</h2>
      <p style={{ margin: "0 0 8px", fontSize: 12, color: "var(--muted)" }}>
        {failures.length} of {answerable} in the most recent run.
      </p>

      {failures.map((failure) => (
        <div
          key={failure.id}
          style={{ display: "flex", gap: 14, padding: "15px 2px", borderBottom: "1px solid var(--border)" }}
        >
          <span
            className="mono"
            style={{
              flex: "none",
              fontSize: 11,
              fontWeight: 500,
              color: "var(--amber)",
              background: "var(--amber-bg)",
              borderRadius: 5,
              padding: "4px 8px",
              height: "fit-content",
            }}
          >
            {failure.id}
          </span>
          <div style={{ minWidth: 0 }}>
            <p style={{ margin: "0 0 5px", fontSize: 13.5, lineHeight: 1.5 }}>{failure.question}</p>
            <p className="mono" style={{ margin: 0, fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5 }}>
              wanted {failure.expected.join(", ") || "—"}
            </p>
          </div>
        </div>
      ))}

      <p style={{ margin: "28px 0 0", fontSize: 12.5, color: "var(--muted)", lineHeight: 1.65, maxWidth: "62ch" }}>
        With {answerable} questions, one question moves the score by{" "}
        {(1 / answerable).toFixed(3)} — so a change has to fix about six before it can be told
        apart from luck.
      </p>
      </div>
    </div>
  );
}
