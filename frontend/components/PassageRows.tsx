import type { SearchHit } from "@/lib/types";

/**
 * Scores shown as a bar as well as a number: the bar makes "much better" visible at a
 * glance, the number keeps it inspectable. Anything below 0.65 greys out, because on
 * this corpus that is roughly where a passage stops being evidence and starts being
 * topically adjacent.
 */
const STRONG = 0.65;

export function pageLabel(hit: Pick<SearchHit, "page_number">): string {
  return hit.page_number === null ? "—" : `p. ${hit.page_number}`;
}

export default function PassageRows({ hits }: { hits: SearchHit[] }) {
  if (hits.length === 0) {
    return (
      <p style={{ margin: "12px 0 0", fontSize: 13, color: "var(--muted)" }}>
        Nothing matched.
      </p>
    );
  }

  return (
    <div>
      {hits.map((hit) => {
        const strong = hit.score >= STRONG;
        return (
          <div className="row" key={hit.chunk_id}>
            <span
              className="mono"
              style={{ flex: "none", width: 54, fontSize: 12, color: "var(--muted)" }}
            >
              {pageLabel(hit)}
            </span>
            <div className="track">
              <span
                style={{
                  width: `${Math.round(Math.max(0, Math.min(1, hit.score)) * 100)}%`,
                  background: strong ? "var(--ink)" : "#a5a5ae",
                }}
              />
            </div>
            <span
              className="mono"
              style={{ flex: "none", fontSize: 12, color: "var(--ink-soft)" }}
            >
              {hit.score.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
