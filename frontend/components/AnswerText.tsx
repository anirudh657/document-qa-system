import type { Citation } from "@/lib/types";

/**
 * Renders the answer with its citation markers resolved.
 *
 * The model is asked to cite as `[chunk_id]` and emits an eight-character hex prefix,
 * so raw answers read "…replaces the 4xAA pack [a30dde05]." Printed as-is that is an
 * internal identifier leaking into prose — it means nothing to a reader, and the
 * citation it refers to is already listed underneath with a number.
 *
 * Each marker becomes that number. A marker matching no supplied citation is dropped
 * rather than shown: `validate_citations` on the server already treats an invented
 * citation as a fault, and printing it here would dress a hallucination up as a source.
 */
const MARKER = /\[([0-9a-f]{6,32})\]/g;

export default function AnswerText({
  text,
  citations,
}: {
  text: string;
  citations: Citation[];
}) {
  // Markers carry an 8-char prefix; citations carry the full id.
  const indexByPrefix = new Map(
    citations.map((citation, index) => [citation.chunk_id.slice(0, 8), index + 1]),
  );

  const parts: React.ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(MARKER)) {
    const at = match.index ?? 0;
    if (at > cursor) parts.push(text.slice(cursor, at));

    const number = indexByPrefix.get(match[1].slice(0, 8));
    if (number !== undefined) {
      parts.push(
        <sup key={`${at}-${number}`} className="marker">
          {number}
        </sup>,
      );
    } else {
      // Dropping the marker leaves the space that preceded it, which reads as
      // "supplied ." or doubles up into "Something  here." Tidying the finished
      // string cannot fix that: the space and the punctuation end up in separate
      // nodes, so nothing ever sees them together. Removing it here does.
      const last = parts[parts.length - 1];
      if (typeof last === "string" && last.endsWith(" ")) {
        parts[parts.length - 1] = last.replace(/ +$/, "");
      }
    }
    cursor = at + match[0].length;
  }

  if (cursor < text.length) parts.push(text.slice(cursor));

  return <p className="answer">{parts}</p>;
}
