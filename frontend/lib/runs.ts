/**
 * Reads evaluation runs from `eval/runs/` on disk.
 *
 * Server-side only. The API has no evaluation endpoint yet, and adding one just to
 * render a table would be the wrong order — the runs are already files, versioned with
 * the repo, and the page that shows them runs on the same machine. If the frontend is
 * ever deployed away from the repo, that is the moment to add `GET /v1/evaluation`
 * rather than now.
 */

import { promises as fs } from "node:fs";
import path from "node:path";

/** One scored question, as `EvalRunner.score_question` writes it. */
export interface QuestionResult {
  id: string;
  question: string;
  expected: string[];
  retrieved: string[];
  hit: number;
  rr: number;
  tags: string[];
  note: string;
  chunk_pages?: string[][];
}

export interface EvalRun {
  started_at: string;
  config: Record<string, string | number | boolean>;
  metrics: Record<string, number>;
  results: QuestionResult[];
  dataset: string;
  git_sha: string;
  /** Added here, not in the file: the name it was saved under. */
  file: string;
}

const RUNS_DIR = path.join(process.cwd(), "..", "eval", "runs");

export async function loadRuns(): Promise<EvalRun[]> {
  let names: string[];
  try {
    names = (await fs.readdir(RUNS_DIR)).filter((n) => n.endsWith(".json"));
  } catch {
    return [];
  }

  const runs = await Promise.all(
    names.map(async (name) => {
      try {
        const raw = await fs.readFile(path.join(RUNS_DIR, name), "utf-8");
        const parsed = JSON.parse(raw) as Omit<EvalRun, "file">;
        if (!parsed?.metrics || !Array.isArray(parsed.results)) return null;
        return { ...parsed, file: name } satisfies EvalRun;
      } catch {
        // A half-written or hand-edited run must not take the page down with it.
        return null;
      }
    }),
  );

  return runs
    .filter((r): r is EvalRun => r !== null)
    .sort((a, b) => a.started_at.localeCompare(b.started_at));
}

/**
 * The earliest run that is comparable to `latest` — same number of scored questions.
 *
 * Not simply the oldest file. `eval/runs/` holds every run this project has ever
 * produced, including early ones against a three-question dataset. Comparing 32
 * questions against 3 finds two in common and reports a confident, meaningless
 * p-value: exactly the failure this project exists to avoid. A run scored on a
 * different dataset is not a baseline, it is a different experiment.
 */
export function pickBaseline(runs: EvalRun[], latest: EvalRun): EvalRun | null {
  const size = latest.results.length;
  const comparable = runs.filter(
    (run) => run !== latest && run.results.length === size,
  );
  return comparable[0] ?? null;
}

/**
 * Paired comparison between two runs, over the questions they share.
 *
 * Only questions where the runs disagree carry information, which is why the fixed and
 * broken counts are reported rather than the difference between two averages. On a
 * 32-question set, a 0.03 move in hit rate is one question.
 */
export function compare(before: EvalRun, after: EvalRun) {
  const left = new Map(before.results.map((r) => [r.id, r]));
  const fixed: string[] = [];
  const broke: string[] = [];

  for (const result of after.results) {
    const previous = left.get(result.id);
    if (!previous) continue;
    if (previous.hit === 0 && result.hit === 1) fixed.push(result.id);
    if (previous.hit === 1 && result.hit === 0) broke.push(result.id);
  }

  return { fixed, broke, p: exactBinomialP(fixed.length, broke.length) };
}

/**
 * McNemar's exact test: the probability of a split this lopsided if the change did
 * nothing. Two-sided, computed directly rather than pulled in as a dependency.
 */
function exactBinomialP(fixed: number, broke: number): number | null {
  const n = fixed + broke;
  if (n === 0) return null;

  const smaller = Math.min(fixed, broke);
  let tail = 0;
  for (let k = 0; k <= smaller; k++) tail += choose(n, k);
  return Math.min(1, (2 * tail) / 2 ** n);
}

function choose(n: number, k: number): number {
  let result = 1;
  for (let i = 1; i <= k; i++) result = (result * (n - i + 1)) / i;
  return result;
}
