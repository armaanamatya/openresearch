/**
 * Human-facing paper-title + run-identity derivation for the lab run list.
 *
 * The backend `/runs` endpoint returns the RAW `demo_status.json` dict per run
 * directory. Multiple directories can share the same `projectId` (the canonical
 * `prj_…` dir plus preserved `prj_…__<timestamp>` snapshots), so `projectId` is
 * NOT a unique React key — `runDirName` derives a stable per-directory id from
 * `outputDir`. `paperDisplayTitle` picks the best human label, defending against
 * placeholder titles ("paper_text", "Untitled", …) that CLI-created runs emit.
 */

/** Inputs shared by the title resolver — a loose shape so any run-summary fits. */
export interface PaperTitleInput {
  paperTitle?: string;
  paper?: { id?: string; title?: string } | null;
  paperId?: string;
  sourceLabel?: string;
  projectId: string;
}

/** Inputs for deriving the unique per-run directory id. */
export interface RunDirInput {
  runDir?: string;
  outputDir?: string;
  projectId: string;
  startedAt?: string;
  updatedAt?: string;
}

/** Inputs for run-numbering — keyed by paper, ordered by recency. */
export interface RunNumberInput {
  paperId?: string;
  paper?: { id?: string } | null;
  projectId: string;
  updatedAt?: string;
}

// Case-insensitive, trimmed titles that carry no information — treat as absent.
const PLACEHOLDER_TITLES = new Set([
  "",
  "paper_text",
  "untitled",
  "untitled paper",
  "paper text",
]);

const ARXIV_ID = /^\d{4,5}\.\d{4,5}$/;

function isMeaningfulTitle(value: string | undefined): value is string {
  if (typeof value !== "string") return false;
  return !PLACEHOLDER_TITLES.has(value.trim().toLowerCase());
}

/**
 * Best human-readable title for a run, in priority order:
 *   1. `paperTitle` / `paper.title` when present and not a placeholder.
 *   2. a paper id (`paperId` / `paper.id`): arXiv-shaped → `arXiv:<id>`, else the id.
 *   3. `sourceLabel` if present, else the raw `projectId`.
 */
export function paperDisplayTitle(run: PaperTitleInput): string {
  const rawTitle = run.paperTitle ?? run.paper?.title;
  if (isMeaningfulTitle(rawTitle)) return rawTitle.trim();

  const paperId = run.paperId ?? run.paper?.id;
  if (typeof paperId === "string" && paperId.trim().length > 0) {
    const id = paperId.trim();
    return ARXIV_ID.test(id) ? `arXiv:${id}` : id;
  }

  if (typeof run.sourceLabel === "string" && run.sourceLabel.trim().length > 0) {
    return run.sourceLabel;
  }
  return run.projectId;
}

/**
 * The UNIQUE + STABLE per-run id, used as the React list key so runs sharing a
 * `projectId` don't collide:
 *   1. `runDir` — the backend-supplied filesystem dir name (truly unique, e.g.
 *      "prj_abc__20260531-231049").
 *   2. else `projectId::startedAt` — `startedAt` is set once at run start, so it
 *      is both unique-per-run and STABLE across re-renders. (`outputDir` is NOT
 *      used: it is the stale canonical path, identical across preserved
 *      snapshots; `updatedAt` is unique but changes mid-run, which would remount
 *      the row on every status tick.)
 */
export function runDirName(run: RunDirInput): string {
  if (typeof run.runDir === "string" && run.runDir.trim().length > 0) {
    return run.runDir.trim();
  }
  const stamp = run.startedAt ?? run.updatedAt ?? "";
  return `${run.projectId}::${stamp}`;
}

function paperKey(run: RunNumberInput): string {
  return run.paperId ?? run.paper?.id ?? run.projectId;
}

/**
 * Assign run numbers within each paper group. Runs sharing a paper key
 * (`paperId ?? paper.id ?? projectId`) are numbered 1..N by `updatedAt`
 * ascending (oldest = run 1). Papers with a single run map to `null` so the
 * caller can suppress a meaningless "run 1" label. Keyed by object identity.
 */
export function numberRunsByPaper<
  T extends { paperId?: string; paper?: { id?: string } | null; projectId: string; updatedAt?: string },
>(runs: T[]): Map<T, number | null> {
  const groups = new Map<string, T[]>();
  for (const run of runs) {
    const key = paperKey(run);
    const bucket = groups.get(key);
    if (bucket) {
      bucket.push(run);
    } else {
      groups.set(key, [run]);
    }
  }

  const result = new Map<T, number | null>();
  for (const bucket of groups.values()) {
    if (bucket.length <= 1) {
      result.set(bucket[0], null);
      continue;
    }
    const ordered = [...bucket].sort(
      (a, b) => (a.updatedAt ?? "").localeCompare(b.updatedAt ?? ""),
    );
    ordered.forEach((run, index) => {
      result.set(run, index + 1);
    });
  }
  return result;
}
