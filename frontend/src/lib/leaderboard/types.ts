export interface LeaderboardRow {
  project_id: string;
  paper_id: string;
  paper_title: string | null;
  /** Human-readable per-run title (e.g. "SDAR full · 2026-05-31 19:05"). */
  title?: string | null;
  mode: "rlm" | "rdr";
  models: {
    planner: string | null;
    executor: string | null;
    verifier: string | null;
    grader: string | null;
  };
  overall_score: number | null;
  /** β3: floor-anchored score; equals overall_score on max-mode runs. */
  compute_adjusted_score: number | null;
  /** β4: "efficient" | "max" | null (legacy). */
  execution_mode: string | null;
  meets_target: boolean;
  degraded: boolean;
  cost_usd: number | null;
  iterations: number;
  wall_clock_s: number | null;
  sandbox: string | null;
  started_at: string | null;
  completed_at: string | null;
  verdict: string;
  status: string;
  attempts: number;
}
