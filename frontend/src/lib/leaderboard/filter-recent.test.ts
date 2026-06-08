import { describe, it, expect } from "vitest";
import { filterRecentRows } from "./filter-recent";
import type { LeaderboardRow } from "./types";

function makeRow(overrides: Partial<LeaderboardRow>): LeaderboardRow {
  return {
    project_id: "prj_x",
    paper_id: "p1",
    paper_title: "Test Paper",
    mode: "rlm",
    models: { planner: null, executor: null, verifier: null, grader: null },
    overall_score: null,
    compute_adjusted_score: null,
    execution_mode: null,
    meets_target: false,
    degraded: false,
    cost_usd: null,
    iterations: 0,
    wall_clock_s: null,
    sandbox: null,
    started_at: null,
    completed_at: null,
    verdict: "",
    status: "completed",
    attempts: 1,
    ...overrides,
  };
}

describe("filterRecentRows", () => {
  it("excludes rows where status is interrupted", () => {
    const rows = [
      makeRow({ project_id: "a", status: "completed" }),
      makeRow({ project_id: "b", status: "interrupted" }),
      makeRow({ project_id: "c", status: "failed" }),
    ];
    const result = filterRecentRows(rows);
    expect(result.map((r) => r.project_id)).toEqual(["a", "c"]);
    expect(result.some((r) => r.status === "interrupted")).toBe(false);
  });

  it("caps the result to 8 rows by default", () => {
    const rows = Array.from({ length: 15 }, (_, i) =>
      makeRow({ project_id: `prj_${i}`, status: "completed" })
    );
    expect(filterRecentRows(rows)).toHaveLength(8);
  });

  it("caps after filtering: 10 mixed rows with 4 interrupted yields at most 8 and none interrupted", () => {
    const rows = Array.from({ length: 10 }, (_, i) =>
      makeRow({
        project_id: `p${i}`,
        status: i % 3 === 0 ? "interrupted" : "completed",
      })
    );
    const result = filterRecentRows(rows);
    expect(result.length).toBeLessThanOrEqual(8);
    expect(result.every((r) => r.status !== "interrupted")).toBe(true);
  });

  it("passes through completed, failed, and stopped rows", () => {
    const rows = [
      makeRow({ project_id: "a", status: "completed" }),
      makeRow({ project_id: "b", status: "failed" }),
      makeRow({ project_id: "c", status: "stopped" }),
    ];
    const result = filterRecentRows(rows);
    expect(result).toHaveLength(3);
  });

  it("returns an empty array when all rows are interrupted", () => {
    const rows = [
      makeRow({ project_id: "a", status: "interrupted" }),
      makeRow({ project_id: "b", status: "interrupted" }),
    ];
    expect(filterRecentRows(rows)).toHaveLength(0);
  });

  it("respects a custom cap argument", () => {
    const rows = Array.from({ length: 5 }, (_, i) =>
      makeRow({ project_id: `p${i}`, status: "completed" })
    );
    expect(filterRecentRows(rows, 3)).toHaveLength(3);
  });
});
