import { render, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { LeaderboardTable, type LeaderboardRow } from "./leaderboard-table";

function row(overrides: Partial<LeaderboardRow> = {}): LeaderboardRow {
  return {
    project_id: "prj_x",
    paper_id: "p1",
    paper_title: "Paper One",
    mode: "rlm",
    models: {
      planner: "gpt-5",
      executor: "claude-sonnet-4-6",
      verifier: null,
      grader: null,
    },
    overall_score: 0.5,
    compute_adjusted_score: 0.5,
    execution_mode: "max",
    meets_target: false,
    degraded: false,
    cost_usd: 1.23,
    iterations: 8,
    wall_clock_s: 300,
    sandbox: "docker",
    started_at: "2026-05-23T04:10:00+00:00",
    completed_at: "2026-05-23T04:15:00+00:00",
    verdict: "partial",
    ...overrides,
  };
}

describe("LeaderboardTable", () => {
  it("renders the empty state when rows is empty", () => {
    const { getByText } = render(<LeaderboardTable rows={[]} />);
    expect(getByText(/no completed runs yet/i)).toBeTruthy();
  });

  it("renders one row per leaderboard entry", () => {
    const { container } = render(
      <LeaderboardTable rows={[
        row({ project_id: "a", overall_score: 0.7 }),
        row({ project_id: "b", overall_score: 0.4 }),
      ]} />,
    );
    expect(container.querySelectorAll("tbody tr").length).toBe(2);
  });

  it("sorts by overall_score desc by default", () => {
    const { container } = render(
      <LeaderboardTable rows={[
        row({ project_id: "a", overall_score: 0.4 }),
        row({ project_id: "b", overall_score: 0.7 }),
      ]} />,
    );
    const firstRowId = container.querySelectorAll("tbody tr")[0].getAttribute("data-project-id");
    expect(firstRowId).toBe("b");
  });

  it("re-sorts when a sortable column header is clicked", () => {
    const { container, getByText } = render(
      <LeaderboardTable rows={[
        row({ project_id: "a", cost_usd: 3.0 }),
        row({ project_id: "b", cost_usd: 1.0 }),
      ]} />,
    );
    fireEvent.click(getByText(/^Cost$/));
    const firstRowId = container.querySelectorAll("tbody tr")[0].getAttribute("data-project-id");
    expect(firstRowId).toBe("b");  // cost asc
  });

  it("renders an em dash for null cost / wall_clock / model", () => {
    const { container } = render(
      <LeaderboardTable rows={[row({
        cost_usd: null,
        wall_clock_s: null,
        models: { planner: null, executor: null, verifier: null, grader: null },
      })]} />,
    );
    const dashes = container.querySelectorAll('[data-dash="true"]');
    expect(dashes.length).toBeGreaterThanOrEqual(3);
  });

  it("treats empty-string model identifiers as missing (em-dash, not blank cell)", () => {
    // Defensive: _build_llm_client can produce models.planner = "" for a
    // misconfigured custom-endpoint root (see code-reviewer finding 2026-05-23).
    // ?? leaks "" through; || correctly falls back to Dash. Pin it.
    const { container } = render(
      <LeaderboardTable rows={[row({
        models: { planner: "", executor: "", verifier: null, grader: null },
      })]} />,
    );
    const dashes = container.querySelectorAll('[data-dash="true"]');
    // Two of the dashes are planner+executor; the row also has no cost_usd
    // or wall_clock_s being null here, so >= 2 is the minimum we need.
    expect(dashes.length).toBeGreaterThanOrEqual(2);
  });
});
