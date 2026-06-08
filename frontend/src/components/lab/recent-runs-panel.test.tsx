import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RecentRunsPanel } from "./recent-runs-panel";
import type { LeaderboardRow } from "@/lib/leaderboard/types";

// next/link renders as a plain <a> in jsdom
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode; [k: string]: unknown }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

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
    status: "completed",
    attempts: 1,
    ...overrides,
  };
}

describe("RecentRunsPanel", () => {
  it("renders the empty state when rows is empty", () => {
    render(<RecentRunsPanel rows={[]} />);
    expect(screen.getByText(/no runs yet/i)).toBeInTheDocument();
  });

  it("renders the error state when error is provided", () => {
    render(<RecentRunsPanel rows={[]} error="Leaderboard unavailable." />);
    expect(screen.getByText(/leaderboard unavailable/i)).toBeInTheDocument();
  });

  it("renders 'View all' link pointing to /leaderboard", () => {
    render(<RecentRunsPanel rows={[row()]} />);
    const viewAll = screen.getByRole("link", { name: /view all/i });
    expect(viewAll).toBeInTheDocument();
    expect(viewAll.getAttribute("href")).toBe("/leaderboard");
  });

  it("renders paper title as a link to /lab?projectId=<id>", () => {
    render(<RecentRunsPanel rows={[row({ project_id: "prj_abc", paper_title: "Test Paper" })]} />);
    const link = screen.getByRole("link", { name: "Test Paper" });
    expect(link.getAttribute("href")).toBe("/lab?projectId=prj_abc");
  });

  it("falls back to paper_id when paper_title is null", () => {
    render(<RecentRunsPanel rows={[row({ paper_title: null, paper_id: "2605.15155" })]} />);
    expect(screen.getByText("2605.15155")).toBeInTheDocument();
  });

  it("renders a Replay link pointing to /lab?replay=<id>", () => {
    render(<RecentRunsPanel rows={[row({ project_id: "prj_replay" })]} />);
    const replayLink = screen.getByRole("link", { name: /replay/i });
    expect(replayLink.getAttribute("href")).toBe("/lab?replay=prj_replay");
  });

  it("shows best score (compute_adjusted_score) formatted to 2 d.p.", () => {
    render(<RecentRunsPanel rows={[row({ compute_adjusted_score: 0.488 })]} />);
    expect(screen.getByText(/0\.49/)).toBeInTheDocument();
  });

  it("renders em-dash when compute_adjusted_score is null", () => {
    render(<RecentRunsPanel rows={[row({ compute_adjusted_score: null })]} />);
    const dashes = document.querySelectorAll('[data-dash="true"]');
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it("adds ×N superscript hint when attempts > 1", () => {
    render(<RecentRunsPanel rows={[row({ attempts: 7, compute_adjusted_score: 0.488 })]} />);
    expect(screen.getByText("×7")).toBeInTheDocument();
  });

  it("does not render ×N hint when attempts === 1", () => {
    render(<RecentRunsPanel rows={[row({ attempts: 1 })]} />);
    expect(document.querySelector("sup")).toBeNull();
  });

  it("renders one row per entry", () => {
    render(
      <RecentRunsPanel rows={[
        row({ project_id: "a" }),
        row({ project_id: "b" }),
        row({ project_id: "c" }),
      ]} />
    );
    expect(document.querySelectorAll("li").length).toBe(3);
  });
});
