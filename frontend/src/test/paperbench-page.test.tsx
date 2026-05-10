import { render, screen } from "@testing-library/react";

import { PaperBenchClient } from "../components/paperbench/paperbench-client";
import type {
  PaperBenchBundleListing,
  PaperBenchRunStatus,
} from "../lib/paperbench/runner";

const SAMPLE_BUNDLES: PaperBenchBundleListing = {
  bundles_root: "/tmp/third_party/paperbench",
  bundles: [
    {
      paper_id: "ftrl",
      metadata: {
        title: "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem",
        venue: "ICML 2024 (Spotlight)",
      },
      has_addendum: true,
      rubric_path: "/tmp/third_party/paperbench/ftrl/rubric.json",
    },
  ],
};

const SAMPLE_RUN: PaperBenchRunStatus = {
  run_group_id: "pb_ftrl_test",
  paper_id: "ftrl",
  bundle_root: "/tmp/third_party/paperbench/ftrl",
  runs_root: "/tmp/runs",
  mode: "dry",
  seeds: [0],
  max_parallel: 1,
  provider: null,
  model: null,
  status: "succeeded",
  started_at: "2026-05-09T00:00:00+00:00",
  updated_at: "2026-05-09T00:00:01+00:00",
  completed_at: "2026-05-09T00:00:01+00:00",
  attempts: [
    {
      attempt_id: "pb_ftrl_test-dry",
      seed: null,
      status: "succeeded",
      submission_dir: "/tmp/runs/paperbench/pb_ftrl_test/submission",
      submission_validation: {
        ok: true,
        errors: [],
        warnings: [],
        total_bytes: 296,
        file_count: 2,
        committed_bytes: null,
      },
      score: null,
    },
  ],
  rubric_summary: {
    node_count: 12,
    leaf_count: 8,
    max_depth: 2,
    task_category_weights: {
      "Code Development": { weight: 0.6, percent: 60, leaf_count: 4 },
      Execution: { weight: 0.2, percent: 20, leaf_count: 2 },
      "Result Match": { weight: 0.2, percent: 20, leaf_count: 2 },
    },
    finegrained_category_weights: {
      Methodology: { weight: 0.8, percent: 80, leaf_count: 6 },
      Results: { weight: 0.2, percent: 20, leaf_count: 2 },
    },
  },
  code_development_ceiling: 0.6,
  published_baselines: {
    claude_3_5_sonnet_basicagent: { mean: 0.093, se: 0.01 },
    o1_basicagent: { mean: 0.017, se: 0.008 },
  },
  blacklist_entries: ["github.com/example"],
  mean_score: null,
  standard_error: null,
  n_attempts: 1,
  error: null,
};

describe("paperbench page", () => {
  it("renders the bundle picker and headline", () => {
    render(<PaperBenchClient initialBundles={SAMPLE_BUNDLES} initialRuns={[]} />);
    expect(screen.getByText(/PaperBench head-to-head/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        /Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem/i
      )
    ).toBeInTheDocument();
  });

  it("renders an empty-state message when no runs exist", () => {
    render(<PaperBenchClient initialBundles={SAMPLE_BUNDLES} initialRuns={[]} />);
    expect(screen.getByText(/No runs yet/i)).toBeInTheDocument();
  });

  it("renders the rubric breakdown and baselines for an existing run", () => {
    render(
      <PaperBenchClient initialBundles={SAMPLE_BUNDLES} initialRuns={[SAMPLE_RUN]} />
    );
    expect(screen.getByText("Code Development")).toBeInTheDocument();
    expect(screen.getByText("Execution")).toBeInTheDocument();
    expect(screen.getByText("Result Match")).toBeInTheDocument();
    expect(screen.getByText("claude_3_5_sonnet_basicagent")).toBeInTheDocument();
    expect(screen.getByText("o1_basicagent")).toBeInTheDocument();
    expect(screen.getAllByText("succeeded").length).toBeGreaterThan(0);
  });

  it("enables provider/model inputs by default (pipeline mode is the default)", () => {
    render(<PaperBenchClient initialBundles={SAMPLE_BUNDLES} initialRuns={[]} />);
    const providerSelect = screen.getByText("Provider").parentElement?.querySelector("select");
    expect(providerSelect).not.toBeDisabled();
  });
});
