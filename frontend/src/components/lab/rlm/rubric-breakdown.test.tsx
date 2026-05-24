import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, afterEach } from "vitest";

import { RubricBreakdown } from "./rubric-breakdown";
import type { DemoClusterStatus, DemoLeafScore, DemoRepairPass } from "@/lib/demo/demo-run-types";

// useRdrArtifacts is a polling hook — stub it out so the component tests
// are synchronous and don't depend on fetch or timers.
vi.mock("@/hooks/use-rdr-artifacts", () => ({
  useRdrArtifacts: vi.fn(),
}));

import { useRdrArtifacts } from "@/hooks/use-rdr-artifacts";

const mockUseRdrArtifacts = useRdrArtifacts as ReturnType<typeof vi.fn>;

const CLUSTER_OK: DemoClusterStatus = {
  index: 0,
  cluster_id: "clus-1",
  title: "Introduction",
  leaf_ids: ["leaf-a"],
  failed: false,
  file_count: 2,
  repair_history: [],
};

const CLUSTER_FAILED: DemoClusterStatus = {
  index: 1,
  cluster_id: "clus-2",
  title: "Methods",
  leaf_ids: ["leaf-b"],
  failed: true,
  file_count: 0,
  repair_history: [{ pass: 1, failed: true, file_count: 0 }],
};

const LEAF: DemoLeafScore = {
  id: "leaf-a",
  score: 0.75,
  justification: "Implementation matches specification.",
};

const REPAIR: DemoRepairPass = {
  pass: 1,
  cluster_count: 2,
  failed_count: 1,
};

describe("RubricBreakdown", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders nothing when there is no data", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [],
      leafScores: [],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    const { container } = render(<RubricBreakdown projectId="prj_empty" isActive={false} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the panel when clusters are present", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [CLUSTER_OK, CLUSTER_FAILED],
      leafScores: [],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(<RubricBreakdown projectId="prj_clusters" isActive={false} />);
    expect(screen.getByTestId("rubric-breakdown")).toBeInTheDocument();
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("Methods")).toBeInTheDocument();
  });

  it("renders leaf scores with score bars", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [],
      leafScores: [LEAF],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(<RubricBreakdown projectId="prj_leaves" isActive={false} />);
    expect(screen.getByTestId("rubric-breakdown")).toBeInTheDocument();
    expect(screen.getByText("leaf-a")).toBeInTheDocument();
    expect(screen.getByText("0.75")).toBeInTheDocument();
    expect(screen.getByText("Implementation matches specification.")).toBeInTheDocument();
  });

  it("shows the repair banner when repair passes exist", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [CLUSTER_OK],
      leafScores: [],
      repairPasses: [REPAIR],
      noRdrArtifacts: false,
    });
    render(<RubricBreakdown projectId="prj_repair" isActive={false} />);
    expect(screen.getByText("Repair pass 1")).toBeInTheDocument();
    expect(screen.getByText(/2 clusters, 1 failed/)).toBeInTheDocument();
  });

  it("does not show repair banner when there are no repair passes", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [CLUSTER_OK],
      leafScores: [],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(<RubricBreakdown projectId="prj_no_repair" isActive={false} />);
    expect(screen.queryByText(/Repair pass/)).not.toBeInTheDocument();
  });

  it("passes isActive to useRdrArtifacts", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [CLUSTER_OK],
      leafScores: [],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(<RubricBreakdown projectId="prj_active" isActive={true} />);
    expect(mockUseRdrArtifacts).toHaveBeenCalledWith("prj_active", true);
  });

  it("test_renders_null_when_no_data: returns null DOM when all arrays are empty", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [],
      leafScores: [],
      repairPasses: [],
      noRdrArtifacts: true,
    });
    const { container } = render(<RubricBreakdown projectId="prj_null" isActive={false} />);
    expect(container.firstChild).toBeNull();
  });

  it("test_renders_when_data_present: renders breakdown panel when cluster and leaf data exist", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [CLUSTER_OK],
      leafScores: [LEAF],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(<RubricBreakdown projectId="prj_has_data" isActive={false} />);
    expect(screen.getByTestId("rubric-breakdown")).toBeInTheDocument();
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("leaf-a")).toBeInTheDocument();
  });

  // ── Per-model mini-bars (Lane γ, 2026-05-23) ──────────────────────────────

  it("renders per-model mini-bars when leaf id has model suffix matching per_model keys", () => {
    const perModelMetrics = {
      qwen3_1_7b: { alfworld: 0.34 },
      qwen2_5_3b: { alfworld: 0.51 },
    };
    // Leaf id "alfworld_qwen3_1_7b" has suffix "qwen3_1_7b" — metric key "alfworld"
    const leafWithSuffix: DemoLeafScore = {
      id: "alfworld_qwen3_1_7b",
      score: 0.34,
      justification: "",
    };
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [],
      leafScores: [leafWithSuffix],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(
      <RubricBreakdown
        projectId="prj_per_model"
        isActive={false}
        perModelMetrics={perModelMetrics}
      />
    );
    const miniBar = screen.getByTestId("per-model-mini-bar");
    expect(miniBar).toBeInTheDocument();
    // Both model values shown in the title attribute
    expect(miniBar.title).toContain("qwen3_1_7b");
    expect(miniBar.title).toContain("qwen2_5_3b");
  });

  it("does NOT render per-model mini-bars when leaf id has no model suffix", () => {
    const perModelMetrics = {
      model_a: { acc: 0.8 },
      model_b: { acc: 0.7 },
    };
    const plainLeaf: DemoLeafScore = {
      id: "overall_accuracy",  // no model suffix matching model_a / model_b
      score: 0.75,
      justification: "",
    };
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [],
      leafScores: [plainLeaf],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(
      <RubricBreakdown
        projectId="prj_no_suffix"
        isActive={false}
        perModelMetrics={perModelMetrics}
      />
    );
    expect(screen.queryByTestId("per-model-mini-bar")).not.toBeInTheDocument();
  });

  it("does NOT render per-model mini-bars when perModelMetrics is null", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [],
      leafScores: [LEAF],
      repairPasses: [],
      noRdrArtifacts: false,
    });
    render(
      <RubricBreakdown
        projectId="prj_null_per_model"
        isActive={false}
        perModelMetrics={null}
      />
    );
    expect(screen.queryByTestId("per-model-mini-bar")).not.toBeInTheDocument();
  });
});
