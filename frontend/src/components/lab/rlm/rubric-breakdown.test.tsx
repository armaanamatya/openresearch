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
    });
    const { container } = render(<RubricBreakdown projectId="prj_empty" isActive={false} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the panel when clusters are present", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [CLUSTER_OK, CLUSTER_FAILED],
      leafScores: [],
      repairPasses: [],
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
    });
    render(<RubricBreakdown projectId="prj_no_repair" isActive={false} />);
    expect(screen.queryByText(/Repair pass/)).not.toBeInTheDocument();
  });

  it("passes isActive to useRdrArtifacts", () => {
    mockUseRdrArtifacts.mockReturnValue({
      clusters: [CLUSTER_OK],
      leafScores: [],
      repairPasses: [],
    });
    render(<RubricBreakdown projectId="prj_active" isActive={true} />);
    expect(mockUseRdrArtifacts).toHaveBeenCalledWith("prj_active", true);
  });
});
