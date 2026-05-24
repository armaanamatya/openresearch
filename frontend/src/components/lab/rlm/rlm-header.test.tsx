import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RlmHeader } from "./rlm-header";

describe("RlmHeader", () => {
  const props = {
    paperTitle: "Attention is all you need",
    paperMeta: "Vaswani et al. · NeurIPS 2017 · arXiv:1706.03762",
    projectId: "prj_8b78ac6368bad043",
    status: "running" as const,
    iterationCount: 13,
    costUsd: 18.4,
  };
  it("renders the paper title, project id, status, and iteration count", () => {
    render(<RlmHeader {...props} />);
    expect(screen.getByText("Attention is all you need")).toBeInTheDocument();
    expect(screen.getByText(/prj_8b78ac6368bad043/)).toBeInTheDocument();
    expect(screen.getByText(/running/i)).toBeInTheDocument();
    expect(screen.getByText(/13/)).toBeInTheDocument();
  });
  it("omits cost when costUsd is null", () => {
    render(<RlmHeader {...props} costUsd={null} />);
    expect(screen.queryByText(/\$/)).not.toBeInTheDocument();
  });
  it("shows runpod visibility before the pod is created", () => {
    render(<RlmHeader {...props} sandboxMode="runpod" primitiveCalls={[]} />);
    expect(screen.getByText("runpod: not yet")).toBeInTheDocument();
  });
});
