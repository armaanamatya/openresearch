import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RlmLab } from "./rlm-lab";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

describe("RlmLab", () => {
  it("renders all regions from an event stream", () => {
    render(<RlmLab events={rlmRunFixture} runMeta={{ projectId: "prj_x",
      paperTitle: "Attention is all you need", paperMeta: "Vaswani et al." }} />);
    expect(screen.getByText("Attention is all you need")).toBeInTheDocument();
    expect(screen.getAllByText(/0\.53/).length).toBeGreaterThan(0);
    expect(screen.getByText(/primitive call history/i)).toBeInTheDocument();
    expect(screen.getAllByRole("button").length).toBeGreaterThan(5);
  });
});
