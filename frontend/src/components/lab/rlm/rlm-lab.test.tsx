import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { RlmLab } from "./rlm-lab";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

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
