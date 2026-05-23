import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PrimitiveHistoryBar } from "./primitive-history-bar";
import type { PrimitiveCallView } from "../../../hooks/use-rlm-run";

const calls = [
  { primitive: "understand_section", status: "ok" as const, iteration: 1 },
  { primitive: "run_experiment", status: "error" as const, iteration: 8 },
] as unknown as PrimitiveCallView[];

describe("PrimitiveHistoryBar", () => {
  it("is collapsed by default, showing only a count summary", () => {
    render(<PrimitiveHistoryBar calls={calls} />);
    expect(screen.getByText(/2 calls/)).toBeInTheDocument();
    expect(screen.queryByText("understand_section")).not.toBeInTheDocument();
  });
  it("expands to a reverse-chronological list on click", () => {
    render(<PrimitiveHistoryBar calls={calls} />);
    fireEvent.click(screen.getByRole("button"));
    const rows = screen.getAllByTestId("primitive-call-row");
    expect(rows[0]).toHaveTextContent("run_experiment"); // newest first
  });
});
