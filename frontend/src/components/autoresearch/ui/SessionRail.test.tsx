import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { SessionRail } from "./SessionRail";
import styles from "./SessionRail.module.css";

const SESSIONS = [
  { id: "s1", title: "New session - 2026-07-05", active: true },
  { id: "s2", title: "Baseline sweep", active: false },
];

describe("SessionRail", () => {
  it("renders the project header", () => {
    render(<SessionRail projectName="SDAR" sessions={SESSIONS} />);
    expect(screen.getByText("SDAR")).toBeInTheDocument();
  });

  it("renders every session's title", () => {
    render(<SessionRail projectName="SDAR" sessions={SESSIONS} />);
    expect(screen.getByText("New session - 2026-07-05")).toBeInTheDocument();
    expect(screen.getByText("Baseline sweep")).toBeInTheDocument();
  });

  it("marks the active session with data-active and a live dot", () => {
    render(<SessionRail projectName="SDAR" sessions={SESSIONS} />);
    const activeRow = screen.getByText("New session - 2026-07-05").closest("button");
    const inactiveRow = screen.getByText("Baseline sweep").closest("button");
    expect(activeRow).toHaveAttribute("data-active", "true");
    expect(activeRow?.querySelector(`.${styles.liveDot}`)).toBeInTheDocument();
    expect(inactiveRow).toHaveAttribute("data-active", "false");
    expect(inactiveRow?.querySelector(`.${styles.liveDot}`)).not.toBeInTheDocument();
  });

  it('renders a "New agent" control that fires onNewAgent when clicked', () => {
    const onNewAgent = vi.fn();
    render(
      <SessionRail projectName="SDAR" sessions={SESSIONS} onNewAgent={onNewAgent} />
    );
    fireEvent.click(screen.getByRole("button", { name: /new agent/i }));
    expect(onNewAgent).toHaveBeenCalledTimes(1);
  });

  it("fires onSelectSession with the clicked session's id", () => {
    const onSelectSession = vi.fn();
    render(
      <SessionRail
        projectName="SDAR"
        sessions={SESSIONS}
        onSelectSession={onSelectSession}
      />
    );
    fireEvent.click(screen.getByText("Baseline sweep"));
    expect(onSelectSession).toHaveBeenCalledWith("s2");
  });

  it("renders cleanly with no sessions", () => {
    render(<SessionRail projectName="SDAR" sessions={[]} />);
    expect(screen.getByText("SDAR")).toBeInTheDocument();
  });

  it("merges a caller-provided className onto the root", () => {
    const { container } = render(
      <SessionRail projectName="SDAR" sessions={SESSIONS} className="custom-x" />
    );
    expect(container.firstElementChild).toHaveClass("custom-x");
  });
});
