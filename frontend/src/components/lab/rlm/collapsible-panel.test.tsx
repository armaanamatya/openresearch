import { render, screen, fireEvent } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { CollapsiblePanel } from "./collapsible-panel";

const KEY = "reprolab:panel-collapsed:test-key";

describe("CollapsiblePanel", () => {
  afterEach(() => {
    window.localStorage.clear();
  });

  it("renders the body expanded by default", () => {
    render(
      <CollapsiblePanel storageKey="test-key" title="Scores" summary="0.62">
        <p>BODY CONTENT</p>
      </CollapsiblePanel>
    );
    expect(screen.getByText("BODY CONTENT")).toBeInTheDocument();
    expect(screen.getByText("Scores")).toBeInTheDocument();
    expect(screen.getByText("0.62")).toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "true");
  });

  it("minimizes the body on click but keeps the header summary visible", () => {
    render(
      <CollapsiblePanel storageKey="test-key" title="Scores" summary="0.62">
        <p>BODY CONTENT</p>
      </CollapsiblePanel>
    );
    fireEvent.click(screen.getByRole("button"));
    // Body is unmounted (space reclaimed) — but title + summary stay in the header.
    expect(screen.queryByText("BODY CONTENT")).not.toBeInTheDocument();
    expect(screen.getByText("Scores")).toBeInTheDocument();
    expect(screen.getByText("0.62")).toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "false");
  });

  it("restores the body when expanded again", () => {
    render(
      <CollapsiblePanel storageKey="test-key" title="Scores">
        <p>BODY CONTENT</p>
      </CollapsiblePanel>
    );
    const button = screen.getByRole("button");
    fireEvent.click(button); // collapse
    fireEvent.click(button); // expand
    expect(screen.getByText("BODY CONTENT")).toBeInTheDocument();
    expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("persists the collapsed state to localStorage", () => {
    render(
      <CollapsiblePanel storageKey="test-key" title="Scores">
        <p>BODY CONTENT</p>
      </CollapsiblePanel>
    );
    fireEvent.click(screen.getByRole("button"));
    expect(window.localStorage.getItem(KEY)).toBe("1");
    fireEvent.click(screen.getByRole("button"));
    expect(window.localStorage.getItem(KEY)).toBe("0");
  });

  it("hydrates the collapsed state from localStorage on mount", () => {
    window.localStorage.setItem(KEY, "1");
    render(
      <CollapsiblePanel storageKey="test-key" title="Scores">
        <p>BODY CONTENT</p>
      </CollapsiblePanel>
    );
    expect(screen.queryByText("BODY CONTENT")).not.toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "false");
  });

  it("respects defaultCollapsed when there is no persisted preference", () => {
    render(
      <CollapsiblePanel storageKey="test-key" title="Scores" defaultCollapsed>
        <p>BODY CONTENT</p>
      </CollapsiblePanel>
    );
    expect(screen.queryByText("BODY CONTENT")).not.toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "false");
  });
});
