import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { PaperTabs } from "./PaperTabs";

describe("PaperTabs", () => {
  it("renders the four tabs", () => {
    render(<PaperTabs active="Paper" onTabChange={() => {}} />);
    expect(screen.getByRole("tab", { name: /paper/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /blog/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /autoresearch/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /audio/i })).toBeInTheDocument();
  });

  it("wraps the tabs in a real tablist", () => {
    render(<PaperTabs active="Paper" onTabChange={() => {}} />);
    expect(screen.getByRole("tablist")).toBeInTheDocument();
  });

  it("marks the active tab via aria-selected, and only that one", () => {
    render(<PaperTabs active="Autoresearch" onTabChange={() => {}} />);
    expect(screen.getByRole("tab", { name: /autoresearch/i })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByRole("tab", { name: /^paper/i })).toHaveAttribute(
      "aria-selected",
      "false"
    );
    expect(screen.getByRole("tab", { name: /blog/i })).toHaveAttribute(
      "aria-selected",
      "false"
    );
    expect(screen.getByRole("tab", { name: /audio/i })).toHaveAttribute(
      "aria-selected",
      "false"
    );
  });

  it("fires onTabChange with the clicked tab's name", () => {
    const onTabChange = vi.fn();
    render(<PaperTabs active="Paper" onTabChange={onTabChange} />);
    fireEvent.click(screen.getByRole("tab", { name: /blog/i }));
    expect(onTabChange).toHaveBeenCalledWith("Blog");
  });

  it("renders each tab as a real button", () => {
    render(<PaperTabs active="Paper" onTabChange={() => {}} />);
    expect(screen.getByRole("tab", { name: /^paper/i }).tagName).toBe("BUTTON");
  });

  it("merges a caller-provided className onto the root", () => {
    const { container } = render(
      <PaperTabs active="Paper" onTabChange={() => {}} className="custom-x" />
    );
    expect(container.firstElementChild).toHaveClass("custom-x");
  });
});
