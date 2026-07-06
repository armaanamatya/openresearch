import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, replace: vi.fn(), prefetch: vi.fn() }),
}));

import { PaperLanding } from "./PaperLanding";
import { isDevSessionSignedIn } from "@/lib/autoresearch/dev-session";

describe("PaperLanding", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    pushMock.mockClear();
  });

  afterEach(() => {
    window.sessionStorage.clear();
  });

  it("renders the welcome eyebrow", () => {
    render(<PaperLanding arxivId="2605.15155" />);
    expect(screen.getByText(/welcome to openresearch/i)).toBeInTheDocument();
  });

  it("renders the arXiv id prominently", () => {
    render(<PaperLanding arxivId="2605.15155" />);
    expect(screen.getByText(/arxiv 2605\.15155/i)).toBeInTheDocument();
  });

  it("renders a fail-soft fallback title when no title is supplied", () => {
    render(<PaperLanding arxivId="2605.15155" />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/reproduce this paper/i);
  });

  it("renders a supplied title verbatim instead of the fallback", () => {
    render(
      <PaperLanding
        arxivId="2605.15155"
        title="Self-Distilled Agentic Reinforcement Learning"
      />
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Self-Distilled Agentic Reinforcement Learning"
    );
  });

  it("renders the two OpenResearch body paragraphs", () => {
    render(<PaperLanding arxivId="2605.15155" />);
    expect(
      screen.getByText(/openresearch deploys an agent to build a minimal reproduction/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/direct the agent to run larger experiments from the paper/i)
    ).toBeInTheDocument();
  });

  it("renders the sign-in CTA as a real button", () => {
    render(<PaperLanding arxivId="2605.15155" />);
    const cta = screen.getByRole("button", { name: /sign in to start/i });
    expect(cta).toBeInTheDocument();
    expect(cta.tagName).toBe("BUTTON");
  });

  it("marks the dev session signed-in and advances to the repo-confirm screen on click", () => {
    render(<PaperLanding arxivId="2605.15155" />);
    expect(isDevSessionSignedIn()).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: /sign in to start/i }));

    expect(isDevSessionSignedIn()).toBe(true);
    expect(pushMock).toHaveBeenCalledWith("/abs/2605.15155?confirm=1");
  });

  it("merges a caller-provided className onto the root", () => {
    const { container } = render(<PaperLanding arxivId="2605.15155" className="custom-x" />);
    expect(container.firstElementChild).toHaveClass("custom-x");
  });
});
