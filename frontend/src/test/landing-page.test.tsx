import { act, fireEvent, render, screen } from "@testing-library/react";

import HomePage from "../app/page";

describe("landing page", () => {
  it("renders the main Stellar headline", () => {
    render(<HomePage />);

    expect(screen.getByText(/work smarter\. move faster\./i)).toBeInTheDocument();
  });

  it("renders the navigation and primary CTA", () => {
    render(<HomePage />);

    expect(screen.getByText("Stellar.ai")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /begin free trial/i })).toBeInTheDocument();
  });

  it("renders all four tabs", () => {
    render(<HomePage />);

    for (const label of ["Analyse", "Train", "Testing", "Deploy"]) {
      expect(screen.getAllByRole("button", { name: new RegExp(label, "i") }).length).toBeGreaterThan(0);
    }
  });

  it("renders the autoplaying video stage", () => {
    render(<HomePage />);

    const video = screen.getByTestId("stellar-video");

    expect(video).toHaveAttribute("autoplay");
    expect(video).toHaveAttribute("loop");
    expect((video as HTMLVideoElement).muted).toBe(true);
  });

  it("shows analyse overlay content by default", () => {
    render(<HomePage />);

    expect(screen.getByText(/set up your ai workspace/i)).toBeInTheDocument();
  });

  it("renders the company logo rail", () => {
    render(<HomePage />);

    for (const label of ["INTERSCOPE", "SPOTIFY", "Nexera", "M3", "LAURA COLE", "vertex"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("switches overlay content when a tab is clicked", () => {
    render(<HomePage />);

    fireEvent.click(screen.getAllByRole("button", { name: /deploy/i })[0]);

    expect(screen.getByText(/deploy to production/i)).toBeInTheDocument();
  });

  it("auto-cycles overlays every 4 seconds", () => {
    vi.useFakeTimers();

    render(<HomePage />);
    expect(screen.getByText(/set up your ai workspace/i)).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(4000);
    });

    expect(screen.getByText(/ai model training/i)).toBeInTheDocument();

    vi.useRealTimers();
  });
});
