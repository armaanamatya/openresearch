import { act, fireEvent, render, screen } from "@testing-library/react";

import HomePage from "../app/page";

describe("landing page", () => {
  it("renders the main ReproLab headline", () => {
    render(<HomePage />);

    expect(screen.getByText(/reproduce smarter\./i)).toBeInTheDocument();
    expect(screen.getByText(/verify faster\./i)).toBeInTheDocument();
  });

  it("renders the navigation and primary CTA", () => {
    render(<HomePage />);

    expect(screen.getByText("ReproLab")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /start now/i })).toHaveAttribute("href", "/lab");
    expect(screen.getByRole("link", { name: /get started free/i })).toHaveAttribute("href", "/lab");
  });

  it("renders all four tabs", () => {
    render(<HomePage />);

    for (const label of ["Ingest", "Reproduce", "Audit", "Report"]) {
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

  it("shows ingest overlay content by default", () => {
    render(<HomePage />);

    expect(screen.getByText(/read the paper first/i)).toBeInTheDocument();
  });

  it("renders the company logo rail", () => {
    render(<HomePage />);

    for (const label of ["INTERSCOPE", "SPOTIFY", "Nexera", "M3", "LAURA COLE", "vertex"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("switches overlay content when a tab is clicked", () => {
    render(<HomePage />);

    fireEvent.click(screen.getAllByRole("button", { name: /report/i })[0]);

    expect(screen.getByText(/ship the reproducibility packet/i)).toBeInTheDocument();
  });

  it("auto-cycles overlays every 4 seconds", () => {
    vi.useFakeTimers();

    render(<HomePage />);
    expect(screen.getByText(/read the paper first/i)).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(4000);
    });

    expect(screen.getByText(/build and run the stack/i)).toBeInTheDocument();

    vi.useRealTimers();
  });
});
