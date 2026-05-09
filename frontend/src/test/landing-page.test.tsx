import { render, screen } from "@testing-library/react";

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
});
