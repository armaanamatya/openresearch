import { fireEvent, render, screen } from "@testing-library/react";

import { ReproLabClient } from "./repro-lab-client";

describe("ReproLabClient", () => {
  it("starts from the upload view and transitions into the workflow view", () => {
    render(<ReproLabClient />);

    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(screen.getByPlaceholderText("arxiv.org/abs/2303.04137")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("arxiv.org/abs/2303.04137"), {
      target: { value: "arxiv.org/abs/2303.04137" }
    });
    fireEvent.click(screen.getByRole("button", { name: /begin/i }));

    expect(screen.getByRole("heading", { name: /arxiv\.org\/abs\/2303\.04137/i })).toBeInTheDocument();
    expect(screen.getByText(/agents complete/i)).toBeInTheDocument();
    expect(screen.getByText("Live activity")).toBeInTheDocument();
  });
});
