import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { RepoField } from "./RepoField";

describe("RepoField", () => {
  it("renders the label and the current value", () => {
    render(
      <RepoField label="GitHub repository" value="ZJU-REAL/SDAR" onChange={() => {}} />
    );
    const input = screen.getByLabelText("GitHub repository");
    expect(input).toBeInTheDocument();
    expect(input).toHaveValue("ZJU-REAL/SDAR");
  });

  it("associates the label with a real <input> via htmlFor/id", () => {
    render(<RepoField label="GitHub repository" value="" onChange={() => {}} />);
    const input = screen.getByLabelText("GitHub repository");
    expect(input.tagName).toBe("INPUT");
  });

  it("fires onChange with the new value when the input changes", () => {
    const onChange = vi.fn();
    render(<RepoField label="GitHub repository" value="" onChange={onChange} />);
    const input = screen.getByLabelText("GitHub repository");
    fireEvent.change(input, { target: { value: "ZJU-REAL/SDAR" } });
    expect(onChange).toHaveBeenCalledWith("ZJU-REAL/SDAR");
  });

  it("renders a github-link affordance pointing at the current value", () => {
    render(
      <RepoField label="GitHub repository" value="ZJU-REAL/SDAR" onChange={() => {}} />
    );
    const link = screen.getByRole("link", { name: /ZJU-REAL\/SDAR/ });
    expect(link).toHaveAttribute("href", "https://github.com/ZJU-REAL/SDAR");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });

  it("omits the github-link affordance when there is no value yet", () => {
    render(<RepoField label="GitHub repository" value="" onChange={() => {}} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("forwards a placeholder to the input", () => {
    render(
      <RepoField
        label="GitHub repository"
        value=""
        onChange={() => {}}
        placeholder="owner/repo"
      />
    );
    expect(screen.getByLabelText("GitHub repository")).toHaveAttribute(
      "placeholder",
      "owner/repo"
    );
  });

  it("merges a caller-provided className onto the root", () => {
    const { container } = render(
      <RepoField label="GitHub repository" value="" onChange={() => {}} className="custom-x" />
    );
    expect(container.firstElementChild).toHaveClass("custom-x");
  });
});
