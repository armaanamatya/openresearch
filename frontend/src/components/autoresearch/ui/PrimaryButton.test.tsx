import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { PrimaryButton } from "./PrimaryButton";
import styles from "./PrimaryButton.module.css";

describe("PrimaryButton", () => {
  it("renders its children as the button label", () => {
    render(<PrimaryButton>Start autoresearch</PrimaryButton>);
    expect(
      screen.getByRole("button", { name: "Start autoresearch" })
    ).toBeInTheDocument();
  });

  it("fires onClick when clicked", () => {
    const onClick = vi.fn();
    render(<PrimaryButton onClick={onClick}>Go</PrimaryButton>);
    fireEvent.click(screen.getByRole("button", { name: "Go" }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("respects disabled: marks the button disabled and suppresses onClick", () => {
    const onClick = vi.fn();
    render(
      <PrimaryButton onClick={onClick} disabled>
        Go
      </PrimaryButton>
    );
    const button = screen.getByRole("button", { name: "Go" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("defaults to type=\"button\" so it never accidentally submits a form", () => {
    render(<PrimaryButton>Go</PrimaryButton>);
    expect(screen.getByRole("button")).toHaveAttribute("type", "button");
  });

  it("forwards an explicit type prop", () => {
    render(<PrimaryButton type="submit">Save</PrimaryButton>);
    expect(screen.getByRole("button")).toHaveAttribute("type", "submit");
  });

  it("carries the neo-brutalist button class", () => {
    render(<PrimaryButton>Go</PrimaryButton>);
    expect(screen.getByRole("button")).toHaveClass(styles.button);
  });

  it("merges a caller-provided className", () => {
    render(<PrimaryButton className="custom-x">Go</PrimaryButton>);
    expect(screen.getByRole("button")).toHaveClass("custom-x");
  });
});
