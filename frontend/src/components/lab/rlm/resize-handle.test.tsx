import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ResizeHandle } from "./resize-handle";
import type { DragHandlerBundle } from "../../../hooks/use-resizable-panels";

function makeBundle(overrides: Partial<DragHandlerBundle> = {}): DragHandlerBundle {
  return {
    onPointerDown: vi.fn(),
    onKeyDown: vi.fn(),
    role: "separator",
    "aria-orientation": "vertical",
    "aria-valuemin": 180,
    "aria-valuemax": 360,
    "aria-valuenow": 240,
    ...overrides,
  };
}

describe("ResizeHandle", () => {
  // ── Render ──────────────────────────────────────────────────────────────────

  it("renders the handle element", () => {
    render(<ResizeHandle {...makeBundle()} />);
    expect(screen.getByTestId("resize-handle")).toBeInTheDocument();
  });

  it("returns null when disabled", () => {
    render(<ResizeHandle {...makeBundle()} disabled />);
    expect(screen.queryByTestId("resize-handle")).toBeNull();
  });

  // ── ARIA attributes ─────────────────────────────────────────────────────────

  it("has role=separator", () => {
    render(<ResizeHandle {...makeBundle()} />);
    const el = screen.getByTestId("resize-handle");
    expect(el).toHaveAttribute("role", "separator");
  });

  it("has aria-orientation=vertical", () => {
    render(<ResizeHandle {...makeBundle()} />);
    expect(screen.getByTestId("resize-handle")).toHaveAttribute("aria-orientation", "vertical");
  });

  it("reflects aria-valuemin/max/now", () => {
    render(<ResizeHandle {...makeBundle({ "aria-valuemin": 200, "aria-valuemax": 500, "aria-valuenow": 350 })} />);
    const el = screen.getByTestId("resize-handle");
    expect(el).toHaveAttribute("aria-valuemin", "200");
    expect(el).toHaveAttribute("aria-valuemax", "500");
    expect(el).toHaveAttribute("aria-valuenow", "350");
  });

  it("is focusable (tabIndex=0)", () => {
    render(<ResizeHandle {...makeBundle()} />);
    expect(screen.getByTestId("resize-handle")).toHaveAttribute("tabindex", "0");
  });

  // ── Pointer events ──────────────────────────────────────────────────────────

  it("calls onPointerDown when pointer is pressed", () => {
    const onPointerDown = vi.fn();
    render(<ResizeHandle {...makeBundle({ onPointerDown })} />);
    fireEvent.pointerDown(screen.getByTestId("resize-handle"));
    expect(onPointerDown).toHaveBeenCalledTimes(1);
  });

  // ── Keyboard events ─────────────────────────────────────────────────────────

  it("calls onKeyDown for ArrowLeft", () => {
    const onKeyDown = vi.fn();
    render(<ResizeHandle {...makeBundle({ onKeyDown })} />);
    fireEvent.keyDown(screen.getByTestId("resize-handle"), { key: "ArrowLeft" });
    expect(onKeyDown).toHaveBeenCalledTimes(1);
    expect((onKeyDown.mock.calls[0][0] as KeyboardEvent).key).toBe("ArrowLeft");
  });

  it("calls onKeyDown for ArrowRight", () => {
    const onKeyDown = vi.fn();
    render(<ResizeHandle {...makeBundle({ onKeyDown })} />);
    fireEvent.keyDown(screen.getByTestId("resize-handle"), { key: "ArrowRight" });
    expect(onKeyDown).toHaveBeenCalledTimes(1);
    expect((onKeyDown.mock.calls[0][0] as KeyboardEvent).key).toBe("ArrowRight");
  });
});
