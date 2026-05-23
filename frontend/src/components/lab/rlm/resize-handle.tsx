"use client";

import type { DragHandlerBundle } from "../../../hooks/use-resizable-panels";
import styles from "./resize-handle.module.css";

interface ResizeHandleProps extends DragHandlerBundle {
  /** When true the handle is invisible and non-interactive (e.g. sidebar collapsed). */
  disabled?: boolean;
}

/**
 * ResizeHandle — a 4px-wide vertical drag handle between a panel and the canvas.
 *
 * Spreads the handler bundle returned by `useResizablePanels().dragHandle()`.
 * Keyboard: ArrowLeft / ArrowRight shift the panel size by 16px.
 */
export function ResizeHandle({
  disabled = false,
  onPointerDown,
  onKeyDown,
  role,
  "aria-orientation": ariaOrientation,
  "aria-valuemin": ariaValuemin,
  "aria-valuemax": ariaValuemax,
  "aria-valuenow": ariaValuenow,
}: ResizeHandleProps) {
  if (disabled) return null;

  return (
    <div
      className={styles.handle}
      role={role}
      aria-orientation={ariaOrientation}
      aria-valuemin={ariaValuemin}
      aria-valuemax={ariaValuemax}
      aria-valuenow={ariaValuenow}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      data-testid="resize-handle"
    >
      <div className={styles.line} aria-hidden="true" />
    </div>
  );
}
