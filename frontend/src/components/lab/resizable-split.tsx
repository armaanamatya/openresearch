"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import "./resizable-split.css";

// User-resizable horizontal split between the canvas (left) and the
// node-details panel (right). Replaces the old fixed-width EdgeDrawer in
// WorkflowView so the operator can dial the canvas vs panel ratio for
// the screen they're on. Persisted to localStorage so the choice
// survives a refresh.
const STORAGE_KEY = "reprolab:split-ratio";
const MIN_RATIO = 0.25;
const MAX_RATIO = 0.85;

export function ResizableSplit({
  left,
  right
}: {
  left: ReactNode;
  right: ReactNode;
}) {
  const [ratio, setRatio] = useState(0.7);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const value = parseFloat(stored);
        if (Number.isFinite(value) && value >= MIN_RATIO && value <= MAX_RATIO) {
          // eslint-disable-next-line react-hooks/set-state-in-effect
          setRatio(value);
        }
      }
    } catch {
      // localStorage may be disabled
    }
  }, []);

  const persistRatio = useCallback((v: number) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(v));
    } catch {
      // non-fatal
    }
  }, []);

  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!dragRef.current || !wrapRef.current) return;
      const rect = wrapRef.current.getBoundingClientRect();
      const next = (e.clientX - rect.left) / rect.width;
      const clamped = Math.min(MAX_RATIO, Math.max(MIN_RATIO, next));
      setRatio(clamped);
    }
    function onUp() {
      if (dragRef.current) {
        dragRef.current = false;
        persistRatio(ratio);
      }
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [ratio, persistRatio]);

  return (
    <div ref={wrapRef} className="resizable-split">
      <div className="resizable-split-pane" style={{ flexBasis: `${ratio * 100}%` }}>
        {left}
      </div>
      <div
        className="resizable-split-divider"
        onMouseDown={() => {
          dragRef.current = true;
        }}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize canvas / node details"
      />
      <div className="resizable-split-pane" style={{ flexBasis: `${(1 - ratio) * 100}%` }}>
        {right}
      </div>
    </div>
  );
}
