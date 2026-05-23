"use client";

import { useEffect, useRef, useState } from "react";

/**
 * useCountUp — tween a numeric value to a target with cubic-out easing.
 *
 * Returns the currently-displayed value. The tween restarts whenever
 * `target` changes; cancelled if the component unmounts.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.3.
 */
export function useCountUp(target: number, durationMs = 400): number {
  const [value, setValue] = useState(target);
  const fromRef = useRef(target);
  const startRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const valueRef = useRef(target);
  valueRef.current = value;

  useEffect(() => {
    fromRef.current = valueRef.current;
    // Initialise start clock in the effect so the FIRST RAF tick sees a
    // non-zero elapsed window. (Lazy-init inside step would compute
    // elapsed=0 on the first frame and never advance.)
    startRef.current = performance.now();

    const step = (now: number) => {
      const elapsed = now - (startRef.current ?? now);
      const t = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - t, 3); // cubic-out
      const next = fromRef.current + (target - fromRef.current) * eased;
      setValue(next);

      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        rafRef.current = null;
      }
    };

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [target, durationMs]);

  return value;
}
