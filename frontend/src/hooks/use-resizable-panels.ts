"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

export type PanelKey = "replRail" | "reportRail" | "detailSidebar";

export interface PanelSizes {
  replRail: number;
  reportRail: number;
  detailSidebar: number;
}

export interface DragHandlerBundle {
  onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void;
  role: "separator";
  "aria-orientation": "vertical";
  "aria-valuemin": number;
  "aria-valuemax": number;
  "aria-valuenow": number;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const STORAGE_KEY = "reprolab:lab-workspace-sizes:v1";

const BOUNDS: Record<PanelKey, { min: number; max: number; default: number }> = {
  replRail:      { min: 180, max: 360, default: 240 },
  reportRail:    { min: 200, max: 360, default: 280 },
  detailSidebar: { min: 280, max: 520, default: 360 },
};

const KEYBOARD_STEP = 16;

// ── Helpers ───────────────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function defaultSizes(): PanelSizes {
  return {
    replRail:      BOUNDS.replRail.default,
    reportRail:    BOUNDS.reportRail.default,
    detailSidebar: BOUNDS.detailSidebar.default,
  };
}

function readStorage(): PanelSizes {
  try {
    const raw = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
    if (!raw) return defaultSizes();
    const parsed = JSON.parse(raw) as Partial<PanelSizes>;
    const def = defaultSizes();
    return {
      replRail:      clamp(parsed.replRail      ?? def.replRail,      BOUNDS.replRail.min,      BOUNDS.replRail.max),
      reportRail:    clamp(parsed.reportRail    ?? def.reportRail,    BOUNDS.reportRail.min,    BOUNDS.reportRail.max),
      detailSidebar: clamp(parsed.detailSidebar ?? def.detailSidebar, BOUNDS.detailSidebar.min, BOUNDS.detailSidebar.max),
    };
  } catch {
    return defaultSizes();
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useResizablePanels() {
  // SSR-safe: render with defaults on the server AND on the client's first
  // render so the hydrated HTML matches. We then load persisted sizes from
  // localStorage in a post-mount effect; the brief visual flip (defaults →
  // stored) is acceptable and avoids a server↔client hydration mismatch.
  const [sizes, setSizes] = useState<PanelSizes>(defaultSizes);

  // viewport-aware collapse flags
  const [collapsedByViewport, setCollapsedByViewport] = useState({
    replRail: false,
    reportRail: false,
  });

  // Debounce timer ref for localStorage writes
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Mount: load persisted sizes from localStorage ───────────────────────
  useEffect(() => {
    // queueMicrotask defers the setState past the initial commit so React
    // never sees a setState-in-effect during the same render frame (caught
    // by react-hooks/set-state-in-effect under the React Compiler ESLint).
    queueMicrotask(() => setSizes(readStorage()));
  }, []);

  // ── Mount: set up matchMedia listeners ──────────────────────────────────
  useEffect(() => {
    const matchMedia = typeof window.matchMedia === "function"
      ? window.matchMedia.bind(window)
      : ((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: () => {},
          removeListener: () => {},
          addEventListener: () => {},
          removeEventListener: () => {},
          dispatchEvent: () => false,
        }) as MediaQueryList);
    const mq1200 = matchMedia("(max-width: 1199px)");
    const mq900  = matchMedia("(max-width: 899px)");

    function sync() {
      setCollapsedByViewport({
        replRail:   mq1200.matches,
        reportRail: mq900.matches,
      });
    }

    sync();
    mq1200.addEventListener("change", sync);
    mq900.addEventListener("change", sync);
    return () => {
      mq1200.removeEventListener("change", sync);
      mq900.removeEventListener("change", sync);
    };
  }, []);

  // ── Bounded write + debounced localStorage persist ───────────────────────
  const setSize = useCallback((panel: PanelKey, px: number) => {
    const bounded = clamp(px, BOUNDS[panel].min, BOUNDS[panel].max);
    setSizes((prev) => {
      const next = { ...prev, [panel]: bounded };
      // debounced write
      if (saveTimer.current !== null) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        } catch {
          // storage quota / private browsing — silently ignore
        }
      }, 200);
      return next;
    });
  }, []);

  // ── Drag handle factory ───────────────────────────────────────────────────
  //
  // `direction` indicates which side of the canvas the panel lives on:
  //   "right" → panel is to the LEFT of canvas (replRail); dragging right grows it
  //   "left"  → panel is to the RIGHT of canvas (reportRail, detailSidebar); dragging left grows it
  //
  // sizes ref so dragHandle closures can read current value without being
  // recreated on every size change (avoids stale closure without extra deps).
  const sizesRef = useRef(sizes);
  useEffect(() => { sizesRef.current = sizes; }, [sizes]);

  const dragHandle = useCallback(
    (panel: PanelKey, direction: "left" | "right"): DragHandlerBundle => {
      function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
        e.preventDefault();
        const startX = e.clientX;
        const currentSize = sizesRef.current[panel];

        function onMove(ev: PointerEvent) {
          const delta = direction === "right"
            ? ev.clientX - startX
            : startX - ev.clientX;
          const next = clamp(currentSize + delta, BOUNDS[panel].min, BOUNDS[panel].max);
          // use requestAnimationFrame to avoid jank during fast drags
          requestAnimationFrame(() => {
            setSizes((prev) => {
              const nextSizes = { ...prev, [panel]: next };
              if (saveTimer.current !== null) clearTimeout(saveTimer.current);
              saveTimer.current = setTimeout(() => {
                try {
                  localStorage.setItem(STORAGE_KEY, JSON.stringify(nextSizes));
                } catch { /* quota */ }
              }, 200);
              return nextSizes;
            });
          });
        }

        function onUp() {
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          window.removeEventListener("pointercancel", onUp);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);

        (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
      }

      function onKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        e.preventDefault();
        const shift = e.key === "ArrowRight" ? KEYBOARD_STEP : -KEYBOARD_STEP;
        const signedShift = direction === "right" ? shift : -shift;
        setSizes((prev) => {
          const next = { ...prev, [panel]: clamp(prev[panel] + signedShift, BOUNDS[panel].min, BOUNDS[panel].max) };
          if (saveTimer.current !== null) clearTimeout(saveTimer.current);
          saveTimer.current = setTimeout(() => {
            try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch { /* quota */ }
          }, 200);
          return next;
        });
      }

      return {
        onPointerDown,
        onKeyDown,
        role: "separator",
        "aria-orientation": "vertical",
        "aria-valuemin": BOUNDS[panel].min,
        "aria-valuemax": BOUNDS[panel].max,
        "aria-valuenow": sizesRef.current[panel],
      };
    },
    []
  );

  return { sizes, setSize, dragHandle, collapsedByViewport };
}
