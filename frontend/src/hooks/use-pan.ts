"use client";

import { useEffect, useRef } from "react";

/**
 * Pan/drag interaction for the lab workflow canvas.
 *
 * Returns a wrapRef to attach to the scrollable container, a dragRef
 * to read drag state from (e.g., for suppressing clicks during pan),
 * and an onMouseDown handler.
 *
 * Centers the viewport on the canvas's middle on mount.
 */
export function usePan() {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef({
    active: false,
    moved: false,
    slx: 0,
    sx: 0,
    sty: 0,
    sy: 0
  });

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    wrap.scrollLeft = Math.max(0, 740 - wrap.clientWidth / 2 + 100);
    wrap.scrollTop = Math.max(0, 310 - wrap.clientHeight / 2 + 40);
  }, []);

  useEffect(() => {
    // PointerEvent unifies mouse / trackpad / touch / pen so the canvas pans
    // reliably across input types. Previously only MouseEvent was wired
    // which broke on devices where pointer-only events fire (trackpads in
    // certain modes, tablets, some browser configurations).
    function onMove(event: PointerEvent) {
      const drag = dragRef.current;
      if (!drag.active || !wrapRef.current) return;
      wrapRef.current.scrollLeft = drag.slx - (event.clientX - drag.sx);
      wrapRef.current.scrollTop = drag.sty - (event.clientY - drag.sy);
      if (Math.abs(event.clientX - drag.sx) + Math.abs(event.clientY - drag.sy) > 4) {
        drag.moved = true;
      }
    }
    function onUp() {
      dragRef.current.active = false;
      if (wrapRef.current) {
        wrapRef.current.style.cursor = "grab";
      }
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, []);

  function onPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if ((event.target as HTMLElement).closest("[data-node]")) return;
    // Ignore secondary buttons so right-click doesn't grab the canvas.
    if (event.button !== 0 && event.pointerType === "mouse") return;
    const wrap = wrapRef.current;
    if (!wrap) return;
    dragRef.current = {
      active: true,
      moved: false,
      slx: wrap.scrollLeft,
      sx: event.clientX,
      sty: wrap.scrollTop,
      sy: event.clientY
    };
    wrap.style.cursor = "grabbing";
  }

  return { wrapRef, dragRef, onPointerDown };
}
