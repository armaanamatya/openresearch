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
    function onMove(event: MouseEvent) {
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
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  function onMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    if ((event.target as HTMLElement).closest("[data-node]")) return;
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

  return { wrapRef, dragRef, onMouseDown };
}
