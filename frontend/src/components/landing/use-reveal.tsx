"use client";

import { useEffect } from "react";
import styles from "./landing.module.css";

/**
 * Adds the `.in` class to every `.reveal` element when it enters the
 * viewport. Honors prefers-reduced-motion (sets `.in` on all targets
 * immediately, no animation).
 *
 * Ported verbatim from docs/design/landing-source/index.html
 * lines 1534-1551.
 */
export function useRevealOnScroll(): void {
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const targets = document.querySelectorAll<HTMLElement>(`.${styles.reveal}`);
    if (reduced) {
      targets.forEach((el) => el.classList.add(styles.in));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add(styles.in);
            io.unobserve(e.target);
          }
        }
      },
      { rootMargin: "-8% 0px -8% 0px", threshold: 0.05 }
    );
    targets.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
}
