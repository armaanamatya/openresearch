"use client";

/**
 * Scroll-reveal removed for launch (2026-05-23). The JS-driven fade-in
 * hid below-fold content from SEO crawlers + full-page captures. Hook
 * kept as a no-op so the existing RevealMount client wrapper compiles.
 */
export function useRevealOnScroll(): void {
  /* intentional no-op */
}
