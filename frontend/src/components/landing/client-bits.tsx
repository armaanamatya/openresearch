"use client";

import { useNavScroll } from "./use-nav-scroll";
import { useRevealOnScroll } from "./use-reveal";

export function NavScrollMount({
  targetId,
  scrolledClass
}: {
  targetId: string;
  scrolledClass: string;
}): null {
  useNavScroll(targetId, scrolledClass);
  return null;
}

export function RevealMount(): null {
  useRevealOnScroll();
  return null;
}
