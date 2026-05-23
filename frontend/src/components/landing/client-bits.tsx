"use client";

import { useNavScroll } from "./use-nav-scroll";

/**
 * Mounts the useNavScroll side-effect inside an otherwise server-
 * rendered LandingPage. Renders nothing; exists purely to push the
 * client/server boundary down to the smallest possible leaf so that
 * the rest of the landing tree stays server-rendered for SSR/SEO.
 */
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
