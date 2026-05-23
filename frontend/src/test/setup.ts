import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// jsdom does not implement matchMedia. Several hooks (useResizablePanels,
// useCommandPalette) read it during render or in mount effects; provide a
// safe default that tests can still spy on / override per-suite.
if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// Default app-router stub. Component tests that mount routes through
// Next's <App /> machinery still need their own mock; this just keeps
// hooks that call useRouter from blowing up when rendered in isolation.
vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>().catch(() => ({} as never));
  return {
    ...actual,
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    useSearchParams: () => ({ get: () => null }),
    usePathname: () => "/",
  };
});
