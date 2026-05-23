import path from "node:path";
import { test, expect } from "@playwright/test";

// Frontend static-values audit (2026-05-22) deliverable:
// Load every public route with no projectId and no live backend data, screenshot
// each, and assert no fixture-shaped leak markers reach the rendered DOM.
//
// This doubles as the category-C containment check the design spec calls out:
// with no dev affordance enabled (no `?rlmFixture=1`-style flag exists in this
// codebase) and no real run identifier, no fixture-shaped value can render.
//
// LEAK_MARKERS pin BOTH:
//   (a) the historical bug from `script-panel.tsx:22-30` — the
//       "91.4% / 492.3 / Reproduced With Caveats" preset values that the
//       workspace_fixture's demo_status template once rendered for a halted run.
//   (b) the descriptive-fallback literals removed by the 2026-05-22 audit
//       (S2 commits b9859f9..e6aafea). If a future change reintroduces a
//       fallback to any of these specific strings, this test catches it.
// "paper.pdf" is intentionally NOT in this list — it is a common short token
// that could legitimately appear in a future page (e.g. a downloads view)
// and would yield false positives. The specific descriptive literals below
// are unique enough to be a safe regression signal.

const LEAK_MARKERS = [
  "91.4",
  "492.3",
  "Reproduced With Caveats",
  "PaperBench-style final benchmark",
  "pending evaluator output",
  "sha256:pending",
];

// The full set of public page routes. `/` is a redirect to `/lab` (handled by
// Playwright's auto-follow); we exercise the destination directly.
const ROUTES = ["/lab", "/demo", "/library", "/paperbench", "/unlock"] as const;

const SCREENSHOT_DIR = path.join(
  __dirname,
  "..",
  "..",
  "docs",
  "design",
  "audit-2026-05-22-screenshots"
);

for (const route of ROUTES) {
  test(`route ${route} renders honest empty state with no leak markers`, async ({ page }) => {
    // `domcontentloaded` is enough here — we are asserting the rendered DOM,
    // not waiting for a long-running pipeline. `networkidle` would hang on
    // routes that hold an open SSE connection to the backend.
    await page.goto(route, { waitUntil: "domcontentloaded" });

    const body = (await page.textContent("body")) ?? "";

    for (const marker of LEAK_MARKERS) {
      expect(
        body,
        `route ${route} rendered the historical leak marker "${marker}" — fixture containment regressed`
      ).not.toContain(marker);
    }

    const slug = route.replace(/^\//, "").replace(/\//g, "-") || "root";
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, `${slug}.png`),
      fullPage: true,
    });
  });
}
