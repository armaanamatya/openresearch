import path from "node:path";
import fs from "node:fs";
import { test, expect } from "@playwright/test";

// Frontend static-values audit (2026-05-22) — post-Phase-6 regression guard.
// Load every public route with no projectId and no live backend data, screenshot
// each, and assert no fixture-shaped leak markers reach the rendered DOM.
//
// This doubles as the category-C containment check the design spec calls out:
// with no dev affordance enabled (no `?rlmFixture=1` query string) and no real
// run identifier, no fixture-shaped value can render.
//
// LEAK_MARKERS pin literals that historically rendered as hardcoded fallbacks:
//   (a) the workspace_fixture's `demo_status` template once rendered "91.4%",
//       "492.3", "Reproduced With Caveats" for a halted demo run (deleted by
//       Phase 6 / PR #72, but kept here as a regression guard against any
//       future code accidentally reintroducing the same shape).
//   (b) descriptive-fallback literals that lived in the deleted
//       `script-panel.tsx:22-30` (also Phase-6-deleted). The 2026-05-22
//       static-values audit (`docs/design/static-values-audit-2026-05-22.md`)
//       flagged these as "rendered hardcoded values masquerading as real
//       data." The source files are gone; this guard prevents the strings
//       from reappearing.
// "paper.pdf" is intentionally NOT in this list — it is a common short token
// that could legitimately appear in a future page (e.g. a downloads view) and
// would yield false positives. The descriptive literals below are unique
// enough to be a safe regression signal.

const LEAK_MARKERS = [
  "91.4",
  "492.3",
  "Reproduced With Caveats",
  "PaperBench-style final benchmark",
  "pending evaluator output",
  "sha256:pending",
];

// Public page routes after the Phase-6 cutover. `/demo` was deleted by PR #72
// (`refactor(rlm): delete /demo route, dead node-runner spawn path, and the
// pipeline.py shim`) so it's excluded here; if `/demo` is ever resurrected
// (intentionally or otherwise), add it back to this list.
const ROUTES = ["/lab", "/library", "/paperbench", "/unlock"] as const;

const SCREENSHOT_DIR = path.join(
  __dirname,
  "..",
  "tmp-screens",
  "static-values-route-pass"
);

fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

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
