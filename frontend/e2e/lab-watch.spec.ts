/**
 * Monitoring Loop 4 — UI watch spec.
 *
 * Designed to be invoked repeatedly (every 5 min by scripts/loops/lab_watch_loop.sh)
 * against a *running* dev/prod server with an in-flight reproduction. Each
 * invocation is one cycle:
 *   - load /lab?projectId=<LAB_PROJECT_ID> and capture a timestamped screenshot
 *   - record any console errors/warnings observed during page load
 *   - assert <body> contains no React hydration-mismatch text
 *   - load /leaderboard and assert it renders without throwing
 *
 * Env knobs:
 *   LAB_PROJECT_ID — required for the lab cycle; if unset the spec only
 *                    exercises /leaderboard (a healthy UI even with no run).
 *   LAB_WATCH_SCREENSHOT_DIR — defaults to /tmp/playwright-ui-loop
 *   LAB_BASE_URL — pointed at by playwright.config.ts (default http://localhost:3001)
 *
 * This file is kept distinct from the existing rlm-lab.spec.ts (which uses the
 * deterministic ?rlmFixture=1) — the watch spec runs against the live UI as it
 * actually serves the in-flight project, so a regression that only shows up
 * with real SSE data still trips a failure.
 */

import { test, expect } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";

const SCREENSHOT_DIR =
  process.env.LAB_WATCH_SCREENSHOT_DIR ?? "/tmp/playwright-ui-loop";

const projectId = process.env.LAB_PROJECT_ID?.trim() || "";

const HYDRATION_MISMATCH_MARKERS = [
  "Hydration failed because",
  "did not match",
  "Text content did not match",
  "Hydration mismatch",
];

async function captureScreenshot(page: import("@playwright/test").Page, label: string) {
  await fs.mkdir(SCREENSHOT_DIR, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const file = path.join(SCREENSHOT_DIR, `${ts}-${label}.png`);
  await page.screenshot({ path: file, fullPage: true });
  console.log(`[lab-watch] screenshot: ${file}`);
  return file;
}

function attachConsoleCapture(page: import("@playwright/test").Page) {
  const errors: string[] = [];
  const warnings: string[] = [];
  page.on("console", (msg) => {
    const text = msg.text();
    // Suppress the Next.js dev-mode allowedDevOrigins noise — it fires on
    // every fresh load against 127.0.0.1 and is unrelated to UI regressions.
    if (text.includes("allowedDevOrigins")) return;
    if (msg.type() === "error") errors.push(text);
    if (msg.type() === "warning") warnings.push(text);
  });
  page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
  return { errors, warnings };
}

(projectId ? test : test.skip)(
  "lab-watch: /lab loads, screenshots, no console errors",
  async ({ page }) => {
    const { errors, warnings } = attachConsoleCapture(page);

    const response = await page.goto(`/lab?projectId=${encodeURIComponent(projectId)}`);
    expect(response, "navigation response").not.toBeNull();
    expect(response!.status(), "/lab HTTP status").toBeLessThan(500);

    // The RLM lab root container should mount within the navigation timeout.
    await expect(page.getByTestId("rlm-lab")).toBeVisible({ timeout: 30_000 });

    // Hydration-mismatch text never belongs in production; flag any.
    const bodyText = await page.locator("body").innerText();
    for (const marker of HYDRATION_MISMATCH_MARKERS) {
      expect(bodyText, `hydration marker '${marker}' must not appear`).not.toContain(marker);
    }

    await captureScreenshot(page, `lab-${projectId}`);

    // Console errors are hard failures; warnings are recorded for trend
    // analysis but don't fail the test (Next.js dev mode produces benign
    // warnings on hot-reload that would create perpetual false positives).
    if (warnings.length > 0) {
      console.log(`[lab-watch] ${warnings.length} console warning(s):\n${warnings.join("\n")}`);
    }
    expect(errors, "console / page errors during /lab load").toEqual([]);
  }
);

test("lab-watch: /leaderboard renders without throwing", async ({ page }) => {
  const { errors } = attachConsoleCapture(page);

  const response = await page.goto("/leaderboard");
  expect(response, "navigation response").not.toBeNull();
  expect(response!.status(), "/leaderboard HTTP status").toBeLessThan(500);

  // The leaderboard always renders at least an h1 heading, even with zero rows.
  await expect(page.locator("h1").first()).toBeVisible({ timeout: 30_000 });

  await captureScreenshot(page, "leaderboard");
  expect(errors, "console / page errors during /leaderboard load").toEqual([]);
});
