import { test } from "@playwright/test";
import path from "node:path";

// Screenshot refresher. By default shots land in the untracked
// test-results/ dir (CI-safe, no churn of tracked artifacts). Set
// UPDATE_DESIGN_SHOTS=1 to deliberately refresh the TRACKED design
// screenshots under docs/design/ (the old behavior — which used to run
// unconditionally via an absolute /Volumes/... macOS path, silently
// dirtying the worktree on every e2e run and crashing ENOENT on CI;
// audit 2026-06-10).
const SHOTS = process.env.UPDATE_DESIGN_SHOTS === "1"
  ? path.join(__dirname, "../../docs/design/2026-05-23-rubric-leaderboard-screens")
  : path.join(__dirname, "../test-results/design-shots");

test.describe.configure({ mode: "serial" });

test("desktop /leaderboard", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/leaderboard");
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.screenshot({ path: `${SHOTS}/leaderboard-desktop.png`, fullPage: true });
});

test("desktop /lab climb fixture", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/lab?rlmFixture=1");
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${SHOTS}/lab-climb-desktop.png`, fullPage: true });
});

test("mobile /leaderboard", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/leaderboard");
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.screenshot({ path: `${SHOTS}/leaderboard-mobile.png`, fullPage: true });
});

test("mobile /lab climb fixture", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/lab?rlmFixture=1");
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${SHOTS}/lab-climb-mobile.png`, fullPage: true });
});
