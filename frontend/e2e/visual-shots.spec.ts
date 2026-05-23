import { test } from "@playwright/test";

const SHOTS = "/Volumes/CS_Stuff/openresearch/docs/design/2026-05-23-rubric-leaderboard-screens";

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
