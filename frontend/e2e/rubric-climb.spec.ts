import { test, expect } from "@playwright/test";

/**
 * Rubric-climb panel e2e — drives the lab UI from the existing climb fixture
 * (?rlmFixture=1). The fixture is "instant-replayed" by RlmFixtureContent, so
 * the final state shows the full sparkline, area chips, and (when applicable)
 * the candidate-attribution tail.
 *
 * Docker-dependent live-stream tests (count-up animation during a real run,
 * mid-run resume) are tracked as deferred for the next session per the plan.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §6.4.
 */
test.describe("Rubric climb panel", () => {
  test("renders the SVG sparkline polyline once the fixture has folded", async ({ page }) => {
    await page.goto("/lab?rlmFixture=1");
    // The SVG sparkline replaces the prior bar sparkline; assert a polyline
    // is in the DOM after the fixture has fully reduced.
    await expect(page.locator("polyline").first()).toBeVisible({ timeout: 8000 });
  });

  test("renders per-area chips with status data attributes", async ({ page }) => {
    await page.goto("/lab?rlmFixture=1");
    // Area chips are tagged with data-area-chip + data-area.
    const firstChip = page.locator("[data-area-chip]").first();
    await expect(firstChip).toBeVisible();
    const areaName = await firstChip.getAttribute("data-area");
    expect(areaName).toBeTruthy();
  });

  test("renders the candidate-attribution tail after the climb", async ({ page }) => {
    await page.goto("/lab?rlmFixture=1");
    // The fixture produces a baseline 0.22 -> final 0.53 climb with attributable
    // candidates between rubric_score events; the strip should show "from candidate ...".
    await expect(page.getByText(/from candidate/i).first()).toBeVisible({ timeout: 8000 });
  });

  test("emits no console errors or failed network requests (criterion 7)", async ({ page }) => {
    const consoleErrors: string[] = [];
    const failedRequests: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("requestfailed", (req) => {
      failedRequests.push(`${req.method()} ${req.url()} — ${req.failure()?.errorText}`);
    });

    await page.goto("/lab?rlmFixture=1");
    await page.waitForLoadState("networkidle").catch(() => {});
    await page.waitForTimeout(500);

    // Filter known dev-mode noise: HMR pings, ResizeObserver loops, dev-only
    // source-map 404s — none of these reflect a real production-visible defect.
    const realErrors = consoleErrors.filter(
      (e) => !/HMR|ResizeObserver|source map|Fast Refresh/i.test(e),
    );
    expect(realErrors).toEqual([]);
    expect(failedRequests).toEqual([]);
  });
});
