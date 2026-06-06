import { test, expect } from "@playwright/test";

/**
 * Leaderboard e2e — fixture-driven empty-state + content rendering.
 *
 * The leaderboard fetches from OPENRESEARCH_BACKEND_URL/leaderboard server-side at
 * request time. In the e2e harness the backend may not be running, in which
 * case the page must render an explicit unavailable state instead of pretending
 * an empty leaderboard loaded successfully.
 *
 * Docker-dependent populated-state e2e (seed N fixture runs and verify
 * ranking + row click navigation) is tracked as deferred per the plan.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §6.4, §3 #12.
 */
test.describe("/leaderboard", () => {
  test("renders the page with title and either rows or empty-state", async ({ page }) => {
    await page.goto("/leaderboard");
    await expect(page.getByRole("heading", { name: "Leaderboard" })).toBeVisible();

    // Either the table is present (backend ran), the empty-state copy rendered,
    // or the explicit backend-outage state is visible.
    const table = page.locator("table").first();
    const empty = page.getByText(/no completed runs yet/i);
    const unavailable = page.getByText(/leaderboard unavailable/i);
    const hasContent = (await Promise.race([
      table.isVisible().catch(() => false),
      empty.isVisible().catch(() => false),
      unavailable.isVisible().catch(() => false),
    ])) || (await table.isVisible().catch(() => false))
      || (await empty.isVisible().catch(() => false))
      || (await unavailable.isVisible().catch(() => false));
    expect(hasContent).toBe(true);
  });

  test("has a nav link back to the lab", async ({ page }) => {
    await page.goto("/leaderboard");
    await expect(page.getByRole("link", { name: "Lab" })).toBeVisible();
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

    await page.goto("/leaderboard");
    await page.waitForLoadState("networkidle").catch(() => {});
    await page.waitForTimeout(500);

    const realErrors = consoleErrors.filter(
      (e) => !/HMR|ResizeObserver|source map|Fast Refresh/i.test(e),
    );
    expect(realErrors).toEqual([]);
    expect(
      failedRequests.filter((failure) => !/\\?_rsc=.*net::ERR_ABORTED/.test(failure)),
    ).toEqual([]);
  });
});
