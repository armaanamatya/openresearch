import { test, expect } from "@playwright/test";

/**
 * Leaderboard e2e — fixture-driven empty-state + content rendering.
 *
 * The leaderboard fetches from REPROLAB_BACKEND_URL/leaderboard server-side at
 * request time. In the e2e harness the backend may not be running, in which
 * case the page falls back to []  ->  the empty-state placeholder. That's
 * exactly what we assert here.
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

    // Either the table is present (backend ran), or the empty-state copy.
    const table = page.locator("table").first();
    const empty = page.getByText(/no completed runs yet/i);
    const hasContent = (await Promise.race([
      table.isVisible().catch(() => false),
      empty.isVisible().catch(() => false),
    ])) || (await table.isVisible().catch(() => false)) || (await empty.isVisible().catch(() => false));
    expect(hasContent).toBe(true);
  });

  test("has a nav link back to the lab", async ({ page }) => {
    await page.goto("/leaderboard");
    await expect(page.getByRole("link", { name: "Lab" })).toBeVisible();
  });
});
