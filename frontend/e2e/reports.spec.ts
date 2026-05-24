import { test, expect } from "@playwright/test";

test.describe("Worker Reports panel", () => {
  test("shows worker reports for a seeded run", async ({ page }) => {
    // Navigate to the seeded run (created by scripts/seed-fake-run.sh)
    await page.goto("/lab?projectId=seed_reports_test");

    // The report rail should be visible with worker reports section
    await expect(page.getByText("worker reports").first()).toBeVisible({ timeout: 10_000 });

    // Should show the count badge (3 workers seeded)
    // Wait for reports to load via the API
    await page.waitForTimeout(3000);

    // At least one worker card should be visible
    const cards = page.getByTestId("worker-report-card");
    await expect(cards.first()).toBeVisible({ timeout: 10_000 });

    // Check for failed/blocked badge
    const statusBadges = page.getByTestId("worker-status-badge");
    const badgeTexts = await statusBadges.allTextContents();
    expect(badgeTexts.some((t) => t === "failed" || t === "blocked")).toBe(true);

    // Check for blocker text (SDK success-with-no-text)
    const blockers = page.getByTestId("worker-blocker");
    await expect(blockers.first()).toBeVisible();

    // Check for command with exit code
    const exitCodes = page.getByTestId("command-exit-code");
    await expect(exitCodes.first()).toBeVisible();

    // Raw JSON disclosure should be present
    const rawJson = page.getByTestId("raw-json-disclosure");
    await expect(rawJson.first()).toBeVisible();
  });
});
