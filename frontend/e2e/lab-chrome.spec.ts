import { test, expect } from "@playwright/test";

// Focused e2e coverage for three chrome features that survived the Task 4
// lab-UI rewrite: Cmd-K command palette, ? shortcut overlay, and /library
// status-filter pills.
//
// These tests run against the dev server at LAB_BASE_URL (default
// http://localhost:3001). Use relative paths so playwright.config's baseURL
// resolves them correctly — going to 127.0.0.1 directly breaks Next.js 16
// dev-mode hydration via the allowedDevOrigins safety check.

test("Cmd-K opens the command palette", async ({ page }) => {
  await page.goto("/lab");

  // Click body first so the window-level keydown hook in
  // use-command-palette.ts has a focus target to fire against.
  await page.locator("body").click();

  await page.keyboard.press("ControlOrMeta+k");

  // cmdk@1 renders a <Command> root with [cmdk-root]; it does not set
  // role="dialog", so we key off the attribute directly.
  const palette = page.locator("[cmdk-root]");
  await expect(palette).toBeVisible({ timeout: 5000 });

  // The Library navigate item must appear in the list.
  await expect(palette.getByText("Library", { exact: false }).first()).toBeVisible();

  // Esc closes the palette.
  await page.keyboard.press("Escape");
  await expect(palette).toBeHidden({ timeout: 2000 });
});

test("? opens the keyboard-shortcut overlay", async ({ page }) => {
  // Use the fixture path: it renders <RlmLab> with no upload form, so no
  // INPUT element receives autofocus and the ? keydown reaches the overlay hook.
  await page.goto("/lab?rlmFixture=1");

  // Wait for the RLM lab root to appear before firing the shortcut, so we
  // know the React tree (and the useShortcutOverlay hook) is mounted.
  // Use .first() — the fixture path can render two [data-testid="rlm-lab"]
  // elements (one hidden via SSR, one hydrated), so strict mode would fail.
  await expect(page.getByTestId("rlm-lab").first()).toBeVisible({ timeout: 10000 });

  await page.keyboard.press("?");

  const overlay = page.getByRole("dialog", { name: /keyboard shortcuts/i });
  await expect(overlay).toBeVisible({ timeout: 3000 });

  // Esc closes the overlay.
  await page.keyboard.press("Escape");
  await expect(overlay).toBeHidden({ timeout: 2000 });
});

test("/library status-filter pill narrows the run list", async ({ page }) => {
  // The library page server-renders its initial run list. When a filter pill
  // is clicked, LibraryTable fires a client-side fetch to /api/runs/list —
  // that fetch is what we mock here. We can't intercept the SSR fetch
  // (it goes server-to-backend, outside Playwright's routing), so we let
  // the initial render use real data and only mock the filtered refetch.
  //
  // Strategy:
  //  1. Navigate and wait for the real initial render (SSR).
  //  2. Set up a route mock for the "Completed" filter request.
  //  3. Click the "Completed" pill and assert the list responds.

  await page.goto("/library");

  // Status filter pills must be visible — confirms the chrome rendered.
  const completedPill = page.getByRole("tab", { name: "Completed" });
  await expect(completedPill).toBeVisible({ timeout: 10000 });

  // Mock only the client-side filtered fetch. Return exactly one completed run.
  const COMPLETED_ONLY = [
    {
      projectId: "prj_chrome_test_completed",
      status: "completed",
      sourceLabel: "Filter test paper (completed)",
      updatedAt: new Date().toISOString(),
      benchmark: null
    }
  ];
  await page.route("**/api/runs/list*", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(COMPLETED_ONLY)
    });
  });

  // Click the Completed pill — this triggers the client-side fetch.
  await completedPill.click();

  // The mocked response has exactly 1 run. Assert the table now shows it.
  const filteredRow = page.locator('tr:has(a[href*="prj_chrome_test_completed"])');
  await expect(filteredRow).toBeVisible({ timeout: 5000 });

  // Also confirm aria-selected toggled so the pill shows as active.
  await expect(completedPill).toHaveAttribute("aria-selected", "true");
});
