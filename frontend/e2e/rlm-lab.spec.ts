import { test, expect } from "@playwright/test";

test("RLM lab renders the exploration tree from the fixture", async ({ page }) => {
  await page.goto("/lab?rlmFixture=1");

  // Paper title is the single h1 inside the RlmLab root (data-testid="rlm-lab").
  await expect(page.getByTestId("rlm-lab").locator("h1").first()).toBeVisible();

  // Final rubric score 0.53 appears in the RubricStrip scoreValue span.
  // It also appears in the climb annotation "baseline 0.22 → 0.53";
  // use .first() to avoid strict-mode ambiguity.
  await expect(page.getByText(/0\.53/).first()).toBeVisible();

  // "learning-rate warmup" is the title of candidate c1 in the ExplorationCanvas.
  await expect(page.getByText("learning-rate warmup").first()).toBeVisible();

  // Click the node — ExplorationCanvas updates the right-docked detail sidebar.
  // The canvas auto-selects c7 "beam-size sweep" on load (it has outcome "running"),
  // so this click redirects selection to c1. We confirm the click landed by
  // asserting the sidebar's content shows "learning-rate warmup" (c1's title),
  // not just that the sidebar is visible (which it would be even without the click).
  await page.getByText("learning-rate warmup").first().click();
  await expect(page.getByTestId("node-detail-sidebar")).toBeVisible();
  await expect(
    page.getByTestId("node-detail-sidebar").getByText("learning-rate warmup")
  ).toBeVisible();
});
