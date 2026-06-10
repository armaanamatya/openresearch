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
  // Select the node through its real affordance: constellation nodes are SVG
  // <g role="button" aria-label="Select <title>"> (canvas labels truncate to
  // 12 chars, so text-matching the full title hits unrelated elements).
  await page.getByRole("button", { name: /select learning-rate warmup/i }).first().click();
  // :visible — Next.js streamed-Suspense leaves a hidden duplicate of the lab
  // in a <div hidden id="S:0"> staging container (no React fiber, zero box);
  // getByTestId strict mode matches it too (diagnosed audit 2026-06-10).
  const sidebar = page.locator('[data-testid="node-detail-sidebar"]:visible');
  await expect(sidebar).toBeVisible();
  // The sidebar defaults to a collapsed rail (Lane Z); selecting a node does
  // not auto-expand it — expand explicitly before asserting the detail body.
  const expand = sidebar.getByRole("button", { name: /expand node detail sidebar/i });
  if (await expand.isVisible().catch(() => false)) {
    await expand.click();
  }
  const openSidebar = page.locator('[data-testid="node-detail-sidebar"]:visible');
  await expect(openSidebar.getByText("learning-rate warmup")).toBeVisible();
});
