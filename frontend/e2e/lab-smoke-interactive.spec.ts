import { test, expect } from "@playwright/test";

// Interactive smoke for the Phase B polish features:
//   - Cmd-K command palette (cmdk)
//   - Canvas keyboard nav + `?` shortcut overlay
//   - Resizable split divider + localStorage persistence
//   - /library status filter pills
//   - /demo tour overlay next/back
//
// These tests need two seeded fake-runs on disk so the workflow view
// renders without firing a real reproduction. Seed them with:
//
//   tools/seed-fake-run.sh prj_diffusion_smoke
//   tools/seed-fake-run.sh prj_completed_smoke
//   sed -i '' 's/"status": "running"/"status": "completed"/' \
//     runs/prj_completed_smoke/demo_status.json
//
// Backend's live-pid check downgrades pid=1 "running" runs to "failed"
// when the process isn't alive — that's expected and doesn't change
// the workflow view's rendering for these specs.

// Use relative paths so playwright.config's `baseURL` (default
// http://localhost:3001) resolves them. Going to 127.0.0.1 directly
// trips Next.js 16's allowedDevOrigins safety check, which silently
// breaks dev-mode hydration even though SSR still renders.
const LAB_URL = "/lab";
const LIBRARY_URL = "/library";
const DEMO_URL = "/demo";

test("Cmd-K opens the command palette and fuzzy-search works", async ({ page }) => {
  await page.goto(LAB_URL);

  // Click into the body so the keyboard event has a focus target.
  // Without this, page.keyboard.press dispatches to nothing on a fresh
  // load (no element has focus), and the window-level "keydown" hook
  // in use-command-palette.ts never fires.
  await page.locator("body").click();

  // Open the palette via the global Cmd/Ctrl-K hook.
  await page.keyboard.press("ControlOrMeta+k");

  // cmdk@1's <Command label="..."> doesn't set role="dialog", so the
  // [cmdk-root] branch is what actually resolves. Keep the dialog role
  // first for forward-compat if cmdk ever adds it.
  const palette = page
    .getByRole("dialog", { name: /command palette/i })
    .or(page.locator("[cmdk-root]"));
  await expect(palette).toBeVisible({ timeout: 2000 });

  // <Command.Input> renders an <input cmdk-input>.
  const input = palette.locator("input[cmdk-input]").first();
  await input.fill("library");

  // The "Library" navigate item should match the fuzzy filter.
  const item = palette.getByText("Library", { exact: false }).first();
  await expect(item).toBeVisible();

  // Esc closes the palette (both the cmdk default and our defensive
  // document-level handler in command-palette.tsx).
  await page.keyboard.press("Escape");
  await expect(palette).toBeHidden({ timeout: 2000 });
});

test("j/k cycles canvas nodes; ? opens shortcut overlay; Esc closes", async ({ page }) => {
  await page.goto(`${LAB_URL}?projectId=prj_diffusion_smoke`);

  // Canvas should render for the seeded run.
  const canvas = page.locator(".canvas-surface").first();
  await expect(canvas).toBeVisible({ timeout: 5000 });

  // Click into the canvas area so focus is not on any button. The
  // keyboard-nav hook ignores presses originating from INPUT/TEXTAREA.
  await canvas.click({ position: { x: 50, y: 50 } });

  // Side panel starts with "Live activity" (no node selected).
  const sidePanelTitle = page.locator(".side-panel-title");
  await expect(sidePanelTitle).toHaveText("Live activity");

  // Press j → selects "src" → title becomes "Paper activity".
  await page.keyboard.press("j");
  await expect(sidePanelTitle).not.toHaveText("Live activity", { timeout: 2000 });

  // Press ? to open the shortcut overlay.
  await page.keyboard.press("?");
  const overlay = page.getByRole("dialog", { name: /keyboard shortcuts/i });
  await expect(overlay).toBeVisible({ timeout: 2000 });

  // Esc closes the overlay.
  await page.keyboard.press("Escape");
  await expect(overlay).toBeHidden({ timeout: 2000 });
});

test("resizable split divider drags and persists ratio in localStorage", async ({ page }) => {
  await page.goto(`${LAB_URL}?projectId=prj_diffusion_smoke`);

  const divider = page.locator(".resizable-split-divider").first();
  await expect(divider).toBeVisible({ timeout: 5000 });

  // Initial left-pane width.
  const leftPane = page.locator(".resizable-split-pane").first();
  const before = await leftPane.boundingBox();
  if (!before) throw new Error("no left-pane bbox before drag");

  // Drag the divider 200px to the left.
  const dividerBox = await divider.boundingBox();
  if (!dividerBox) throw new Error("no divider bbox");
  const startX = dividerBox.x + dividerBox.width / 2;
  const startY = dividerBox.y + dividerBox.height / 2;
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX - 200, startY, { steps: 12 });
  await page.mouse.up();

  await page.waitForTimeout(150); // let React re-render
  const after = await leftPane.boundingBox();
  if (!after) throw new Error("no left-pane bbox after drag");
  expect(after.width).toBeLessThan(before.width - 100);

  // localStorage should record the new ratio (key from resizable-split.tsx).
  const ratio = await page.evaluate(() =>
    window.localStorage.getItem("reprolab:split-ratio")
  );
  expect(ratio).not.toBeNull();
  expect(parseFloat(ratio ?? "1")).toBeLessThan(0.7);

  // Reload and confirm the ratio restores.
  await page.reload();
  await expect(divider).toBeVisible({ timeout: 5000 });
  const restored = await leftPane.boundingBox();
  if (!restored) throw new Error("no left-pane bbox after reload");
  expect(Math.abs(restored.width - after.width)).toBeLessThan(20);
});

test("library status filter narrows the table", async ({ page }) => {
  await page.goto(LIBRARY_URL);

  // The visible project-id text is truncated to 14 chars (library-table.tsx:179),
  // so match rows by their href attribute, which carries the full id.
  const diffusionRow = page.locator('tr:has(a[href*="prj_diffusion_smoke"])');
  const completedRow = page.locator('tr:has(a[href*="prj_completed_smoke"])');

  // Both seeded runs visible under "All".
  await expect(diffusionRow).toBeVisible({ timeout: 5000 });
  await expect(completedRow).toBeVisible();

  // Click the "Completed" status pill (role="tab").
  await page.getByRole("tab", { name: "Completed" }).click();
  await page.waitForTimeout(500); // debounced refetch

  // Only the completed run remains. (prj_diffusion_smoke is downgraded
  // to "failed" by the backend live-pid check, so the Completed filter
  // hides it — same intent: the filter narrows the table.)
  await expect(completedRow).toBeVisible();
  await expect(diffusionRow).toBeHidden();
});

test("demo tour overlay steps forward and back", async ({ page }) => {
  await page.goto(DEMO_URL);

  // The overlay renders "Step 1 of N" where N = the count of nodes with
  // a tour_caption in backend/agents/topology.py. As of D.5.1 that is 6
  // (read, env, impl, opt, audit, report). If a topology change adds or
  // removes a tour caption, update the count below to match.
  await expect(page.getByText(/step 1 of 6/i)).toBeVisible({ timeout: 5000 });

  // Next twice.
  await page.getByRole("button", { name: /^next$/i }).click();
  await expect(page.getByText(/step 2 of 6/i)).toBeVisible();
  await page.getByRole("button", { name: /^next$/i }).click();
  await expect(page.getByText(/step 3 of 6/i)).toBeVisible();

  // Back returns to step 2.
  await page.getByRole("button", { name: /^back$/i }).click();
  await expect(page.getByText(/step 2 of 6/i)).toBeVisible();

  // The dismiss control has aria-label="Dismiss tour" (demo-overlay.tsx:86).
  await page.getByRole("button", { name: /dismiss/i }).click();
  await expect(page.getByText(/step 2 of 6/i)).toBeHidden();
});
