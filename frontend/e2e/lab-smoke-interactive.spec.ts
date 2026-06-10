import { test, expect } from "@playwright/test";

// Interactive smoke for the RLM lab UI:
//   - Cmd-K command palette (cmdk)
//   - `?` shortcut overlay (uses ?rlmFixture=1, no seeded run needed)
//   - /library status filter pills
//
// The library filter test needs two seeded fake-runs on disk so the
// library table renders real rows. Seed them with:
//
//   scripts/seed-fake-run.sh prj_diffusion_smoke
//   scripts/seed-fake-run.sh prj_completed_smoke
//   sed -i '' 's/"status": "running"/"status": "completed"/' \
//     runs/prj_completed_smoke/demo_status.json
//
// Backend's live-pid check downgrades pid=1 "running" runs to "failed"
// when the process isn't alive — that's expected and doesn't change
// the library table's rendering for these specs.

// Use relative paths so playwright.config's `baseURL` (default
// http://localhost:3001) resolves them. Going to 127.0.0.1 directly
// trips Next.js 16's allowedDevOrigins safety check, which silently
// breaks dev-mode hydration even though SSR still renders.
const LAB_URL = "/lab";
const LIBRARY_URL = "/library";

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

test("? opens shortcut overlay; Esc closes", async ({ page }) => {
  // Use the rlmFixture path so no seeded backend run is required.
  await page.goto(`${LAB_URL}?rlmFixture=1`);

  // Click into the body so the window-level keydown handler in
  // use-shortcut-overlay.ts has a non-input focus target (same pattern
  // as the Cmd-K test above).
  await page.locator("body").click();

  // Press ? to open the shortcut overlay.
  await page.keyboard.press("?");
  const overlay = page.getByRole("dialog", { name: /keyboard shortcuts/i });
  await expect(overlay).toBeVisible({ timeout: 2000 });

  // Esc closes the overlay.
  await page.keyboard.press("Escape");
  await expect(overlay).toBeHidden({ timeout: 2000 });
});


test("library status filter narrows the table", async ({ page }) => {
  // Seeded by scripts/seed_fake_run.py; skip cleanly when unseeded (the
  // dead-backend early-return below covers no-backend, but a LIVE backend
  // without seeds used to fail here).
  const fs = await import("node:fs");
  const path = await import("node:path");
  test.skip(
    !fs.existsSync(path.join(__dirname, "../../runs/prj_diffusion_smoke")),
    "run `python scripts/seed_fake_run.py` first",
  );
  await page.goto(LIBRARY_URL);
  const unavailable = page.getByText(/run library unavailable/i);
  if (await unavailable.isVisible().catch(() => false)) {
    await expect(unavailable).toBeVisible();
    return;
  }

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
