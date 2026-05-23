import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 720_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.LAB_BASE_URL ?? "http://localhost:3001",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    actionTimeout: 15_000,
    navigationTimeout: 60_000
  },
  // Always start the *current worktree's* Next dev server. Never reuse a
  // server already on :3001 — in multi-worktree setups that server is
  // almost certainly a different branch's code, and the e2e tests would
  // silently validate the wrong build. (The Phase 6 cutover e2e once
  // chased a phantom "double <h1>" coming from another worktree's UI.)
  // Honors LAB_BASE_URL as an escape hatch when pointing at an external
  // staging server.
  webServer: process.env.LAB_BASE_URL
    ? undefined
    : {
        // E2E runs against the production build, not `next dev`. Dev mode
        // has rendering artifacts (extra mount cycles, HMR state) that
        // produce spurious failures unrelated to production behaviour —
        // e.g. a duplicated <h1> in the RLM lab tree that disappears once
        // built. Building takes ~30–60 s; the long timeout reflects that.
        command: "npm run build && npx next start -p 3001",
        url: "http://localhost:3001",
        reuseExistingServer: false,
        timeout: 240_000,
        env: { REPROLAB_BACKEND_URL: "http://127.0.0.1:8000" }
      },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], headless: true }
    }
  ]
});
