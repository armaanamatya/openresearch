import * as http from "node:http";
import type { AddressInfo } from "node:net";
import { test, expect } from "@playwright/test";

/**
 * Task 14 — end-to-end coverage for the autonomous upload/arXiv-entry flow
 * (alphaXiv screens `_3`/`_4`/D/`_5`): landing → repo-confirm → cost-guard
 * → a MOCKED backend launch + SSE stream → the spec-validation stepper
 * advancing through its stages → the swap to the live reasoning view.
 *
 * No real backend/run is involved. Two things are mocked:
 *
 *   - POST /api/demo/arxiv — intercepted directly via `page.route()` (a
 *     single-shot request/response, no streaming concerns). We assert its
 *     JSON body carries `autonomous: true` and the edited `repo_url`.
 *
 *   - GET /api/demo/events — the SessionPage's live EventSource. This is
 *     NOT fulfilled directly. `useRun`'s `source.onerror` handler (see
 *     frontend/src/hooks/use-run.ts) explicitly calls `source.close()` and
 *     falls back to REST polling on *any* disconnect — and per the
 *     EventSource spec, `error` fires before *every* reconnect, including a
 *     clean one. So a naive "one `route.fulfill()` per stage, rely on
 *     EventSource auto-reconnect between them" scheme cannot work here: the
 *     very first clean close permanently kills the SSE path in this app
 *     (verified empirically while writing this test). Instead, `page.route`
 *     redirects the request to a real local `http` server (spun up below)
 *     that holds ONE connection open for the whole test and paces genuine
 *     `res.write()` calls in real time — an actual streaming response, the
 *     same shape a live backend would produce, so the connection never
 *     closes and `onerror` never fires. The redirect target is a different
 *     origin (127.0.0.1:<port> vs. the app's localhost:3001), so the local
 *     server sends `Access-Control-Allow-Origin` for the EventSource's CORS
 *     fetch to succeed.
 *
 *   Every RLM-domain event (repl_iteration, primitive_call, spec_*) is
 *   wrapped inside an OUTER SSE frame literally named "dashboard_event"
 *   (see backend/services/events/live_runs.py's `_read_dashboard_events` +
 *   `sse_event("dashboard_event", dash_event, ...)`); `useRun` listens for
 *   that outer event name and JSON-parses the inner RLM event out of
 *   `data`. Getting this wrapping right is what makes the mock actually
 *   reach `dashboardEvents`.
 *
 * GET /papers/{arxivId}/repo (T3b) is NOT mocked: `/abs/[arxivId]/page.tsx`
 * fetches it server-side (inside the Next.js Node process during SSR),
 * which is not reachable from a browser-context `page.route()` intercept.
 * Its outcome therefore depends on the environment: with no backend it
 * fails soft to "" (blank RepoField), while CI's e2e job runs a real
 * backend that resolves this paper's official repo and pre-fills the
 * field. The test is agnostic to which happened — it asserts the field is
 * editable, then overwrites whatever is there (fill() replaces the whole
 * value) to strengthen the posted-body assertion.
 */

const ARXIV_ID = "2605.15155";
const RUN_ID = "prj_autonomous_e2e_test";
const REPO_URL = "octocat/hello-world";
const CORPUS_MARKER = "ZZZ_CORPUS_LEAK_MARKER_9c21f7";
const REASONING_TEXT = "Beginning paper comprehension for the autonomous run.";
const LEAF_COUNT = 12;
const VALIDATOR_MODEL = "grok-4.3";

// Real wall-clock milliseconds between each staged event write. Generous
// relative to Playwright's default 30s expect timeout (total stream time is
// ~6 * 350ms ≈ 2.1s), but slow enough that React has genuine yield points
// to paint each intermediate stepper stage rather than batching them away.
const STAGE_DELAY_MS = 350;

function sseFrame(payload: Record<string, unknown>): string {
  return `event: dashboard_event\ndata: ${JSON.stringify(payload)}\n\n`;
}

// The 6 mocked events, in the exact order the task calls for:
// spec_generation_started -> spec_generated -> spec_validation_started ->
// spec_validated -> repl_iteration -> primitive_call. repl_iteration and
// primitive_call are two distinct stages so the reasoning text is provably
// visible before its tool chip lands, mirroring how the real root loop
// streams them.
const SSE_STAGES: Array<Record<string, unknown>> = [
  { event: "spec_generation_started", timestamp: "2026-07-06T00:00:00Z" },
  { event: "spec_generated", timestamp: "2026-07-06T00:00:01Z", leaf_count: LEAF_COUNT },
  {
    event: "spec_validation_started",
    timestamp: "2026-07-06T00:00:02Z",
    validator_model: VALIDATOR_MODEL,
  },
  {
    event: "spec_validated",
    timestamp: "2026-07-06T00:00:03Z",
    verdict: "clean",
    flagged_leaves: [],
  },
  {
    event: "repl_iteration",
    timestamp: "2026-07-06T00:00:04Z",
    iteration: 1,
    response: REASONING_TEXT,
    code_blocks: [],
    sub_calls: 0,
    timing: 1.2,
    // Raw fields a hypothetical backend regression might let ride along.
    // IterationView only ever destructures {iteration, response,
    // code_blocks, sub_calls, timing} (see SessionReasoningView.tsx's
    // corpus-safe-helpers comment) — nothing here reads `context`/`prompt`,
    // so these must never reach the DOM no matter what the mock includes.
    context: CORPUS_MARKER,
    prompt: CORPUS_MARKER,
  },
  {
    event: "primitive_call",
    timestamp: "2026-07-06T00:00:05Z",
    primitive: "understand_section",
    status: "ok",
    args_summary: {},
    result_summary: null,
    iteration: 1,
    rubric_delta: null,
  },
];

/** Starts a real local SSE server: one open connection per request, each
 * writing the full staged event sequence in real time and then staying open
 * (a live backend never closes an idle stream) so the browser's
 * EventSource never sees a disconnect. Returns the server + its base URL. */
async function startFakeSseServer(): Promise<{ server: http.Server; url: string }> {
  const server = http.createServer((req, res) => {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "access-control-allow-origin": "*",
    });
    let i = 0;
    const pushNext = () => {
      if (res.writableEnded) return;
      if (i < SSE_STAGES.length) {
        res.write(sseFrame(SSE_STAGES[i]));
        i += 1;
        timer = setTimeout(pushNext, STAGE_DELAY_MS);
      }
      // Once all stages are sent the stream is simply left open — the
      // real backend does the same between events.
    };
    let timer: ReturnType<typeof setTimeout> = setTimeout(pushNext, 0);
    req.on("close", () => clearTimeout(timer));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { server, url: `http://127.0.0.1:${port}/` };
}

test("autonomous launch: landing -> repo-confirm -> cost-guard -> mocked run -> spec stepper -> reasoning view, no corpus leak", async ({
  page,
}) => {
  const { server: sseServer, url: sseServerUrl } = await startFakeSseServer();

  try {
    let arxivRequestBody: Record<string, unknown> | null = null;
    await page.route("**/api/demo/arxiv", async (route) => {
      arxivRequestBody = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          projectId: RUN_ID,
          outputDir: `runs/${RUN_ID}`,
          runMode: "rlm",
          // "completed" (not "queued") is deliberate: it keeps RepoConfirm's
          // own transient useRun() instance from also opening an EventSource
          // to the fake SSE server right before router.push unmounts it —
          // one fewer incidental connection. Harmless either way (each
          // connection to the fake server independently replays the full
          // stage sequence from the top), but SessionPage never reads this
          // response at all: it hardcodes its own status:"queued"
          // initialRun regardless (mirrors `okRunStateResponse` in
          // RepoConfirm.test.tsx).
          status: "completed",
          payload: null,
          log: "",
        }),
      });
    });

    await page.route("**/api/demo/events**", async (route) => {
      await route.fulfill({
        status: 302,
        headers: {
          location: sseServerUrl,
          "access-control-allow-origin": "*",
        },
      });
    });

    // ── 1. Landing (alphaXiv `_3`) ────────────────────────────────────────
    await page.goto(`/abs/${ARXIV_ID}`);
    await expect(page.getByText(/welcome to openresearch/i)).toBeVisible();
    await expect(page.getByText(new RegExp(`arxiv ${ARXIV_ID}`, "i"))).toBeVisible();
    const signInBtn = page.getByRole("button", { name: /sign in to start/i });
    await expect(signInBtn).toBeVisible();

    // ── 2. Sign in -> repo-confirm (`?confirm=1`) ─────────────────────────
    await signInBtn.click();
    await page.waitForURL(/confirm=1/);
    await expect(page.getByRole("heading", { level: 1 })).toContainText(/reproduce this paper/i);

    // ── 3. RepoField renders + is editable. Whether it starts blank (no
    // backend → server-side resolve fails soft to "") or pre-filled (CI's
    // real backend resolves this paper's repo) is environment-dependent —
    // see the file header note — so overwrite it either way. ───────────────
    const repoField = page.getByLabel("GitHub repository");
    await expect(repoField).toBeVisible();
    await expect(repoField).toBeEditable();
    await repoField.fill(REPO_URL);
    await expect(repoField).toHaveValue(REPO_URL);

    // ── 4. Cost-guard confirm dialog (D4) ─────────────────────────────────
    await page.getByRole("button", { name: /start autoresearch/i }).click();
    const dialog = page.getByRole("dialog", { name: /confirm autoresearch launch/i });
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText("$10");

    // ── 5. Confirm & launch -> POST /api/demo/arxiv with autonomous:true ──
    await dialog.getByRole("button", { name: /confirm.*launch/i }).click();
    await page.waitForURL(new RegExp(`/sessions/${RUN_ID}`));

    expect(arxivRequestBody).toBeTruthy();
    expect(arxivRequestBody!.autonomous).toBe(true);
    expect(arxivRequestBody!.repo_url).toBe(REPO_URL);
    expect(arxivRequestBody!.url).toBe(`https://arxiv.org/abs/${ARXIV_ID}`);

    // ── 6. Spec-validation stepper (screen D) advances through its stages ─
    await expect(page.getByText(/generating reproduction spec/i)).toBeVisible();
    await expect(
      page.getByText(new RegExp(`generating reproduction spec.*${LEAF_COUNT} leaves`, "i"))
    ).toBeVisible();
    await expect(
      page.getByText(new RegExp(`validating spec with ${VALIDATOR_MODEL}`, "i"))
    ).toBeVisible();

    // ── 7. spec_validated flips the route from the stepper to the live
    // reasoning view (`_5`); the repl_iteration/primitive_call render the
    // reasoning text + a tool-call ReasoningChip ───────────────────────────
    await expect(page.getByTestId("session-reasoning-view")).toBeVisible();
    await expect(page.getByText(REASONING_TEXT)).toBeVisible();
    await expect(page.getByText("Understand section")).toBeVisible();

    // ── 8. Hard corpus assertion — the raw context/prompt fields planted on
    // the repl_iteration mock event must never reach the rendered DOM or the
    // underlying HTML (attribute-based leaks would show up in page.content()
    // but not in body textContent, so both are checked). ──────────────────
    await expect(page.locator("body")).not.toContainText(CORPUS_MARKER);
    expect(await page.content()).not.toContain(CORPUS_MARKER);
  } finally {
    sseServer.closeAllConnections();
    await new Promise<void>((resolve) => sseServer.close(() => resolve()));
  }
});
