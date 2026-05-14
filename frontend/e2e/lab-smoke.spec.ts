import { test, expect, type Page } from "@playwright/test";
import path from "node:path";
import http from "node:http";

const BACKEND_BASE = process.env.LAB_BACKEND_URL ?? "http://127.0.0.1:8001";
const PROXY_BASE = process.env.LAB_BASE_URL ?? "http://localhost:3001";
const PDF_PATH = path.resolve(__dirname, "..", "..", "demo_paper.pdf");

const NODE_AGENTS = [
  "Paper",
  "Reader",
  "Forge",
  "Architect",
  "Builder",
  "Vesta",
  "Athena",
  "Orion",
  "Lyra",
  "Pyxis",
  "Hermes",
  "Scribe"
] as const;

// pipeline_state.json is only written at gate_1/2/3/complete (orchestrator.py:1030,1167,1391,1456),
// not on every stage transition. So `counter > 1` requires the run to reach gate_1_passed,
// which can take 10-15+ min of wall time with real Anthropic LLM calls.
const PIPELINE_PROGRESS_TIMEOUT_MS = 14 * 60 * 1_000;

async function startFixtureRun(page: Page) {
  const arxivInput = page.getByPlaceholder("arxiv.org/abs/2303.04137");
  await expect(arxivInput).toBeVisible();
  await arxivInput.fill("arxiv.org/abs/0000.00000");
  const beginBtn = page.getByRole("button", { name: /begin/i });
  await expect(beginBtn).toBeEnabled();
  await beginBtn.click();
}

async function uploadPdfRun(page: Page, pdfPath: string) {
  const fileInput = page.locator('input[type="file"][aria-label="Upload paper PDF"]');
  await fileInput.setInputFiles(pdfPath);
}

async function getProjectId(page: Page): Promise<string> {
  const eyebrow = page.locator(".workflow-header .eyebrow");
  await expect(eyebrow).toBeVisible({ timeout: 30_000 });
  const text = (await eyebrow.textContent()) ?? "";
  const match = text.match(/workflow\s*-\s*(\S+)/i);
  if (!match) throw new Error(`could not extract projectId from header: ${JSON.stringify(text)}`);
  return match[1];
}

async function getDoneCount(page: Page): Promise<number> {
  const counter = page.locator(".workflow-meta .mono").first();
  await expect(counter).toBeVisible();
  const text = (await counter.textContent()) ?? "";
  const m = text.match(/(\d+)\/(\d+) agents complete/);
  if (!m) throw new Error(`counter text not parsed: ${text}`);
  return Number(m[1]);
}

function nodeCard(page: Page, agent: string) {
  return page.locator(`div[data-node="1"]:has(.node-agent:text-is("${agent}"))`).first();
}

async function clickNode(page: Page, agent: string) {
  const card = nodeCard(page, agent);
  await expect(card).toBeVisible();
  await card.click({ force: true });
}

async function fetchSse(url: string, ms: number): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    let buf = "";
    const req = http.get(url, { headers: { Accept: "text/event-stream" } }, (res) => {
      res.setEncoding("utf-8");
      res.on("data", (chunk: string) => {
        buf += chunk;
      });
      res.on("error", reject);
    });
    req.on("error", reject);
    setTimeout(() => {
      req.destroy();
      resolve(buf);
    }, ms);
  });
}

const consoleErrors: string[] = [];

function attachConsoleListener(page: Page) {
  page.on("pageerror", (err) => {
    consoleErrors.push(`pageerror: ${err.message}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(`console.error: ${msg.text()}`);
    }
  });
}

// ---------------------------------------------------------------------------
// Sections A (fast structural checks), D, C, E share one fixture run.
// Section A2 separately exercises the slow "counter advances past 1" check.
// Section B has its own upload run.
// ---------------------------------------------------------------------------

test.describe.configure({ mode: "serial" });

test.describe("Lab pipeline — fixture run shared session", () => {
  let page: Page;
  let projectId = "";

  test.beforeAll(async ({ browser }) => {
    consoleErrors.length = 0;
    page = await browser.newPage();
    attachConsoleListener(page);
    await page.goto("/lab");
    await startFixtureRun(page);
    await expect(page.locator(".workflow-header")).toBeVisible({ timeout: 30_000 });
    projectId = await getProjectId(page);
  });

  test.afterAll(async () => {
    await page?.close();
  });

  test("A1. workflow structure — 12 nodes, src done, read running, rail populates, gate-1 chip", async () => {
    // 12 node cards rendered
    await expect(page.locator('div[data-node="1"]')).toHaveCount(12, { timeout: 15_000 });
    for (const agent of NODE_AGENTS) {
      await expect(nodeCard(page, agent)).toBeVisible();
    }

    // src done immediately
    await expect(nodeCard(page, "Paper").locator(".node-check")).toBeVisible({ timeout: 30_000 });

    // Reader transitions to running OR done within 60s (pipeline status: running)
    const reader = nodeCard(page, "Reader");
    await expect
      .poll(
        async () =>
          (await reader.locator(".node-check").count()) > 0 ||
          (await reader.locator(".wf-ring").count()) > 0,
        { timeout: 60_000 }
      )
      .toBe(true);

    // counter ≥ 1
    await expect.poll(() => getDoneCount(page), { timeout: 30_000 }).toBeGreaterThanOrEqual(1);

    // right rail "Live agents" populates within 30s
    await expect(page.getByText(/^Live agents$/)).toBeVisible();
    await expect
      .poll(() => page.locator(".timeline-agents .timeline-agent").count(), { timeout: 60_000 })
      .toBeGreaterThan(0);

    // gate-1 chip rendered (any state)
    const gate1Chip = page.locator(".gate-chip").first();
    await expect(gate1Chip).toBeVisible({ timeout: 15_000 });
    test.info().annotations.push({
      type: "A1.gate1.classes",
      description: (await gate1Chip.getAttribute("class")) ?? "(none)"
    });

    // No uncaught browser exceptions
    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });

  test("D. per-node panels — clickable nodes open structural content (others are upcoming)", async () => {
    const tested: string[] = [];
    const skippedUpcoming: string[] = [];

    for (const agent of NODE_AGENTS) {
      await test.step(`open ${agent}`, async () => {
        const card = nodeCard(page, agent);
        await expect(card).toBeVisible();
        // upcoming nodes have opacity 0.4 and no-op click
        const opacity = await card.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
        if (opacity <= 0.5) {
          skippedUpcoming.push(agent);
          return;
        }
        await card.click({ force: true });
        await expect(page.locator(".agent-name", { hasText: agent }).first()).toBeVisible({
          timeout: 10_000
        });
        await expect(page.locator(".agent-task").first()).toBeVisible();
        tested.push(agent);
      });
    }

    test.info().annotations.push({
      type: "D.tested_nodes",
      description: tested.join(", ") || "(none)"
    });
    test.info().annotations.push({
      type: "D.skipped_upcoming",
      description: skippedUpcoming.join(", ") || "(none)"
    });

    // Reader panel multi-line tail (spec: ≥ 2 lines)
    if (tested.includes("Reader")) {
      await clickNode(page, "Reader");
      const readerLogItems = await page.locator(".agent-log-list .agent-log-item").count();
      test.info().annotations.push({
        type: "D.reader.log_lines",
        description: String(readerLogItems)
      });
    }

    // We require at least Paper (src) and Reader (read) to be clickable in early pipeline.
    expect(tested).toEqual(expect.arrayContaining(["Paper", "Reader"]));

    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });

  test("C. navigation — brand resets, Library/Hermes nav, return to lab", async () => {
    // Click brand → upload screen
    await page.locator(".brand-row").click();
    await expect(page.locator(".upload-zone")).toBeVisible({ timeout: 10_000 });

    // Library nav
    await page.getByRole("link", { name: /library/i }).click();
    await page.waitForURL(/\/papers/, { timeout: 15_000 });

    // Hermes nav
    await page.goto("/lab");
    await page.getByRole("link", { name: /hermes/i }).click();
    await page.waitForURL(/\/hermes/, { timeout: 15_000 });

    // Return to Lab — repro-lab-client.tsx does NOT auto-restore from FastAPI;
    // it always remounts to the upload view. Document this explicitly.
    await page.goto("/lab");
    const upload = page.locator(".upload-zone");
    const workflow = page.locator(".workflow-header");
    const which = await Promise.race([
      upload
        .waitFor({ state: "visible", timeout: 8_000 })
        .then(() => "upload")
        .catch(() => null),
      workflow
        .waitFor({ state: "visible", timeout: 8_000 })
        .then(() => "workflow")
        .catch(() => null)
    ]);
    test.info().annotations.push({
      type: "C.return_to_lab",
      description: `view after return: ${which ?? "neither"}`
    });
    expect(which).not.toBeNull();
    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });

  test("E. SSE — backend events + proxy enrichment (live run)", async () => {
    expect(projectId).toMatch(/^(ui_sdk|prj_)/);

    const backend = await fetchSse(`${BACKEND_BASE}/runs/${projectId}/events`, 8_000);
    expect(backend, "backend SSE stream produced no data").not.toEqual("");

    const backendEventTypes = Array.from(backend.matchAll(/^event:\s*(\S+)/gm)).map((m) => m[1]);
    test.info().annotations.push({
      type: "E.backend_events",
      description: backendEventTypes.slice(0, 30).join(", ")
    });
    expect(backendEventTypes).toContain("run_state");
    // dashboard_event / agent_log / heartbeat — at least one streaming event type
    expect(
      backendEventTypes.some((t) => ["dashboard_event", "agent_log", "heartbeat"].includes(t))
    ).toBe(true);

    const proxied = await fetchSse(
      `${PROXY_BASE}/api/demo/events?projectId=${encodeURIComponent(projectId)}`,
      8_000
    );
    expect(proxied, "proxy SSE stream produced no data").not.toEqual("");

    const proxyEventTypes = Array.from(proxied.matchAll(/^event:\s*(\S+)/gm)).map((m) => m[1]);
    test.info().annotations.push({
      type: "E.proxy_events",
      description: proxyEventTypes.slice(0, 30).join(", ")
    });
    expect(proxyEventTypes).toContain("run_state");

    // Synth-N frames are emitted only when stableEnrichedHash changes between
    // emits. For a run still in paper-understanding (no pipeline_state.json yet)
    // OR a static completed run, the underlying state doesn't change → no synth.
    // We record this as info; the deterministic check uses prj_1621776362bfa518 below.
    const synthIds = Array.from(proxied.matchAll(/^id:\s*(synth-\d+)/gm)).map((m) => m[1]);
    test.info().annotations.push({
      type: "E.synth_ids_live_run",
      description:
        synthIds.length > 0
          ? synthIds.slice(0, 10).join(", ")
          : "(none — expected when pipeline_state.json hasn't yet written or run is static)"
    });
  });

  test("E2. payload.summary.stage populated for post-gate-1 run (deterministic)", async () => {
    // prj_1621776362bfa518 is a known checkpointed run that reached gate_1_passed
    // (verified via runs/<id>/pipeline_state.json on disk). The bridge fix
    // (server-payload.ts → enrichRunStateWithPayload) should populate
    // payload.summary.stage from pipeline_state.json. Pre-fix this was null.
    const KNOWN_RUN = "prj_1621776362bfa518";
    const proxied = await fetchSse(
      `${PROXY_BASE}/api/demo/events?projectId=${encodeURIComponent(KNOWN_RUN)}`,
      6_000
    );
    expect(proxied, "proxy SSE produced no data for known run").not.toEqual("");

    const dataLines = proxied
      .split(/\n\n/g)
      .filter((b) => /^event:\s*run_state/m.test(b))
      .map((b) => {
        const m = b.match(/^data:\s*(.*)$/m);
        return m ? m[1] : "";
      })
      .filter(Boolean);

    let foundStage: string | null = null;
    for (const json of dataLines) {
      try {
        const parsed = JSON.parse(json);
        const stage = parsed?.payload?.summary?.stage;
        if (typeof stage === "string" && stage.length > 0) {
          foundStage = stage;
          break;
        }
      } catch {
        // ignore
      }
    }
    test.info().annotations.push({
      type: "E2.payload_stage",
      description: foundStage ?? "(none)"
    });
    expect(foundStage, "expected payload.summary.stage from bridge fix").toBe("gate_1_passed");
  });

  test("A2. SLOW — pipeline counter advances past 1 (read becomes done)", async () => {
    test.setTimeout(PIPELINE_PROGRESS_TIMEOUT_MS + 60_000);
    await expect
      .poll(() => getDoneCount(page), {
        timeout: PIPELINE_PROGRESS_TIMEOUT_MS,
        intervals: [3000]
      })
      .toBeGreaterThan(1);
    test.info().annotations.push({
      type: "A2.final_count",
      description: String(await getDoneCount(page))
    });
  });

  test("D2. SLOW — re-test previously upcoming nodes after pipeline advance", async () => {
    test.setTimeout(120_000);
    const tested: string[] = [];
    const stillUpcoming: string[] = [];
    for (const agent of NODE_AGENTS) {
      const card = nodeCard(page, agent);
      const opacity = await card.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
      if (opacity <= 0.5) {
        stillUpcoming.push(agent);
        continue;
      }
      await card.click({ force: true });
      const ok = await page
        .locator(".agent-name", { hasText: agent })
        .first()
        .waitFor({ state: "visible", timeout: 5_000 })
        .then(() => true)
        .catch(() => false);
      if (ok) tested.push(agent);
    }
    test.info().annotations.push({
      type: "D2.tested_nodes",
      description: tested.join(", ") || "(none)"
    });
    test.info().annotations.push({
      type: "D2.still_upcoming",
      description: stillUpcoming.join(", ") || "(none)"
    });

    // Test HermesAuditPanel and ScriptPanel structural rendering if their nodes are clickable
    if (tested.includes("Hermes")) {
      await clickNode(page, "Hermes");
      const eyebrow = page
        .locator(".agent-section .eyebrow", { hasText: /hermes audit( timeline)?/i })
        .first();
      const ok = await eyebrow
        .waitFor({ state: "visible", timeout: 5_000 })
        .then(() => true)
        .catch(() => false);
      test.info().annotations.push({
        type: "D2.hermes_audit_panel_visible",
        description: String(ok)
      });
    }
    if (tested.includes("Scribe")) {
      await clickNode(page, "Scribe");
      const ok = await page
        .locator(".script-panel")
        .waitFor({ state: "visible", timeout: 5_000 })
        .then(() => true)
        .catch(() => false);
      test.info().annotations.push({
        type: "D2.script_panel_visible",
        description: String(ok)
      });
    }

    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Section B — the PDF upload flow ("the one that used to not work").
// Mirrors A1 + D + E shape but specifically against the upload path so we
// validate the actual user journey end-to-end.
// ---------------------------------------------------------------------------

test.describe("Section B — PDF upload end-to-end", () => {
  test.describe.configure({ mode: "serial" });

  let page: Page;
  let projectId = "";

  test.beforeAll(async ({ browser }) => {
    consoleErrors.length = 0;
    page = await browser.newPage();
    attachConsoleListener(page);
    await page.goto("/lab");
    await uploadPdfRun(page, PDF_PATH);
    await expect(page.locator(".workflow-header")).toBeVisible({ timeout: 60_000 });
    projectId = await getProjectId(page);
  });

  test.afterAll(async () => {
    await page?.close();
  });

  test("B1. upload accepted — workflow view, 12 nodes, projectId issued", async () => {
    expect(projectId).toMatch(/^(prj_|ui_sdk)/);
    test.info().annotations.push({ type: "B1.projectId", description: projectId });

    await expect(page.locator('div[data-node="1"]')).toHaveCount(12, { timeout: 15_000 });
    for (const agent of NODE_AGENTS) {
      await expect(nodeCard(page, agent)).toBeVisible();
    }

    // Page header reflects uploaded source (sourceLabel pulled from the PDF, not the fixture)
    const headerTitle = await page.locator(".workflow-title").textContent();
    test.info().annotations.push({
      type: "B1.workflow_title",
      description: headerTitle?.slice(0, 120) ?? "(none)"
    });

    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });

  test("B2. workflow nodes animate — src done, Reader running, gate-1 chip, rail populates", async () => {
    // src done immediately
    await expect(nodeCard(page, "Paper").locator(".node-check")).toBeVisible({ timeout: 30_000 });

    // Reader transitions to running OR done within 60s
    const reader = nodeCard(page, "Reader");
    await expect
      .poll(
        async () =>
          (await reader.locator(".node-check").count()) > 0 ||
          (await reader.locator(".wf-ring").count()) > 0,
        { timeout: 60_000 }
      )
      .toBe(true);

    // Counter is ≥ 1
    await expect.poll(() => getDoneCount(page), { timeout: 30_000 }).toBeGreaterThanOrEqual(1);

    // Right rail "Live agents" populates within 60s
    await expect(page.getByText(/^Live agents$/)).toBeVisible();
    await expect
      .poll(() => page.locator(".timeline-agents .timeline-agent").count(), { timeout: 60_000 })
      .toBeGreaterThan(0);

    // Reasoning rail header exists
    await expect(page.getByText(/^Reasoning$/)).toBeVisible();

    // gate-1 chip visible (state may be pending/checking — depends on stage)
    const gate1Chip = page.locator(".gate-chip").first();
    await expect(gate1Chip).toBeVisible({ timeout: 15_000 });
    test.info().annotations.push({
      type: "B2.gate1_class",
      description: (await gate1Chip.getAttribute("class")) ?? "(none)"
    });

    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });

  test("B3. ScriptPanel renders source PDF + Preview/Download/Benchmark", async () => {
    await clickNode(page, "Scribe");
    await expect(page.locator(".script-panel")).toBeVisible({ timeout: 10_000 });

    const previewBtn = page.locator(".script-panel .pdf-actions a", { hasText: /preview pdf/i });
    const downloadBtn = page.locator(".script-panel .pdf-actions a", { hasText: /^download$/i });
    await expect(previewBtn).toBeVisible();
    await expect(downloadBtn).toBeVisible();

    // The preview/download href should resolve to a populated source-pdf URL,
    // not "/api/demo/source-pdf?projectId=" with no value.
    const previewHref = await previewBtn.getAttribute("href");
    const downloadHref = await downloadBtn.getAttribute("href");
    test.info().annotations.push({
      type: "B3.preview_href",
      description: previewHref ?? "(none)"
    });
    test.info().annotations.push({
      type: "B3.download_href",
      description: downloadHref ?? "(none)"
    });
    expect(previewHref).toContain(projectId);
    expect(downloadHref).toContain(projectId);

    // Source PDF actually serves bytes through the proxy
    const pdfResp = await page.request.get(previewHref!);
    expect(pdfResp.status()).toBe(200);
    const ctype = pdfResp.headers()["content-type"] ?? "";
    test.info().annotations.push({ type: "B3.pdf_content_type", description: ctype });
    expect(ctype).toMatch(/application\/pdf/);

    // Benchmark card and final-report link present
    await expect(page.locator(".benchmark-card")).toBeVisible();
    await expect(page.locator(".final-report-link")).toBeVisible();

    // PDF metadata row shows non-zero size
    const pdfMeta = await page.locator(".pdf-meta").textContent();
    test.info().annotations.push({ type: "B3.pdf_meta", description: pdfMeta ?? "(none)" });
    expect(pdfMeta).toMatch(/\d+(\.\d+)?\s*(KB|MB|bytes)/i);

    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });

  test("B4. per-node panels — clickable nodes open content (early pipeline)", async () => {
    const tested: string[] = [];
    const skipped: string[] = [];

    for (const agent of NODE_AGENTS) {
      const card = nodeCard(page, agent);
      await expect(card).toBeVisible();
      const opacity = await card.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
      if (opacity <= 0.5) {
        skipped.push(agent);
        continue;
      }
      await card.click({ force: true });
      const ok = await page
        .locator(".agent-name", { hasText: agent })
        .first()
        .waitFor({ state: "visible", timeout: 8_000 })
        .then(() => true)
        .catch(() => false);
      if (ok) tested.push(agent);
    }

    test.info().annotations.push({
      type: "B4.tested_nodes",
      description: tested.join(", ") || "(none)"
    });
    test.info().annotations.push({
      type: "B4.upcoming_nodes",
      description: skipped.join(", ") || "(none)"
    });
    expect(tested).toEqual(expect.arrayContaining(["Paper", "Reader"]));

    // Reader panel: log entries (multi-line tail expected by spec)
    await clickNode(page, "Reader");
    const readerLogItems = await page.locator(".agent-log-list .agent-log-item").count();
    test.info().annotations.push({
      type: "B4.reader_log_items",
      description: String(readerLogItems)
    });

    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });

  test("B5. backend runner.stderr.log shows pipeline stages advancing", async () => {
    // Read the runner.stderr.log via the project's path on disk to mirror what
    // the smoke prompt asks the operator to do with `tail -f`.
    const fs = await import("node:fs/promises");
    const runDir = path.resolve(__dirname, "..", "..", "runs", projectId);
    const logPath = path.join(runDir, "runner.stderr.log");
    const log = await fs.readFile(logPath, "utf-8").catch(() => "");
    const stageMarkers = Array.from(log.matchAll(/Starting:\s*(\S+)/g)).map((m) => m[1]);
    test.info().annotations.push({
      type: "B5.stage_markers",
      description: stageMarkers.slice(0, 8).join(", ") || "(none)"
    });
    // We expect at least paper_understood to have started by now.
    expect(stageMarkers).toContain("paper_understood");
    // No fatal "Pipeline exited with status" or "Traceback" errors at this point.
    expect(log).not.toMatch(/Pipeline exited with status [^0]/);
  });

  test("B6. SSE — backend events + proxy enrichment for uploaded run", async () => {
    const backend = await fetchSse(`${BACKEND_BASE}/runs/${projectId}/events`, 8_000);
    expect(backend, "backend SSE produced no data for upload run").not.toEqual("");
    const backendEvents = Array.from(backend.matchAll(/^event:\s*(\S+)/gm)).map((m) => m[1]);
    test.info().annotations.push({
      type: "B6.backend_events",
      description: backendEvents.slice(0, 30).join(", ")
    });
    expect(backendEvents).toContain("run_state");

    const proxied = await fetchSse(
      `${PROXY_BASE}/api/demo/events?projectId=${encodeURIComponent(projectId)}`,
      8_000
    );
    expect(proxied, "proxy SSE produced no data for upload run").not.toEqual("");
    const proxyEvents = Array.from(proxied.matchAll(/^event:\s*(\S+)/gm)).map((m) => m[1]);
    test.info().annotations.push({
      type: "B6.proxy_events",
      description: proxyEvents.slice(0, 30).join(", ")
    });
    expect(proxyEvents).toContain("run_state");

    // payload may be null (pipeline_state.json not yet written); record actual value
    const dataLines = proxied
      .split(/\n\n/g)
      .filter((b) => /^event:\s*run_state/m.test(b))
      .map((b) => {
        const m = b.match(/^data:\s*(.*)$/m);
        return m ? m[1] : "";
      })
      .filter(Boolean);
    let stage: string | null = null;
    let payloadKey = "missing";
    for (const j of dataLines) {
      try {
        const p = JSON.parse(j);
        if (p?.payload) {
          payloadKey = "present";
          const s = p?.payload?.summary?.stage;
          if (typeof s === "string") {
            stage = s;
            break;
          }
        } else {
          payloadKey = "null";
        }
      } catch {
        // ignore
      }
    }
    test.info().annotations.push({
      type: "B6.payload_summary_stage",
      description: `${payloadKey} stage=${stage ?? "(none)"}`
    });
  });

  test("B7. SLOW — counter advances past 1 (pipeline reaches gate_1)", async () => {
    test.setTimeout(PIPELINE_PROGRESS_TIMEOUT_MS + 60_000);
    const before = await getDoneCount(page);
    test.info().annotations.push({ type: "B7.start_count", description: String(before) });

    await expect
      .poll(() => getDoneCount(page), {
        timeout: PIPELINE_PROGRESS_TIMEOUT_MS,
        intervals: [3000]
      })
      .toBeGreaterThan(1);

    const after = await getDoneCount(page);
    test.info().annotations.push({ type: "B7.final_count", description: String(after) });
    expect(consoleErrors.filter((e) => e.startsWith("pageerror"))).toEqual([]);
  });
});
