import { test, expect, type Page } from "@playwright/test";
import path from "node:path";
import http from "node:http";
import fs from "node:fs";
import fsp from "node:fs/promises";

// ---------------------------------------------------------------------------
// Full end-to-end PDF reproduction test.
//
// Drives the entire /lab pipeline from PDF upload to `complete` and verifies
// the 8-section test matrix in docs/lab-e2e-full-reproduction-prompt.md. A real
// run uses live Anthropic calls (25-40 min wall time), so the driver test is
// mostly an instrument: it records a sync timeline + rail/node/gate snapshots
// into module state and a JSON artifact, then later tests assert against it.
// `expect.soft` is used in the long driver so every invariant is checked even
// if an earlier one fails.
// ---------------------------------------------------------------------------

const BACKEND_BASE = process.env.LAB_BACKEND_URL ?? "http://127.0.0.1:8001";
const PROXY_BASE = process.env.LAB_BASE_URL ?? "http://localhost:3001";
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PDF_PATH = path.join(REPO_ROOT, "demo_paper.pdf");
const RUNS_DIR = path.join(REPO_ROOT, "runs");
const ARTIFACT_PATH = path.join(__dirname, "..", "test-results", "lab-e2e-full-report.json");

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

// agent label -> workflow node id
const AGENT_TO_ID: Record<string, string> = {
  Paper: "src",
  Reader: "read",
  Forge: "env",
  Architect: "plan",
  Builder: "impl",
  Vesta: "opt",
  Athena: "bb",
  Orion: "aug",
  Lyra: "hor",
  Pyxis: "div",
  Hermes: "audit",
  Scribe: "report"
};
const PATH_AGENTS = ["Vesta", "Athena", "Orion", "Lyra", "Pyxis"] as const;

// Conservative (worst case: all-upcoming pathStates) done-count for each backend
// stage. Mirrors stateMapForRun's contract; section 2 asserts the live UI
// counter is never below this for the stage currently on disk.
const EXPECTED_COUNTER: Record<string, number> = {
  ingested: 1,
  paper_understood: 2,
  artifacts_discovered: 2,
  environment_built: 3,
  plan_created: 3,
  gate_1_passed: 4,
  baseline_implemented: 4,
  baseline_run: 4,
  gate_2_passed: 5,
  improvements_selected: 5,
  improvements_run: 5,
  gate_3_passed: 10,
  research_map_generated: 10,
  complete: 12
};
const STAGE_RANK: Record<string, number> = Object.fromEntries(
  Object.keys(EXPECTED_COUNTER).map((s, i) => [s, i])
);

const POLL_INTERVAL_MS = 4_000;
const RUN_TIMEOUT_MS = 42 * 60 * 1_000;
const DRIVER_TEST_TIMEOUT_MS = RUN_TIMEOUT_MS + 4 * 60 * 1_000;
// "within ~10s of a new disk stage the UI reflects it" — 3 poll intervals ≈ 12s.
const LAG_TOLERANCE_SAMPLES = 3;

// ---------------------------------------------------------------------------
// Types + module state shared across the serial test block.
// ---------------------------------------------------------------------------

type NodeVisualState = "done" | "running" | "upcoming";

interface Sample {
  tSec: number;
  status: string;
  diskStage: string | null;
  diskJsonValid: boolean;
  diskGates: { gate_1: string | null; gate_2: string | null; gate_3: string | null };
  uiCounter: number;
  uiGateChips: string[];
  rail: { agents: number; reasoning: number; context: number; decisions: number };
  nodeStates: Record<string, NodeVisualState>;
}

interface SseCollector {
  label: string;
  url: string;
  events: Record<string, number>;
  synthFrames: { id: string; stage: string | null }[];
  runStateStages: (string | null)[];
  startedAt: number;
  lastDataAt: number;
  bytes: number;
  stop: () => void;
}

let sharedPage: Page;
let projectId = "";
let runDir = "";
const consoleErrors: string[] = [];
const samples: Sample[] = [];
let invalidJsonReads = 0;
let backendSse: SseCollector | null = null;
let proxySse: SseCollector | null = null;
let driverReachedComplete = false;
let finalStatus = "";
let finalDiskStage: string | null = null;
const panelResults: Record<string, unknown> = {};
const sectionFindings: Record<string, unknown> = {};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function startSseCollector(label: string, url: string): SseCollector {
  const c: SseCollector = {
    label,
    url,
    events: {},
    synthFrames: [],
    runStateStages: [],
    startedAt: Date.now(),
    lastDataAt: Date.now(),
    bytes: 0,
    stop: () => undefined
  };
  let pending = "";
  const req = http.get(url, { headers: { Accept: "text/event-stream" } }, (res) => {
    res.setEncoding("utf-8");
    res.on("data", (chunk: string) => {
      c.lastDataAt = Date.now();
      c.bytes += chunk.length;
      pending += chunk;
      const frames = pending.split("\n\n");
      pending = frames.pop() ?? "";
      for (const frame of frames) {
        if (!frame.trim()) continue;
        const ev = frame.match(/^event:\s*(\S+)/m)?.[1] ?? "message";
        const id = frame.match(/^id:\s*(\S+)/m)?.[1] ?? null;
        const data = frame.match(/^data:\s*(.*)$/m)?.[1] ?? null;
        c.events[ev] = (c.events[ev] ?? 0) + 1;
        let stage: string | null = null;
        if (ev === "run_state" && data) {
          try {
            stage = JSON.parse(data)?.payload?.summary?.stage ?? null;
          } catch {
            stage = null;
          }
          c.runStateStages.push(stage);
        }
        if (id && id.startsWith("synth-")) {
          c.synthFrames.push({ id, stage });
        }
      }
    });
    res.on("error", () => undefined);
  });
  req.on("error", () => undefined);
  c.stop = () => {
    try {
      req.destroy();
    } catch {
      /* ignore */
    }
  };
  return c;
}

async function getProjectId(page: Page): Promise<string> {
  const eyebrow = page.locator(".workflow-header .eyebrow");
  await expect(eyebrow).toBeVisible({ timeout: 60_000 });
  const text = (await eyebrow.textContent()) ?? "";
  const match = text.match(/workflow\s*-\s*(\S+)/i);
  if (!match) throw new Error(`could not extract projectId from header: ${JSON.stringify(text)}`);
  return match[1];
}

async function getDoneCount(page: Page): Promise<number> {
  try {
    const counter = page.locator(".workflow-meta .mono").first();
    const text = (await counter.textContent({ timeout: 5_000 })) ?? "";
    const m = text.match(/(\d+)\/(\d+) agents complete/);
    return m ? Number(m[1]) : -1;
  } catch {
    return -1;
  }
}

function nodeCard(page: Page, agent: string) {
  return page.locator(`div[data-node="1"]:has(.node-agent:text-is("${agent}"))`).first();
}

async function readNodeStates(page: Page): Promise<Record<string, NodeVisualState>> {
  const out: Record<string, NodeVisualState> = {};
  for (const agent of NODE_AGENTS) {
    const id = AGENT_TO_ID[agent];
    try {
      const card = nodeCard(page, agent);
      if ((await card.locator(".node-check").count()) > 0) out[id] = "done";
      else if ((await card.locator(".wf-ring").count()) > 0) out[id] = "running";
      else out[id] = "upcoming";
    } catch {
      out[id] = "upcoming";
    }
  }
  return out;
}

async function readGateChips(page: Page): Promise<string[]> {
  try {
    const chips = page.locator(".gate-chip");
    const n = await chips.count();
    const states: string[] = [];
    for (let i = 0; i < n; i += 1) {
      const cls = (await chips.nth(i).getAttribute("class")) ?? "";
      const m = cls.match(/gate-chip-(\S+)/);
      states.push(m ? m[1] : "unknown");
    }
    return states;
  } catch {
    return [];
  }
}

async function readRail(page: Page) {
  const count = async (sel: string) => {
    try {
      return await page.locator(sel).count();
    } catch {
      return 0;
    }
  };
  return {
    agents: await count(".timeline-agents .timeline-agent"),
    reasoning: await count(".timeline-reason"),
    context: await count(".timeline-context"),
    decisions: await count(".timeline-decision")
  };
}

function readPipelineState(): { valid: boolean; stage: string | null; gates: Sample["diskGates"] } {
  const file = path.join(runDir, "pipeline_state.json");
  if (!fs.existsSync(file)) {
    return { valid: true, stage: null, gates: { gate_1: null, gate_2: null, gate_3: null } };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(file, "utf-8"));
    const gateStatus = (g: unknown): string | null => {
      if (g && typeof g === "object") {
        const obj = g as Record<string, unknown>;
        if (typeof obj.status === "string") return obj.status;
        if (typeof obj.passed === "boolean") return obj.passed ? "passed" : "failed";
      }
      return null;
    };
    return {
      valid: true,
      stage: typeof parsed.stage === "string" ? parsed.stage : null,
      gates: {
        gate_1: gateStatus(parsed.gate_1),
        gate_2: gateStatus(parsed.gate_2),
        gate_3: gateStatus(parsed.gate_3)
      }
    };
  } catch {
    // Half-written read — the atomic-write fix should make this never happen.
    return { valid: false, stage: null, gates: { gate_1: null, gate_2: null, gate_3: null } };
  }
}

function readDemoStatus(): string {
  const file = path.join(runDir, "demo_status.json");
  try {
    const parsed = JSON.parse(fs.readFileSync(file, "utf-8"));
    return typeof parsed.status === "string" ? parsed.status : "unknown";
  } catch {
    return "unknown";
  }
}

async function writeArtifact() {
  const artifact = {
    projectId,
    runDir,
    generatedAt: new Date().toISOString(),
    driverReachedComplete,
    finalStatus,
    finalDiskStage,
    sampleCount: samples.length,
    invalidJsonReads,
    consoleErrors,
    samples,
    backendSse: backendSse
      ? {
          events: backendSse.events,
          synthFrames: backendSse.synthFrames.length,
          runStateStages: backendSse.runStateStages,
          bytes: backendSse.bytes
        }
      : null,
    proxySse: proxySse
      ? {
          events: proxySse.events,
          synthFrames: proxySse.synthFrames,
          runStateStages: proxySse.runStateStages,
          bytes: proxySse.bytes
        }
      : null,
    panelResults,
    sectionFindings
  };
  await fsp.mkdir(path.dirname(ARTIFACT_PATH), { recursive: true });
  await fsp.writeFile(ARTIFACT_PATH, JSON.stringify(artifact, null, 2), "utf-8");
}

// ---------------------------------------------------------------------------
// Not `serial` mode on purpose: tests run sequentially via the config's
// `workers: 1` + `fullyParallel: false`, but if the driver (test 2) fails
// because the run did not reach `complete`, tests 3-8 must still run so the
// partial verification is recorded — per the prompt's "partial sync
// verification still counts".

test.describe("Lab pipeline — full PDF reproduction E2E", () => {
  test.beforeAll(async ({ browser }) => {
    consoleErrors.length = 0;
    samples.length = 0;

    sharedPage = await browser.newPage();

    // Count EventSource instantiations so section 7 can prove the stream is
    // opened once (not reconnect-looping).
    await sharedPage.addInitScript(() => {
      const RealES = window.EventSource;
      const log: string[] = [];
      (window as unknown as { __esLog: string[] }).__esLog = log;
      class CountingEventSource extends RealES {
        constructor(url: string | URL, init?: EventSourceInit) {
          super(url, init);
          log.push(String(url));
        }
      }
      (window as unknown as { EventSource: typeof EventSource }).EventSource =
        CountingEventSource as unknown as typeof EventSource;
    });

    sharedPage.on("pageerror", (err) => {
      consoleErrors.push(`pageerror: ${err.message}`);
    });
    sharedPage.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
    });

    await sharedPage.goto("/lab");
    const fileInput = sharedPage.locator(
      'input[type="file"][aria-label="Upload paper PDF"]'
    );
    await fileInput.setInputFiles(PDF_PATH);
    await expect(sharedPage.locator(".workflow-header")).toBeVisible({ timeout: 60_000 });
    projectId = await getProjectId(sharedPage);
    runDir = path.join(RUNS_DIR, projectId);

    backendSse = startSseCollector("backend", `${BACKEND_BASE}/runs/${projectId}/events`);
    proxySse = startSseCollector(
      "proxy",
      `${PROXY_BASE}/api/demo/events?projectId=${encodeURIComponent(projectId)}`
    );
  });

  test.afterAll(async () => {
    backendSse?.stop();
    proxySse?.stop();
    await writeArtifact().catch(() => undefined);
    await sharedPage?.close();
  });

  // -------------------------------------------------------------------------
  test("1. Upload + handoff", async () => {
    expect(projectId, "projectId from workflow header").toMatch(/^prj_/);

    await expect(sharedPage.locator('div[data-node="1"]')).toHaveCount(12, {
      timeout: 15_000
    });
    for (const agent of NODE_AGENTS) {
      await expect(nodeCard(sharedPage, agent)).toBeVisible();
    }

    // runs/<projectId>/ created and demo_status.json shows the run is live.
    await expect
      .poll(() => fs.existsSync(runDir), { timeout: 30_000, intervals: [1000] })
      .toBe(true);
    await expect
      .poll(() => readDemoStatus(), { timeout: 30_000, intervals: [1000] })
      .toMatch(/^(queued|running)$/);

    sectionFindings.section1 = {
      projectId,
      runDirExists: fs.existsSync(runDir),
      demoStatus: readDemoStatus(),
      nodeCount: await sharedPage.locator('div[data-node="1"]').count()
    };

    expect(
      consoleErrors.filter((e) => e.startsWith("pageerror")),
      "no uncaught exceptions on upload"
    ).toEqual([]);
  });

  // -------------------------------------------------------------------------
  test("2. Live stage sync — drive to complete", async () => {
    test.setTimeout(DRIVER_TEST_TIMEOUT_MS);
    const t0 = Date.now();

    const terminal = new Set(["completed", "failed", "stopped"]);
    let status = readDemoStatus();

    while (Date.now() - t0 < RUN_TIMEOUT_MS) {
      status = readDemoStatus();
      const disk = readPipelineState();
      if (!disk.valid) invalidJsonReads += 1;

      const sample: Sample = {
        tSec: Math.round((Date.now() - t0) / 1000),
        status,
        diskStage: disk.stage,
        diskJsonValid: disk.valid,
        diskGates: disk.gates,
        uiCounter: await getDoneCount(sharedPage),
        uiGateChips: await readGateChips(sharedPage),
        rail: await readRail(sharedPage),
        nodeStates: await readNodeStates(sharedPage)
      };
      samples.push(sample);

      if (terminal.has(status)) break;
      await sharedPage.waitForTimeout(POLL_INTERVAL_MS);
    }

    // After terminal status, give the UI up to 30s to render the final frame.
    if (terminal.has(status)) {
      const settleDeadline = Date.now() + 30_000;
      while (Date.now() < settleDeadline) {
        const disk = readPipelineState();
        if (!disk.valid) invalidJsonReads += 1;
        const sample: Sample = {
          tSec: Math.round((Date.now() - t0) / 1000),
          status,
          diskStage: disk.stage,
          diskJsonValid: disk.valid,
          diskGates: disk.gates,
          uiCounter: await getDoneCount(sharedPage),
          uiGateChips: await readGateChips(sharedPage),
          rail: await readRail(sharedPage),
          nodeStates: await readNodeStates(sharedPage)
        };
        samples.push(sample);
        if (sample.uiCounter >= 12) break;
        await sharedPage.waitForTimeout(3_000);
      }
    }

    finalStatus = status;
    finalDiskStage = [...samples].reverse().find((s) => s.diskStage)?.diskStage ?? null;
    const peakCounter = samples.reduce((m, s) => Math.max(m, s.uiCounter), 0);
    driverReachedComplete = finalDiskStage === "complete" && status === "completed";
    await writeArtifact().catch(() => undefined);

    // --- invariants checked regardless of how far the run got -------------
    // UI counter is monotonic (this is the stateMapForRun fix under test).
    for (let i = 1; i < samples.length; i += 1) {
      const prev = samples[i - 1].uiCounter;
      const cur = samples[i].uiCounter;
      if (prev < 0 || cur < 0) continue; // skip transient counter read misses
      expect
        .soft(
          cur,
          `UI counter regressed at t=${samples[i].tSec}s (disk stage ${samples[i].diskStage}): ${prev} → ${cur}`
        )
        .toBeGreaterThanOrEqual(prev);
    }

    // UI counter is never more than LAG_TOLERANCE_SAMPLES behind the disk stage.
    let maxLag = 0;
    for (let i = 0; i < samples.length; i += 1) {
      const ref = samples[Math.max(0, i - LAG_TOLERANCE_SAMPLES)];
      const refStage = ref.diskStage;
      if (!refStage || !(refStage in EXPECTED_COUNTER)) continue;
      const expected = EXPECTED_COUNTER[refStage];
      if (samples[i].uiCounter < 0) continue;
      if (samples[i].uiCounter < expected) {
        // count how long the UI has been behind
        let lag = 0;
        for (let j = i; j >= 0 && samples[j].uiCounter < expected; j -= 1) lag += 1;
        maxLag = Math.max(maxLag, lag);
      }
    }
    expect
      .soft(maxLag, `UI counter lagged the disk stage for ${maxLag} poll intervals`)
      .toBeLessThanOrEqual(LAG_TOLERANCE_SAMPLES);

    // pipeline_state.json is always valid JSON when read mid-run (atomic write).
    expect.soft(invalidJsonReads, "half-written pipeline_state.json reads").toBe(0);

    // No uncaught browser exceptions during the run.
    expect
      .soft(consoleErrors.filter((e) => e.startsWith("pageerror")), "uncaught exceptions during run")
      .toEqual([]);

    // Node animation order: src done immediately; later anchors done by their stage.
    const firstDoneAt = (id: string) =>
      samples.findIndex((s) => s.nodeStates[id] === "done");
    const stageReachedAt = (stage: string) =>
      samples.findIndex((s) => s.diskStage && STAGE_RANK[s.diskStage] >= STAGE_RANK[stage]);
    const animationOrder = {
      srcDoneAtSample: firstDoneAt("src"),
      readDoneAtSample: firstDoneAt("read"),
      implDoneAtSample: firstDoneAt("impl"),
      auditDoneAtSample: firstDoneAt("audit"),
      reportDoneAtSample: firstDoneAt("report"),
      paperUnderstoodAtSample: stageReachedAt("paper_understood"),
      gate2AtSample: stageReachedAt("gate_2_passed"),
      completeAtSample: stageReachedAt("complete")
    };
    sectionFindings.section2 = {
      finalStatus,
      finalDiskStage,
      peakCounter,
      sampleCount: samples.length,
      maxLagSamples: maxLag,
      invalidJsonReads,
      animationOrder,
      pathNodesEverRunning: PATH_AGENTS.map((a) => AGENT_TO_ID[a]).filter((id) =>
        samples.some((s) => s.nodeStates[id] === "running")
      ),
      pathNodesStuckRunningAtEnd: PATH_AGENTS.map((a) => AGENT_TO_ID[a]).filter(
        (id) => samples[samples.length - 1]?.nodeStates[id] === "running"
      )
    };

    // src must be done within the first few samples.
    expect.soft(firstDoneAt("src"), "src node done early").toBeGreaterThanOrEqual(0);
    expect.soft(firstDoneAt("src"), "src node done early").toBeLessThanOrEqual(3);

    // Final state, only asserted when the run actually completed.
    if (driverReachedComplete) {
      expect.soft(peakCounter, "counter reached 12/12").toBe(12);
      expect
        .soft(samples[samples.length - 1].uiCounter, "final counter is 12/12")
        .toBe(12);
      expect.soft(readDemoStatus(), "demo_status.json").toBe("completed");
      expect.soft(readPipelineState().stage, "pipeline_state.json stage").toBe("complete");
      // No path node may be stuck on `running` at the end.
      for (const agent of PATH_AGENTS) {
        expect
          .soft(
            samples[samples.length - 1].nodeStates[AGENT_TO_ID[agent]],
            `${agent} not stuck running`
          )
          .not.toBe("running");
      }
    }

    // Hard gate: surface incompleteness clearly, but only after every soft
    // invariant above has been recorded.
    expect(
      driverReachedComplete,
      `run did not reach complete — furthest disk stage: ${finalDiskStage}, status: ${finalStatus}`
    ).toBe(true);
  });

  // -------------------------------------------------------------------------
  test("3. Gate chips", async () => {
    // From the recorded timeline: once a gate has a decision on disk, the chip
    // must not be stuck on `pending`.
    const gateKeys = ["gate_1", "gate_2", "gate_3"] as const;
    const findings: Record<string, unknown> = {};
    for (let g = 0; g < gateKeys.length; g += 1) {
      const key = gateKeys[g];
      // first sample where the gate decision is present on disk
      const decidedIdx = samples.findIndex((s) => s.diskGates[key]);
      const chipStatesAfterDecision = decidedIdx >= 0
        ? samples.slice(decidedIdx).map((s) => s.uiGateChips[g]).filter(Boolean)
        : [];
      const stuckPending =
        decidedIdx >= 0 &&
        // allow a few poll intervals for the chip to catch up
        samples
          .slice(decidedIdx + LAG_TOLERANCE_SAMPLES)
          .every((s) => s.uiGateChips[g] === "pending");
      findings[key] = {
        diskDecisionFirstSeenAtSample: decidedIdx,
        diskDecision: decidedIdx >= 0 ? samples[decidedIdx].diskGates[key] : null,
        chipStatesObserved: Array.from(new Set(samples.map((s) => s.uiGateChips[g]).filter(Boolean))),
        chipStatesAfterDecision: Array.from(new Set(chipStatesAfterDecision)),
        stuckPendingAfterDecision: stuckPending
      };
      // If the gate was decided on disk during the run, the chip must resolve.
      if (decidedIdx >= 0 && driverReachedComplete) {
        expect
          .soft(stuckPending, `${key} chip stuck on pending after on-disk decision`)
          .toBe(false);
        expect
          .soft(
            chipStatesAfterDecision.some((st) => ["passed", "caveat", "failed"].includes(st)),
            `${key} chip reached a resolved state`
          )
          .toBe(true);
      }
    }
    // Live read of the final chip states.
    findings.finalChipStates = await readGateChips(sharedPage);
    sectionFindings.section3 = findings;
    await writeArtifact().catch(() => undefined);

    expect(
      samples.some((s) => s.uiGateChips.length === 3),
      "three gate chips rendered during the run"
    ).toBe(true);
  });

  // -------------------------------------------------------------------------
  test("4. Right rail", async () => {
    const peak = samples.reduce(
      (acc, s) => ({
        agents: Math.max(acc.agents, s.rail.agents),
        reasoning: Math.max(acc.reasoning, s.rail.reasoning),
        context: Math.max(acc.context, s.rail.context),
        decisions: Math.max(acc.decisions, s.rail.decisions)
      }),
      { agents: 0, reasoning: 0, context: 0, decisions: 0 }
    );
    const liveRail = await readRail(sharedPage);
    sectionFindings.section4 = { peakDuringRun: peak, liveRail };
    await writeArtifact().catch(() => undefined);

    expect.soft(peak.agents, "Live agents populated during run").toBeGreaterThan(0);
    expect.soft(peak.reasoning, "Reasoning populated during run").toBeGreaterThan(0);
    expect.soft(peak.context, "Context handoffs populated during run").toBeGreaterThan(0);
    expect.soft(peak.decisions, "Decisions populated during run").toBeGreaterThan(0);
  });

  // -------------------------------------------------------------------------
  test("5. Per-node panels", async () => {
    const results: Record<string, unknown> = {};
    for (const agent of NODE_AGENTS) {
      const id = AGENT_TO_ID[agent];
      const card = nodeCard(sharedPage, agent);
      await expect(card).toBeVisible();
      const opacity = await card.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
      if (opacity <= 0.5) {
        results[id] = { agent, clickable: false, note: "still upcoming" };
        continue;
      }
      await card.click({ force: true });
      const opened = await sharedPage
        .locator(".agent-name", { hasText: agent })
        .first()
        .waitFor({ state: "visible", timeout: 8_000 })
        .then(() => true)
        .catch(() => false);

      const entry: Record<string, unknown> = { agent, clickable: true, panelOpened: opened };
      if (opened) {
        entry.agentTask =
          (await sharedPage.locator(".agent-task").first().textContent().catch(() => null))?.slice(
            0,
            160
          ) ?? null;
        entry.telemetryRows = await sharedPage.locator(".telemetry-list .telemetry-row").count();
        entry.logLines = await sharedPage.locator(".agent-log-list .agent-log-item").count();
        entry.sectionEyebrows = await sharedPage
          .locator(".agent-section .eyebrow")
          .allTextContents();

        if (id === "audit") {
          entry.hermesPanelVisible = await sharedPage
            .locator(".hermes-panel")
            .first()
            .isVisible()
            .catch(() => false);
          entry.hermesText =
            (await sharedPage.locator(".hermes-panel").first().textContent().catch(() => null))?.slice(
              0,
              240
            ) ?? null;
        }
        if (id === "report") {
          const scriptPanel = sharedPage.locator(".script-panel");
          entry.scriptPanelVisible = await scriptPanel.isVisible().catch(() => false);
          const previewBtn = scriptPanel.locator(".pdf-actions a", { hasText: /preview pdf/i });
          const downloadBtn = scriptPanel.locator(".pdf-actions a", { hasText: /^download$/i });
          entry.previewHref = await previewBtn.getAttribute("href").catch(() => null);
          entry.downloadHref = await downloadBtn.getAttribute("href").catch(() => null);
          entry.pdfMeta =
            (await scriptPanel.locator(".pdf-meta").textContent().catch(() => null)) ?? null;
          entry.benchmarkText =
            (await scriptPanel.locator(".benchmark-card").textContent().catch(() => null))?.slice(
              0,
              240
            ) ?? null;
          entry.finalReportHref = await sharedPage
            .locator(".final-report-link")
            .getAttribute("href")
            .catch(() => null);
        }
      }
      results[id] = entry;
    }

    panelResults.section5 = results;
    await writeArtifact().catch(() => undefined);

    if (driverReachedComplete) {
      // All 12 nodes are clickable on a completed run.
      for (const agent of NODE_AGENTS) {
        const id = AGENT_TO_ID[agent];
        expect
          .soft((results[id] as { panelOpened?: boolean })?.panelOpened, `${agent} panel opens`)
          .toBe(true);
      }
      // read/env/plan/impl: multi-line "Latest log" tail (≥2 lines).
      for (const id of ["read", "env", "plan", "impl"]) {
        expect
          .soft((results[id] as { logLines?: number })?.logLines ?? 0, `${id} log tail ≥ 2 lines`)
          .toBeGreaterThanOrEqual(2);
      }
      // report node: hrefs include projectId.
      const report = results.report as { previewHref?: string; downloadHref?: string };
      expect.soft(report?.previewHref ?? "", "preview href includes projectId").toContain(projectId);
      expect
        .soft(report?.downloadHref ?? "", "download href includes projectId")
        .toContain(projectId);
    }
  });

  // -------------------------------------------------------------------------
  test("6. Final report integrity", async () => {
    const sourcePdfUrl = `${PROXY_BASE}/api/demo/source-pdf?projectId=${encodeURIComponent(
      projectId
    )}`;
    const finalReportUrl = `${PROXY_BASE}/api/demo/final-report?projectId=${encodeURIComponent(
      projectId
    )}`;

    const pdfResp = await sharedPage.request.get(sourcePdfUrl);
    const pdfStatus = pdfResp.status();
    const pdfCtype = pdfResp.headers()["content-type"] ?? "";
    const pdfBody = await pdfResp.body();

    const reportResp = await sharedPage.request.get(finalReportUrl);
    const reportStatus = reportResp.status();
    const reportCtype = reportResp.headers()["content-type"] ?? "";
    const reportText = await reportResp.text();

    sectionFindings.section6 = {
      sourcePdf: { url: sourcePdfUrl, status: pdfStatus, contentType: pdfCtype, bytes: pdfBody.length },
      finalReport: {
        url: finalReportUrl,
        status: reportStatus,
        contentType: reportCtype,
        length: reportText.length,
        head: reportText.slice(0, 240)
      },
      benchmarkText: (panelResults.section5 as Record<string, { benchmarkText?: string }>)?.report
        ?.benchmarkText
    };
    await writeArtifact().catch(() => undefined);

    expect.soft(pdfStatus, "source-pdf HTTP status").toBe(200);
    expect.soft(pdfCtype, "source-pdf content-type").toMatch(/application\/pdf/);
    expect.soft(pdfBody.length, "source-pdf has bytes").toBeGreaterThan(0);

    if (driverReachedComplete) {
      expect.soft(reportStatus, "final-report HTTP status").toBe(200);
      expect.soft(reportText.length, "final-report has content").toBeGreaterThan(0);
      const benchmark =
        (panelResults.section5 as Record<string, { benchmarkText?: string }>)?.report
          ?.benchmarkText ?? "";
      expect
        .soft(/pending|n\/a/i.test(benchmark), "benchmark card shows real numbers")
        .toBe(false);
    }
  });

  // -------------------------------------------------------------------------
  test("7. SSE health", async () => {
    const backend = backendSse!;
    const proxy = proxySse!;

    // First non-null proxy stage, then every later run_state must stay non-null.
    const firstNonNull = proxy.runStateStages.findIndex((s) => s != null);
    const nullAfterFirst =
      firstNonNull >= 0 &&
      proxy.runStateStages.slice(firstNonNull).some((s) => s == null);

    const stageTransitions = (() => {
      let n = 0;
      let last: string | null = null;
      for (const s of samples) {
        if (s.diskStage && s.diskStage !== last) {
          if (last != null) n += 1;
          last = s.diskStage;
        }
      }
      return n;
    })();

    let esLog: string[] = [];
    try {
      esLog = await sharedPage.evaluate(
        () => (window as unknown as { __esLog?: string[] }).__esLog ?? []
      );
    } catch {
      esLog = [];
    }

    sectionFindings.section7 = {
      backendEvents: backend.events,
      backendBytes: backend.bytes,
      proxyEvents: proxy.events,
      proxyBytes: proxy.bytes,
      proxySynthFrames: proxy.synthFrames.length,
      proxySynthStages: proxy.synthFrames.map((f) => f.stage),
      proxyRunStateStageSamples: proxy.runStateStages.length,
      proxyFirstNonNullStageIdx: firstNonNull,
      proxyNullStageAfterFirstNonNull: nullAfterFirst,
      observedStageTransitions: stageTransitions,
      eventSourceOpenCount: esLog.length,
      eventSourceUrls: esLog
    };
    await writeArtifact().catch(() => undefined);

    // Backend stream carried the core event types.
    expect.soft(backend.events.run_state ?? 0, "backend run_state frames").toBeGreaterThan(0);
    expect
      .soft(
        (backend.events.agent_log ?? 0) +
          (backend.events.dashboard_event ?? 0) +
          (backend.events.heartbeat ?? 0),
        "backend streaming events (agent_log/dashboard_event/heartbeat)"
      )
      .toBeGreaterThan(0);

    // Proxy stream carried run_state and emitted synthetic enriched frames.
    expect.soft(proxy.events.run_state ?? 0, "proxy run_state frames").toBeGreaterThan(0);
    expect
      .soft(proxy.synthFrames.length, "proxy emitted synthetic enriched frames during the run")
      .toBeGreaterThan(0);

    // payload.summary.stage never regresses to null once populated.
    expect.soft(nullAfterFirst, "proxy payload.summary.stage went back to null").toBe(false);

    // EventSource opened a small number of times — not reconnect-looping.
    expect.soft(esLog.length, "EventSource open count").toBeGreaterThan(0);
    expect.soft(esLog.length, "EventSource not reconnect-looping").toBeLessThanOrEqual(4);
    for (const url of esLog) {
      expect.soft(url, "EventSource URL targets this run").toContain(projectId);
    }
  });

  // -------------------------------------------------------------------------
  test("8. Robustness / regression", async () => {
    const stderrPath = path.join(runDir, "runner.stderr.log");
    const stderr = await fsp.readFile(stderrPath, "utf-8").catch(() => "");
    const tmpLeftover = fs.existsSync(path.join(runDir, "pipeline_state.json.tmp"));
    const badExit = /Pipeline exited with status [^0\s]/.test(stderr);
    const traceback = /Traceback \(most recent call last\)/.test(stderr);

    sectionFindings.section8 = {
      runnerStderrBytes: stderr.length,
      pipelineExitedNonZero: badExit,
      tracebackInStderr: traceback,
      invalidJsonReads,
      tmpFileLeftover: tmpLeftover,
      pageErrors: consoleErrors.filter((e) => e.startsWith("pageerror")),
      consoleErrorCount: consoleErrors.length
    };
    await writeArtifact().catch(() => undefined);

    expect
      .soft(consoleErrors.filter((e) => e.startsWith("pageerror")), "no uncaught exceptions for the run")
      .toEqual([]);
    expect.soft(badExit, "no non-zero pipeline exit in runner.stderr.log").toBe(false);
    expect.soft(traceback, "no Traceback in runner.stderr.log").toBe(false);
    expect.soft(invalidJsonReads, "no half-written pipeline_state.json reads").toBe(0);
    expect.soft(tmpLeftover, "no pipeline_state.json.tmp left behind").toBe(false);
  });
});
