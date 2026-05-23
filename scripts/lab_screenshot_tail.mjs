#!/usr/bin/env node
// Periodic /lab screenshot loop. Saves PNGs + console-error logs into
// screenshots/ every 30s while a run is active. Stops when the run terminates.
//
//   node scripts/lab_screenshot_tail.mjs <projectId> [intervalSec]
//
// Auto-stops when runs/<projectId>/final_report.json exists OR
// runs/<projectId>/demo_status.json status is "failed" or "completed".

// Resolve playwright-core from frontend/node_modules without depending on
// NODE_PATH (which is ignored by Node's ESM resolver).
const { chromium } = await import(
  resolve(dirname(fileURLToPath(import.meta.url)), "..", "frontend", "node_modules", "playwright-core", "index.mjs")
);
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const REPO_ROOT = resolve(dirname(__filename), "..");

const projectId = process.argv[2];
const intervalSec = Number.parseInt(process.argv[3] ?? "30", 10);
if (!projectId) {
  console.error("usage: lab_screenshot_tail.mjs <projectId> [intervalSec]");
  process.exit(2);
}

const screenshotsDir = resolve(REPO_ROOT, "screenshots");
const runDir = resolve(REPO_ROOT, "runs", projectId);
await mkdir(screenshotsDir, { recursive: true });

async function runTerminated() {
  if (existsSync(resolve(runDir, "final_report.json"))) return "final_report.json";
  try {
    const raw = await readFile(resolve(runDir, "demo_status.json"), "utf8");
    const s = JSON.parse(raw).status;
    if (s === "failed" || s === "completed") return `status=${s}`;
  } catch { /* status not readable yet */ }
  return null;
}

console.log(`[tail] starting; project=${projectId} interval=${intervalSec}s`);
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1480, height: 900 } });
const page = await ctx.newPage();

const url = `http://localhost:3000/lab?projectId=${encodeURIComponent(projectId)}`;
const errors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push({ ts: new Date().toISOString(), text: msg.text() });
});
page.on("pageerror", (err) => errors.push({ ts: new Date().toISOString(), text: `pageerror: ${err.message}` }));

// Navigate ONCE at startup. Subsequent cycles use page.reload() which doesn't
// re-trigger SSR and avoids the timeout cliff when the backend is busy mid-
// implement_baseline. Initial navigate gets a longer timeout (60s) because the
// /lab SSR initial-state fetch can stall while the backend is processing a
// long primitive call.
try {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
  console.log(`[tail] initial navigate ok`);
} catch (e) {
  console.log(`[tail] initial navigate failed: ${e.message} — will retry per cycle`);
}

const wedgeLogPath = resolve(screenshotsDir, "wedge-log.tsv");
// Header (only if file is new — append-only otherwise)
if (!existsSync(wedgeLogPath)) {
  await writeFile(wedgeLogPath, "ts\tno_signal_secs\titeration\tstatus\tnotes\n");
}

let i = 0;
try {
  while (true) {
    const reason = await runTerminated();
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    let wedgeSecs = "";
    let iterationLabel = "";
    let statusLabel = "";
    let pageFallback = false;
    try {
      // page.reload() is cheaper than page.goto() and avoids re-triggering the
      // SSR initial-state fetch on every cycle. If the page is not on /lab
      // (e.g., we just started and the initial navigate failed, or we got
      // redirected), fall back to goto.
      const currentUrl = page.url();
      if (!currentUrl.includes("/lab?")) {
        await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
      } else {
        await page.reload({ waitUntil: "domcontentloaded", timeout: 60_000 });
      }
      await page.waitForTimeout(2000); // let SSE settle
      const pngPath = resolve(screenshotsDir, `lab-${ts}.png`);
      await page.screenshot({ path: pngPath, fullPage: true });
      // Scrape header indicators from the page text.
      const headerText = await page.evaluate(() => document.body.innerText || "");
      const wedgeMatch = headerText.match(/no signal\s+(\d+)s/i);
      const iterMatch  = headerText.match(/iteration\s+(\d+)/i);
      const statusMatch = headerText.match(/\b(queued|running|completed|failed|partial)\b/i);
      wedgeSecs = wedgeMatch ? wedgeMatch[1] : "";
      iterationLabel = iterMatch ? iterMatch[1] : "";
      statusLabel = statusMatch ? statusMatch[1] : "";
      // Detect the Upload-landing fallback: page rendered the upload card
      // instead of the lab. Means /lab?projectId=... SSR failed and Next.js
      // rendered the alternative state. Treat as a hard error for the cycle.
      const looksLikeUpload = /Upload PDF/i.test(headerText) && !iterationLabel && !statusLabel;
      if (looksLikeUpload) {
        pageFallback = true;
        console.log(`[tail] #${++i} ${pngPath} ⚠ PAGE_FALLBACK (rendered Upload landing, projectId lookup failed SSR)`);
      } else {
        const flag = wedgeSecs ? ` ⚠ no_signal=${wedgeSecs}s` : "";
        console.log(`[tail] #${++i} ${pngPath} iter=${iterationLabel} status=${statusLabel}${flag}${reason ? ` (terminating: ${reason})` : ""}`);
      }
    } catch (e) {
      console.log(`[tail] #${i} reload/screenshot failed: ${e.message}`);
    }
    if (errors.length) {
      const errPath = resolve(screenshotsDir, `console-errors-${ts}.json`);
      await writeFile(errPath, JSON.stringify(errors.splice(0), null, 2));
      console.log(`[tail]    console errors → ${errPath}`);
    }
    // Always append a wedge-log row. notes column carries PAGE_FALLBACK when
    // the page didn't render the lab view — distinguishes "chip not showing
    // because heartbeats are caught up" from "we screenshotted the wrong page."
    const notes = [pageFallback ? "PAGE_FALLBACK" : "", reason ?? ""].filter(Boolean).join("; ");
    await import("node:fs").then((fs) =>
      fs.promises.appendFile(
        wedgeLogPath,
        `${new Date().toISOString()}\t${wedgeSecs}\t${iterationLabel}\t${statusLabel}\t${notes}\n`,
      ),
    );
    if (reason) {
      console.log(`[tail] stopping: ${reason}`);
      break;
    }
    await new Promise((r) => setTimeout(r, intervalSec * 1000));
  }
} finally {
  await browser.close();
}
