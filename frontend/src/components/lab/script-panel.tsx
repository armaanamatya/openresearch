"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

import {
  finalReportUrl,
  formatBytes,
  shortHash,
  sourcePdfUrl,
  sourceTitle,
  verdictLabel
} from "./agent-info-helpers";
import { ICONS } from "./icons";

import "./script-panel.css";

export function ScriptPanel({ run }: { run: LiveDemoRunState }) {
  const pdf = run.sourcePdf;
  const benchmark = run.benchmark;
  const pdfUrl = sourcePdfUrl(run);
  const reportUrl = finalReportUrl(run);
  const score =
    benchmark && benchmark.overallScore > 0 ? `${benchmark.overallScore.toFixed(1)}%` : "Pending";
  const delta =
    benchmark && benchmark.reproducedValue > 0
      ? `${benchmark.deltaValue >= 0 ? "+" : ""}${benchmark.deltaValue.toFixed(1)}`
      : "pending";

  return (
    <div className="agent-section script-panel">
      <div className="script-panel-head">
        <div>
          <div className="eyebrow">Script panel</div>
          <div className="agent-task">Source PDF and final benchmark</div>
        </div>
        <span className="script-chip">code root</span>
      </div>

      <div className="pdf-card">
        <div className="pdf-preview" aria-hidden="true">
          {pdf ? (
            <object data={`${pdfUrl}#toolbar=0&navpanes=0`} type="application/pdf">
              <div className="pdf-fallback">{ICONS.doc}</div>
            </object>
          ) : (
            <div className="pdf-fallback">{ICONS.doc}</div>
          )}
        </div>
        <div className="pdf-copy">
          <div className="pdf-title">{pdf?.title ?? sourceTitle(run)}</div>
          <div className="pdf-meta">
            {pdf?.fileName ?? "paper.pdf"} · {formatBytes(pdf?.sizeBytes)}
            {pdf?.pageCount ? ` · ${pdf.pageCount} pages` : ""}
          </div>
          <div className="pdf-hash mono">sha256:{shortHash(pdf?.sha256)}</div>
          <div className="pdf-actions">
            <a className="btn btn-sm btn-dark" href={pdfUrl} target="_blank" rel="noreferrer">
              Preview PDF
            </a>
            <a className="btn btn-sm" href={pdfUrl} download="paper.pdf">
              Download
            </a>
          </div>
        </div>
      </div>

      <div className="code-root-row">
        <span className="code-root-label">Generated root</span>
        <span className="mono code-root-path">{pdf?.codePath ?? `${run.outputDir}/code/paper.pdf`}</span>
      </div>

      <a className="final-report-link" href={reportUrl} target="_blank" rel="noreferrer">
        <span className="final-report-kicker">Final report</span>
        <span className="final-report-title">
          {benchmark ? verdictLabel(benchmark.verdict) : "Benchmark report"}
        </span>
        <span className="final-report-copy">
          Open the formatted Markdown report generated alongside the codebase.
        </span>
        <span className="final-report-action">Open final report</span>
      </a>

      <div className="benchmark-card">
        <div className="benchmark-head">
          <div>
            <div className="benchmark-title">
              {benchmark?.benchmarkName ?? "PaperBench-style final benchmark"}
            </div>
            <div className="benchmark-subtitle">
              {benchmark?.paperbenchTaskId ?? "pending evaluator output"}
            </div>
          </div>
          <div className="benchmark-score">{score}</div>
        </div>
        <div className="metric-compare">
          <div>
            <span className="metric-label">Paper target</span>
            <strong>{benchmark ? benchmark.targetValue.toFixed(1) : "n/a"}</strong>
          </div>
          <div>
            <span className="metric-label">Reproduced</span>
            <strong>{benchmark && benchmark.reproducedValue > 0 ? benchmark.reproducedValue.toFixed(1) : "n/a"}</strong>
          </div>
          <div>
            <span className="metric-label">Delta</span>
            <strong>{delta}</strong>
          </div>
        </div>
        <div className="benchmark-verdict">
          <span className="status-dot" />
          {verdictLabel(benchmark?.verdict)}
        </div>
      </div>
    </div>
  );
}
