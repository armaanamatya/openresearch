"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

import "./hermes-audit-panel.css";

export function HermesAuditPanel({ run }: { run: LiveDemoRunState }) {
  const hermes = run.payload?.hermes;
  if (!hermes) {
    return (
      <div className="agent-section">
        <div className="eyebrow">Hermes audit</div>
        <div className="agent-detail">No audit findings yet.</div>
      </div>
    );
  }
  // Defensive defaults: backend Hermes payload can ship with missing keys
  // when an early-stage agent fails, an offline run skips audit, or an old
  // pipeline_state.json predates these fields. Don't crash the UI on it.
  const stepReports = hermes.stepReports && typeof hermes.stepReports === "object" ? hermes.stepReports : {};
  const checkpointReports = hermes.checkpointReports && typeof hermes.checkpointReports === "object" ? hermes.checkpointReports : {};
  const interventions = Array.isArray(hermes.interventions) ? hermes.interventions : [];
  const stepEntries = Object.entries(stepReports);
  const checkpointEntries = Object.entries(checkpointReports);

  return (
    <div className="agent-section hermes-panel">
      <div className="eyebrow">Hermes audit timeline</div>
      {stepEntries.length === 0 && checkpointEntries.length === 0 ? (
        <div className="agent-detail">No audit findings yet.</div>
      ) : null}
      {stepEntries.map(([stage, reports]) => {
        const latest = reports[reports.length - 1];
        if (!latest) return null;
        return (
          <div key={`step-${stage}`} className="hermes-row">
            <div className="hermes-row-head">
              <span className="hermes-row-stage">step / {stage}</span>
              <span className={`hermes-status hermes-status-${latest.status ?? "unknown"}`}>
                {latest.status ?? "unknown"}
              </span>
            </div>
            {latest.summary ? <div className="hermes-row-summary">{latest.summary}</div> : null}
            {latest.findings && latest.findings.length > 0 ? (
              <ul className="hermes-findings">
                {latest.findings.slice(0, 3).map((finding, idx) => (
                  <li key={idx}>{finding}</li>
                ))}
              </ul>
            ) : null}
          </div>
        );
      })}
      {checkpointEntries.map(([stage, reports]) => {
        const latest = reports[reports.length - 1];
        if (!latest) return null;
        return (
          <div key={`ckpt-${stage}`} className="hermes-row">
            <div className="hermes-row-head">
              <span className="hermes-row-stage">checkpoint / {stage}</span>
              <span className={`hermes-status hermes-status-${latest.status ?? "unknown"}`}>
                {latest.status ?? "unknown"}
              </span>
            </div>
            {latest.summary ? <div className="hermes-row-summary">{latest.summary}</div> : null}
          </div>
        );
      })}
      {interventions.length > 0 ? (
        <>
          <div className="eyebrow hermes-section-eyebrow">Interventions</div>
          <ul className="hermes-interventions">
            {interventions.slice(-5).reverse().map((intervention, idx) => (
              <li key={idx} className="hermes-intervention">
                <span className="hermes-intervention-action">{intervention.action ?? "action"}</span>
                <span className="hermes-intervention-target">on {intervention.target ?? "target"}</span>
                {intervention.reason ? (
                  <div className="hermes-intervention-reason">{intervention.reason}</div>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}
