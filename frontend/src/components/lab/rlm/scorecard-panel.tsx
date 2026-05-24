"use client";

import type { RlmRunState } from "../../../hooks/use-rlm-run";
import styles from "./scorecard-panel.module.css";

interface ScorecardPanelProps {
  rubric: RlmRunState["rubric"];
  projectId: string;
  paperTitle: string;
}

function judgeLabel(status: "pass" | "partial" | "fail"): string {
  return status === "pass" ? "MATCH" : status === "partial" ? "DEVIATION" : "FAIL";
}

function judgeClass(status: "pass" | "partial" | "fail", styles: Record<string, string>): string {
  return status === "pass" ? styles.badgeMatch : status === "partial" ? styles.badgeDeviation : styles.badgeFail;
}

/**
 * ScorecardPanel — full-width reproduction scorecard table.
 *
 * Appears in Band 2.5 (between RubricStrip and workspace) when the run
 * has emitted at least one rubric_score event with per-area breakdown.
 * Styled to match the VerificationMock (FIG § 5.1) from the landing page.
 */
export function ScorecardPanel({ rubric, projectId, paperTitle }: ScorecardPanelProps) {
  const { areas, current, baseline, target } = rubric;

  if (areas.length === 0) return null;

  const passingCount = areas.filter((a) => a.status === "pass").length;
  const partialCount = areas.filter((a) => a.status === "partial").length;
  const delta = current !== null && baseline !== null ? current - baseline : null;

  const shortId = projectId.slice(0, 16);

  return (
    <div className={styles.panel} data-testid="scorecard-panel">
      {/* FIG label */}
      <div className={styles.figLabel}>FIG § 5.1 &nbsp;—&nbsp; REPRODUCTION SCORECARD</div>

      {/* Chrome bar */}
      <div className={styles.chrome}>
        <span className={styles.crumbs}>
          verification&nbsp;/&nbsp;<b>scorecard&nbsp;·&nbsp;{shortId}</b>
        </span>
        <span className={styles.chromeRight}>
          induced rubric&nbsp;·&nbsp;{areas.length}&nbsp;{areas.length === 1 ? "area" : "areas"}
          {target !== null ? ` · target ${target.toFixed(2)}` : ""}
        </span>
      </div>

      {/* Title row */}
      <div className={styles.titleRow}>
        <span className={styles.title}>Reproduction scorecard</span>
        {paperTitle && (
          <span className={styles.subtitle}>&nbsp;·&nbsp;{paperTitle.slice(0, 60)}{paperTitle.length > 60 ? "…" : ""}</span>
        )}
      </div>

      {/* Table */}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr className={styles.thead}>
              <th className={styles.thId}>ID</th>
              <th className={styles.thClaim}>AREA</th>
              <th className={styles.thNum}>SCORE</th>
              <th className={styles.thNum}>WEIGHT</th>
              <th className={styles.thJudge}>JUDGE</th>
            </tr>
          </thead>
          <tbody>
            {areas.map((a, i) => {
              const claimId = `C-${String(i + 1).padStart(3, "0")}`;
              return (
                <tr key={a.area || claimId} className={styles.row} data-status={a.status}>
                  <td className={styles.tdId}>{claimId}</td>
                  <td className={styles.tdClaim}>{a.area || "—"}</td>
                  <td className={styles.tdNum}>{a.score.toFixed(3)}</td>
                  <td className={styles.tdNum}>{(a.weight * 100).toFixed(0)}%</td>
                  <td className={styles.tdJudge}>
                    <span className={judgeClass(a.status, styles)}>
                      {judgeLabel(a.status)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Footer stat bar */}
      <div className={styles.footer}>
        <div className={styles.footerCell}>
          <div className={styles.footerLabel}>AREAS PASSING</div>
          <div className={styles.footerValue}>
            {passingCount}&nbsp;<span className={styles.footerDim}>/ {areas.length}</span>
          </div>
          <div className={styles.footerSub}>{partialCount} partial · scorecard</div>
        </div>
        <div className={styles.footerCell}>
          <div className={styles.footerLabel}>REPRODUCTION SCORE</div>
          <div className={styles.footerValue}>
            {current !== null ? current.toFixed(2) : <span className={styles.footerDim}>—</span>}
          </div>
          <div className={styles.footerSub}>rubric · weighted mean</div>
        </div>
        <div className={styles.footerCell}>
          <div className={styles.footerLabel}>DRIFT BAND</div>
          <div className={styles.footerValue}>
            {delta !== null ? (
              <>
                {delta >= 0 ? "+" : ""}
                {delta.toFixed(3)}
              </>
            ) : (
              <span className={styles.footerDim}>—</span>
            )}
          </div>
          <div className={styles.footerSub}>vs baseline</div>
        </div>
        <div className={styles.footerCell}>
          <div className={styles.footerLabel}>ARTIFACT</div>
          <div className={styles.footerValueMono}>report.md · run.json</div>
          <div className={styles.footerSub}>seed-pinned · reproducible</div>
        </div>
      </div>
    </div>
  );
}
