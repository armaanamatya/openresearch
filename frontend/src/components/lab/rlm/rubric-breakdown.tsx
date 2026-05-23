"use client";

import { useState } from "react";

import type { DemoClusterStatus, DemoLeafScore, DemoRepairPass } from "@/lib/demo/demo-run-types";
import { useRdrArtifacts } from "@/hooks/use-rdr-artifacts";
import styles from "./rubric-breakdown.module.css";

interface RubricBreakdownProps {
  projectId: string;
  isActive: boolean;
}

function clusterStatus(c: DemoClusterStatus): "pending" | "success" | "failed" {
  if (c.failed === null) return "pending";
  return c.failed ? "failed" : "success";
}

function ClusterGrid({ clusters }: { clusters: DemoClusterStatus[] }) {
  if (clusters.length === 0) return null;
  return (
    <div>
      <p className={styles.sectionLabel}>Clusters</p>
      <div className={styles.clusterGrid}>
        {clusters.map((c) => {
          const status = clusterStatus(c);
          const label = c.title || c.cluster_id;
          const tip =
            status === "failed" && c.repair_history.length > 0
              ? `${label} — failed (${c.repair_history.length} repair attempt${c.repair_history.length !== 1 ? "s" : ""})`
              : label;
          return (
            <span
              key={c.cluster_id}
              className={styles.clusterPill}
              data-status={status}
              title={tip}
            >
              {label}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function RepairBanner({ passes }: { passes: DemoRepairPass[] }) {
  if (passes.length === 0) return null;
  const last = passes[passes.length - 1];
  return (
    <div className={styles.repairBanner}>
      <span className={styles.repairBannerLabel}>
        Repair pass {last.pass}
      </span>
      <span>
        {last.cluster_count} cluster{last.cluster_count !== 1 ? "s" : ""},{" "}
        {last.failed_count} failed
      </span>
    </div>
  );
}

function LeafRow({ leaf }: { leaf: DemoLeafScore }) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(Math.min(1, Math.max(0, leaf.score)) * 100);
  return (
    <div className={styles.leafRow}>
      <span className={styles.leafId} title={leaf.id}>
        {leaf.id}
      </span>
      <div className={styles.leafBarTrack}>
        <div className={styles.leafBarFill} style={{ width: `${pct}%` }} />
      </div>
      <span className={styles.leafScore}>{leaf.score.toFixed(2)}</span>
      {leaf.justification ? (
        <span
          className={styles.leafJustification}
          data-expanded={expanded ? "true" : "false"}
          title={expanded ? undefined : leaf.justification}
          onClick={() => setExpanded((v) => !v)}
        >
          {leaf.justification}
        </span>
      ) : null}
    </div>
  );
}

function LeafScoreList({
  leafScores,
  overallScore,
}: {
  leafScores: DemoLeafScore[];
  overallScore?: number;
}) {
  if (leafScores.length === 0) return null;
  return (
    <div>
      <p className={styles.sectionLabel}>
        Leaf scores
        {overallScore !== undefined ? (
          <>
            {" "}
            <span className={styles.overallScore}>{overallScore.toFixed(3)}</span>{" "}
            <span className={styles.overallScoreLabel}>overall</span>
          </>
        ) : null}
      </p>
      <div className={styles.leafList}>
        {leafScores.map((l) => (
          <LeafRow key={l.id} leaf={l} />
        ))}
      </div>
    </div>
  );
}

/**
 * RubricBreakdown — cluster grid + leaf score list for rdr/rlm runs.
 *
 * Renders only when the run has rdr artifact data. Polls the three
 * /clusters, /leaf-scores, and /repair-iterations proxy endpoints via
 * useRdrArtifacts. Empty-state (no data yet) renders nothing.
 */
export function RubricBreakdown({ projectId, isActive }: RubricBreakdownProps) {
  const { clusters, leafScores, repairPasses } = useRdrArtifacts(projectId, isActive);

  const hasData = clusters.length > 0 || leafScores.length > 0;
  if (!hasData) return null;

  return (
    <div className={styles.panel} data-testid="rubric-breakdown">
      <ClusterGrid clusters={clusters} />
      <RepairBanner passes={repairPasses} />
      <LeafScoreList leafScores={leafScores} />
    </div>
  );
}
