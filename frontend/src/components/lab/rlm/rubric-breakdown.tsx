"use client";

import { useState } from "react";

import type { DemoClusterStatus, DemoLeafScore, DemoRepairPass } from "@/lib/demo/demo-run-types";
import { useRdrArtifacts } from "@/hooks/use-rdr-artifacts";
import { CollapsiblePanel } from "./collapsible-panel";
import styles from "./rubric-breakdown.module.css";

interface RubricBreakdownProps {
  projectId: string;
  isActive: boolean;
  /**
   * Per-model metrics from the latest experiment_completed event.
   * When present and leaf IDs have a model-name suffix, shows a stacked
   * mini-bar per model next to each leaf row.
   * Lane γ: per-model multi-model UI (2026-05-23).
   */
  perModelMetrics?: Record<string, Record<string, number>> | null;
}

/**
 * Detect whether a leaf id carries a model-name suffix from the per_model dict.
 * Leaf ids like "alfworld_qwen2_5_3b" match a model key "qwen2_5_3b".
 * Returns the matched model key or null.
 */
function leafModelSuffix(
  leafId: string,
  modelKeys: string[]
): string | null {
  for (const key of modelKeys) {
    if (leafId.endsWith("_" + key) || leafId === key) return key;
  }
  return null;
}

/**
 * PerModelMiniBar — a stacked row of tiny bars, one per model, showing
 * the value of a given metric across models.  Only rendered when
 * perModelMetrics is non-null and has ≥2 model variants.
 */
function PerModelMiniBar({
  metricKey,
  perModelMetrics,
}: {
  metricKey: string;
  perModelMetrics: Record<string, Record<string, number>>;
}) {
  const modelKeys = Object.keys(perModelMetrics).sort();
  // Find entries that have the given metric key.
  const entries = modelKeys
    .map((m) => ({ model: m, val: perModelMetrics[m][metricKey] }))
    .filter((e) => typeof e.val === "number");
  if (entries.length < 2) return null;

  const maxVal = Math.max(...entries.map((e) => e.val), 1e-9);

  return (
    <div
      className={styles.perModelMiniBarGroup}
      data-testid="per-model-mini-bar"
      title={entries.map((e) => `${e.model}: ${e.val.toFixed(3)}`).join(" | ")}
    >
      {entries.map((e) => {
        const pct = Math.min(1, Math.max(0, e.val / maxVal)) * 100;
        return (
          <div key={e.model} className={styles.perModelMiniBarRow}>
            <span className={styles.perModelMiniBarLabel}>{e.model}</span>
            <div className={styles.perModelMiniBarTrack}>
              <div
                className={styles.perModelMiniBarFill}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className={styles.perModelMiniBarVal}>{e.val.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
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

function LeafRow({
  leaf,
  perModelMetrics,
}: {
  leaf: DemoLeafScore;
  perModelMetrics?: Record<string, Record<string, number>> | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(Math.min(1, Math.max(0, leaf.score)) * 100);

  // Detect if this leaf's id has a model-name suffix that matches one of the
  // per_model keys.  If so, derive a metric key by stripping the suffix.
  const modelKeys = perModelMetrics ? Object.keys(perModelMetrics).sort() : [];
  const matchedSuffix = perModelMetrics
    ? leafModelSuffix(leaf.id, modelKeys)
    : null;
  // Derive the metric key: leaf.id with the model suffix removed.
  // e.g. "alfworld_qwen2_5_3b" with suffix "qwen2_5_3b" → "alfworld"
  const metricKey =
    matchedSuffix && leaf.id.endsWith("_" + matchedSuffix)
      ? leaf.id.slice(0, leaf.id.length - matchedSuffix.length - 1)
      : null;
  // Whether to show mini-bars: perModelMetrics must have ≥2 models AND the
  // derived metric key must exist in at least 2 model dicts.
  const showMiniBar =
    perModelMetrics &&
    modelKeys.length >= 2 &&
    metricKey !== null &&
    modelKeys.filter(
      (m) => typeof perModelMetrics[m][metricKey] === "number"
    ).length >= 2;

  return (
    <div className={styles.leafRowWrapper}>
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
      {showMiniBar && perModelMetrics && metricKey && (
        <PerModelMiniBar
          metricKey={metricKey}
          perModelMetrics={perModelMetrics}
        />
      )}
    </div>
  );
}

function LeafScoreList({
  leafScores,
  overallScore,
  perModelMetrics,
}: {
  leafScores: DemoLeafScore[];
  overallScore?: number;
  perModelMetrics?: Record<string, Record<string, number>> | null;
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
          <LeafRow key={l.id} leaf={l} perModelMetrics={perModelMetrics} />
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
export function RubricBreakdown({ projectId, isActive, perModelMetrics }: RubricBreakdownProps) {
  const { clusters, leafScores, repairPasses } = useRdrArtifacts(projectId, isActive);

  const hasAny = clusters.length > 0 || leafScores.length > 0 || repairPasses.length > 0;
  if (!hasAny) return null;

  const failedClusters = clusters.filter((c) => c.failed === true).length;
  const summaryParts: string[] = [];
  if (leafScores.length > 0)
    summaryParts.push(`${leafScores.length} ${leafScores.length === 1 ? "leaf" : "leaves"}`);
  if (clusters.length > 0)
    summaryParts.push(`${clusters.length} ${clusters.length === 1 ? "cluster" : "clusters"}`);
  if (failedClusters > 0) summaryParts.push(`${failedClusters} failed`);
  const summary = summaryParts.join(" · ");

  return (
    <CollapsiblePanel storageKey="rubric-breakdown" title="Rubric breakdown" summary={summary}>
      <div className={styles.panel} data-testid="rubric-breakdown">
        <ClusterGrid clusters={clusters} />
        <RepairBanner passes={repairPasses} />
        <LeafScoreList leafScores={leafScores} perModelMetrics={perModelMetrics} />
      </div>
    </CollapsiblePanel>
  );
}
