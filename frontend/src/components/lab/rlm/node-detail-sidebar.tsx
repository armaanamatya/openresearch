"use client";

import { memo, useState, type CSSProperties } from "react";
import type { TreeNode, IterationView, PrimitiveCallView, SubRlmView, GpuPlan } from "../../../hooks/use-rlm-run";
import type { ChatMessage } from "../../../hooks/use-steering-chat";
import { SteeringChat } from "./steering-chat";
import styles from "./node-detail-sidebar.module.css";

export interface NodeDetailSidebarProps {
  node: TreeNode | null;
  iteration: IterationView | null;
  primitiveCalls: PrimitiveCallView[];
  paperMeta: string;
  projectId: string;
  chatMessages: ChatMessage[];
  onSendChat: (content: string) => Promise<void>;
  chatSending?: boolean;
  subRlms: SubRlmView[];
  iterationCount: number;
  candidatesProposed: number;
  candidatesPromoted: number;
  /**
   * Resolved GPU provisioning plan from the gpu_resolved SSE event, or null
   * before resolve_gpu_requirements completes.  When present a small badge is
   * rendered beneath the aggregate-counters strip.
   */
  gpuPlan?: GpuPlan | null;
  /** Lift collapsed state to parent (optional — falls back to internal state). */
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
  style?: CSSProperties;
}

const COMPREHENSION_PRIMITIVES = new Set([
  "understand_section",
  "extract_hyperparameters",
]);
const ENVIRONMENT_PRIMITIVES = new Set([
  "detect_environment",
  "build_environment",
]);

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + "…" : s;
}

function formatDuration(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Score threshold labels — mirrors sse_bridge.py thresholds (≥0.7 pass, ≥0.4 partial). */
function rubricStatusLabel(score: number): { label: string; className: string } {
  if (score >= 0.7) return { label: "pass", className: styles.chipPass };
  if (score >= 0.4) return { label: "partial", className: styles.chipPartial };
  return { label: "fail", className: styles.chipFail };
}

/** Kind badge label. */
function kindLabel(kind: TreeNode["kind"]): string {
  switch (kind) {
    case "paper": return "paper";
    case "work": return "work";
    case "baseline": return "baseline";
    case "candidate": return "candidate";
    case "subrlm": return "sub-rlm";
    case "declined-group": return "declined";
    default: return kind;
  }
}

// ── Kind-specific content panels ──────────────────────────────────────────────

function PaperPanel({ paperMeta }: { paperMeta: string }) {
  // paperMeta is a JSON-serialised object or a plain string.
  let parsed: Record<string, unknown> | null = null;
  try {
    const v = JSON.parse(paperMeta);
    if (typeof v === "object" && v !== null) parsed = v as Record<string, unknown>;
  } catch {
    /* not JSON */
  }

  if (parsed) {
    return (
      <dl className={styles.metaList}>
        {Object.entries(parsed).map(([k, v]) => (
          <div key={k} className={styles.metaRow}>
            <dt className={styles.metaKey}>{k}</dt>
            <dd className={styles.metaVal}>{String(v ?? "—")}</dd>
          </div>
        ))}
      </dl>
    );
  }

  return <p className={styles.prose}>{paperMeta || "—"}</p>;
}

function PrimitiveList({
  calls,
  allowedPrimitives,
  emptyMsg,
}: {
  calls: PrimitiveCallView[];
  allowedPrimitives: Set<string>;
  emptyMsg: string;
}) {
  const filtered = calls.filter((c) => allowedPrimitives.has(c.primitive));
  if (filtered.length === 0) {
    return <p className={styles.noDetail}>{emptyMsg}</p>;
  }
  return (
    <ul className={styles.primitiveList}>
      {filtered.map((c, i) => (
        <li key={i} className={styles.primitiveItem}>
          <span className={styles.primitiveName}>{c.primitive}</span>
          <span
            className={
              c.status === "error"
                ? styles.primitiveStatusError
                : styles.primitiveStatusOk
            }
          >
            {c.status}
          </span>
          {c.result_summary && (
            c.result_summary.startsWith("[hint] ") ? (
              <span
                className={styles.primitiveHint}
                title="Harness hint: try rlm_query for this size"
              >
                <span className={styles.primitiveHintDot} aria-hidden="true" />
                {truncate(c.result_summary.slice("[hint] ".length), 200)}
              </span>
            ) : (
              <span className={styles.primitiveResult}>
                {truncate(c.result_summary, 200)}
              </span>
            )
          )}
        </li>
      ))}
    </ul>
  );
}

function CandidatePanel({
  node,
}: {
  node: TreeNode;
}) {
  const { candidate, rubricDelta } = node;
  if (!candidate) return null;
  return (
    <div className={styles.candidateMeta}>
      <span className={styles.category}>{candidate.category}</span>
      <span className={styles.description}>{candidate.description}</span>
      {rubricDelta != null && (
        <span className={styles.rubricDelta}>
          {rubricDelta >= 0 ? "+" : ""}
          {rubricDelta.toFixed(2)} rubric
        </span>
      )}
    </div>
  );
}

// ── Sidebar body dispatcher ───────────────────────────────────────────────────

function SidebarBody({
  node,
  iteration,
  primitiveCalls,
  paperMeta,
  subRlms,
}: {
  node: TreeNode;
  iteration: IterationView | null;
  primitiveCalls: PrimitiveCallView[];
  paperMeta: string;
  subRlms: SubRlmView[];
}) {
  const { kind } = node;

  // "What it's doing right now" block — shown for all kinds when iteration is live.
  const nowBlock = iteration ? (
    <div className={styles.nowBlock}>
      <span className={styles.nowLabel}>now</span>
      <p className={styles.response}>{truncate(iteration.response, 300)}</p>
    </div>
  ) : null;

  if (kind === "paper") {
    return (
      <div className={styles.body}>
        <PaperPanel paperMeta={paperMeta} />
        {nowBlock}
      </div>
    );
  }

  if (kind === "work") {
    // Map work phase to appropriate primitive set.
    const isEnv = node.phase === "environment";
    const allowed = isEnv ? ENVIRONMENT_PRIMITIVES : COMPREHENSION_PRIMITIVES;
    const emptyMsg = isEnv
      ? "no detect_environment / build_environment calls yet"
      : "no understand_section / extract_hyperparameters calls yet";
    return (
      <div className={styles.body}>
        <PrimitiveList
          calls={primitiveCalls}
          allowedPrimitives={allowed}
          emptyMsg={emptyMsg}
        />
        {nowBlock}
      </div>
    );
  }

  if (kind === "candidate") {
    return (
      <div className={styles.body}>
        <CandidatePanel node={node} />
        {iteration ? (
          <div className={styles.iterationBody}>
            <p className={styles.response}>{truncate(iteration.response, 300)}</p>
          </div>
        ) : (
          <p className={styles.noDetail}>no iteration detail</p>
        )}
      </div>
    );
  }

  if (kind === "subrlm") {
    // Node id is "subrlm-N" (1-indexed). Extract N to index into subRlms[].
    const match = node.id.match(/^subrlm-(\d+)$/);
    const subRlmView = match ? subRlms[parseInt(match[1], 10) - 1] ?? null : null;

    return (
      <div className={styles.body}>
        {subRlmView ? (
          <>
            <dl className={styles.metaList}>
              <div className={styles.metaRow}>
                <dt className={styles.metaKey}>model</dt>
                <dd className={styles.metaVal}>{subRlmView.model}</dd>
              </div>
              <div className={styles.metaRow}>
                <dt className={styles.metaKey}>depth</dt>
                <dd className={styles.metaVal}>{subRlmView.depth}</dd>
              </div>
              <div className={styles.metaRow}>
                <dt className={styles.metaKey}>duration</dt>
                <dd className={styles.metaVal}>{formatDuration(subRlmView.duration_ms)}</dd>
              </div>
              {subRlmView.error && (
                <div className={styles.metaRow}>
                  <dt className={styles.metaKey}>error</dt>
                  <dd className={styles.metaValError}>{subRlmView.error}</dd>
                </div>
              )}
            </dl>
            <pre className={styles.promptPreview}>
              {truncate(subRlmView.prompt_preview, 400)}
            </pre>
          </>
        ) : (
          <p className={styles.noDetail}>no sub-RLM detail</p>
        )}
        {nowBlock}
      </div>
    );
  }

  if (kind === "baseline") {
    const score = node.rubricScore ?? null;
    return (
      <div className={styles.body}>
        {score !== null ? (
          <div className={styles.baselineMeta}>
            <span className={styles.baselineScore}>
              Rubric score: {score.toFixed(2)}
            </span>
            <span className={rubricStatusLabel(score).className}>
              {rubricStatusLabel(score).label}
            </span>
          </div>
        ) : (
          <p className={styles.noDetail}>no rubric score yet</p>
        )}
        {nowBlock}
      </div>
    );
  }

  // declined-group, or unknown
  return (
    <div className={styles.body}>
      {nowBlock || <p className={styles.noDetail}>no iteration detail</p>}
    </div>
  );
}

// ── Main sidebar component ────────────────────────────────────────────────────

export const NodeDetailSidebar = memo(function NodeDetailSidebar({
  node,
  iteration,
  primitiveCalls,
  paperMeta,
  projectId,
  chatMessages,
  onSendChat,
  chatSending = false,
  subRlms,
  iterationCount,
  candidatesProposed,
  candidatesPromoted,
  gpuPlan = null,
  collapsed: collapsedProp,
  onCollapsedChange,
  style,
}: NodeDetailSidebarProps) {
  const [collapsedInternal, setCollapsedInternal] = useState(false);
  const collapsed = collapsedProp ?? collapsedInternal;
  function setCollapsed(next: boolean) {
    setCollapsedInternal(next);
    onCollapsedChange?.(next);
  }

  if (collapsed) {
    return (
      <aside
        className={styles.sidebarCollapsed}
        data-testid="node-detail-sidebar"
        aria-label="Node detail sidebar (collapsed)"
        style={undefined}
      >
        <button
          type="button"
          className={styles.toggleBtn}
          onClick={() => setCollapsed(false)}
          aria-label="Expand node detail sidebar"
          title="Expand"
        >
          {"<<"}
        </button>
      </aside>
    );
  }

  return (
    <aside
      className={styles.sidebar}
      style={style}
      data-testid="node-detail-sidebar"
      aria-label="Node detail sidebar"
    >
      {/* Header */}
      <div className={styles.header}>
        <button
          type="button"
          className={styles.toggleBtn}
          onClick={() => setCollapsed(true)}
          aria-label="Collapse node detail sidebar"
          title="Collapse"
        >
          {">>"}
        </button>
        {node ? (
          <div className={styles.identity}>
            <span className={styles.title}>{node.title}</span>
            <span className={styles.kindBadge}>{kindLabel(node.kind)}</span>
            {node.round != null && (
              <span className={styles.roundBadge}>round {node.round}</span>
            )}
          </div>
        ) : (
          <span className={styles.noSelection}>no node selected</span>
        )}
      </div>

      {/* Aggregate counters — always visible */}
      <div className={styles.countersStrip} data-testid="aggregate-counters">
        <div className={styles.counterTile}>
          <span className={styles.counterNum}>{iterationCount}</span>
          <span className={styles.counterLabel}>iterations</span>
        </div>
        <div className={styles.counterTile}>
          <span className={styles.counterNum}>{primitiveCalls.length}</span>
          <span className={styles.counterLabel}>primitive calls</span>
        </div>
        <div className={styles.counterTile}>
          <span className={styles.counterNum}>{candidatesProposed}</span>
          <span className={styles.counterLabel}>proposed</span>
        </div>
        <div className={styles.counterTile}>
          <span className={styles.counterNum}>{candidatesPromoted}</span>
          <span className={styles.counterLabel}>promoted</span>
        </div>
      </div>

      {/* GPU plan badge — visible once resolve_gpu_requirements completes */}
      {gpuPlan && (
        <div className={styles.gpuPlanBadge} data-testid="gpu-plan-badge" data-source={gpuPlan.source}>
          <span className={styles.gpuPlanSku}>{gpuPlan.short_name}</span>
          <span className={styles.gpuPlanVram}>{gpuPlan.vram_gb} GB</span>
          {gpuPlan.gpu_count > 1 && (
            <span className={styles.gpuPlanCount}>×{gpuPlan.gpu_count}</span>
          )}
          <span className={styles.gpuPlanCost}>${gpuPlan.total_usd_per_hr.toFixed(2)}/hr</span>
          {gpuPlan.source === "fallback" && (
            <span
              className={styles.gpuPlanFallback}
              title={gpuPlan.requirements.reasoning}
            >
              fallback
            </span>
          )}
        </div>
      )}

      {/* Kind-specific content */}
      <div className={styles.content}>
        {node ? (
          <SidebarBody
            node={node}
            iteration={iteration}
            primitiveCalls={primitiveCalls}
            paperMeta={paperMeta}
            subRlms={subRlms}
          />
        ) : (
          <p className={styles.noDetail}>click a node to see details</p>
        )}
      </div>

      {/* Chat panel */}
      <SteeringChat
        projectId={projectId}
        messages={chatMessages}
        onSend={onSendChat}
        disabled={chatSending}
      />
    </aside>
  );
});
