"use client";

import { useState } from "react";
import type { TreeNode, IterationView, PrimitiveCallView } from "../../../hooks/use-rlm-run";
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
            <span className={styles.primitiveResult}>
              {truncate(c.result_summary, 200)}
            </span>
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
}: {
  node: TreeNode;
  iteration: IterationView | null;
  primitiveCalls: PrimitiveCallView[];
  paperMeta: string;
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
    return (
      <div className={styles.body}>
        {nowBlock || <p className={styles.noDetail}>no iteration detail</p>}
      </div>
    );
  }

  // baseline, declined-group, or unknown
  return (
    <div className={styles.body}>
      {nowBlock || <p className={styles.noDetail}>no iteration detail</p>}
    </div>
  );
}

// ── Main sidebar component ────────────────────────────────────────────────────

export function NodeDetailSidebar({
  node,
  iteration,
  primitiveCalls,
  paperMeta,
  projectId,
  chatMessages,
  onSendChat,
  chatSending = false,
}: NodeDetailSidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <aside
        className={styles.sidebarCollapsed}
        data-testid="node-detail-sidebar"
        aria-label="Node detail sidebar (collapsed)"
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

      {/* Kind-specific content */}
      <div className={styles.content}>
        {node ? (
          <SidebarBody
            node={node}
            iteration={iteration}
            primitiveCalls={primitiveCalls}
            paperMeta={paperMeta}
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
}
