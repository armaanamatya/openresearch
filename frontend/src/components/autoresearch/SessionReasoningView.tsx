"use client";

import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";

import type { RlmDashboardEvent } from "@/lib/events/rlm-events";
import {
  useRlmRunBatched,
  type IterationView,
  type PrimitiveCallView,
  type RlmRunState,
  type RubricArea,
} from "@/hooks/use-rlm-run";
import { useSteeringChat, type ChatMessage } from "@/hooks/use-steering-chat";
import { ReasoningChip, type ReasoningChipStatus } from "./ui/ReasoningChip";
import styles from "./SessionReasoningView.module.css";

export interface SessionReasoningViewProps {
  /** The run's project id — the docked steering chat's endpoint and the
   * "Session <id>" pill shown at the top of the log. */
  runId: string;
  /**
   * The already-sanitized RLM event stream for this run. This component
   * does NOT own a live subscription — the parent page subscribes (via
   * use-run.ts's EventSource/dashboardEvents, filtered through isRlmEvent)
   * and hands the resulting array down here, so it renders identically
   * whether fed a growing live array or a replayed fixture (mirrors how
   * the dark lab's RlmLab receives `events` as a prop rather than
   * subscribing itself).
   */
  events: RlmDashboardEvent[];
  className?: string;
}

// ─── Corpus-safe rendering helpers ─────────────────────────────────────────
//
// Every helper below reads ONLY the named, already-sanitized fields off
// IterationView / PrimitiveCallView (response, code_blocks metadata,
// primitive, status) — never a raw event object dump. That is what makes
// the corpus invariant hold structurally: there is no code path here that
// could surface a field the SSE egress sanitizer didn't already intend to
// ship (see CLAUDE.md's sse_bridge.sanitize_iteration).

function chipStatusFor(status: PrimitiveCallView["status"]): ReasoningChipStatus {
  // "error" has no dedicated chip visual (per T9b's own note on
  // ReasoningChip) — the honest mapping is "active" while in flight,
  // "done" once settled, regardless of success/failure.
  return status === "start" ? "active" : "done";
}

function humanizePrimitiveLabel(primitive: string): string {
  const spaced = primitive.replace(/_/g, " ").trim();
  if (!spaced) return primitive;
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

interface LogBlock {
  key: string;
  /** The repl_iteration for this position, or null when this block is a
   * calls-only trailing group (see buildLogBlocks). */
  iteration: IterationView | null;
  calls: PrimitiveCallView[];
}

/**
 * Groups primitiveCalls by iteration number and interleaves them with the
 * dense `iterations` array in stream order: for each completed iteration,
 * its own response paragraph is followed by every primitive_call recorded
 * against that iteration number (both the "start" and terminal entries —
 * rendered as two chips transitioning active -> done, an honest log rather
 * than a collapsed summary). Calls whose iteration hasn't produced a
 * repl_iteration yet (live in-progress work) or that carry no iteration
 * number at all render as trailing calls-only blocks, in iteration order.
 */
function buildLogBlocks(state: RlmRunState): LogBlock[] {
  const byIteration = new Map<number, PrimitiveCallView[]>();
  const unassigned: PrimitiveCallView[] = [];
  for (const call of state.primitiveCalls) {
    if (call.iteration == null) {
      unassigned.push(call);
      continue;
    }
    const existing = byIteration.get(call.iteration);
    if (existing) {
      existing.push(call);
    } else {
      byIteration.set(call.iteration, [call]);
    }
  }

  const blocks: LogBlock[] = [];
  for (const iteration of state.iterations) {
    const calls = byIteration.get(iteration.iteration) ?? [];
    blocks.push({ key: `iter-${iteration.iteration}`, iteration, calls });
    byIteration.delete(iteration.iteration);
  }
  const leftoverIterations = [...byIteration.keys()].sort((a, b) => a - b);
  for (const iterationNumber of leftoverIterations) {
    blocks.push({
      key: `calls-${iterationNumber}`,
      iteration: null,
      calls: byIteration.get(iterationNumber)!,
    });
  }
  if (unassigned.length > 0) {
    blocks.push({ key: "unassigned", iteration: null, calls: unassigned });
  }
  return blocks;
}

// Matches an http(s) URL or a UUID-shaped token embedded in already-sanitized
// reasoning prose, so they can render as a maroon link / mono id pill
// (alphaXiv screen `_5`) instead of plain text. This re-renders the SAME
// sanitized string that already reached the client — it never reaches
// around to a raw event field.
const TOKEN_RE = /(https?:\/\/[^\s)]+)|\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b/gi;

function renderReasoningText(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  TOKEN_RE.lastIndex = 0;
  let lastIndex = 0;
  let tokenIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = TOKEN_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const [full, url, id] = match;
    if (url) {
      nodes.push(
        <a
          key={`${keyPrefix}-tok-${tokenIndex}`}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className={styles.link}
        >
          {url}
        </a>
      );
    } else if (id) {
      nodes.push(
        <span key={`${keyPrefix}-tok-${tokenIndex}`} className={styles.idPill}>
          {id}
        </span>
      );
    }
    lastIndex = match.index + full.length;
    tokenIndex += 1;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

const AREA_STATUS_GLYPH: Record<RubricArea["status"], string> = {
  pass: "✓",
  partial: "◐",
  fail: "✗",
};

const AREA_STATUS_CLASS: Record<RubricArea["status"], string> = {
  pass: styles.rubricPass,
  partial: styles.rubricPartial,
  fail: styles.rubricFail,
};

/** Compact score/leaf strip — a light-maroon equivalent of the dark lab's
 * RubricStrip (rlm-lab/rubric-strip.tsx), built fresh rather than editing
 * it. Honesty rule preserved: an unscored run shows "—", never a fabricated
 * number; the headline is the best-of-run score, matching the dark lab's
 * own "a failed final attempt should not erase an earlier result" rule. */
function CompactRubricStrip({ rubric }: { rubric: RlmRunState["rubric"] }) {
  const headline = rubric.best ?? rubric.current;
  const isScored = headline !== null && headline !== undefined;

  return (
    <div className={styles.rubricStrip} role="region" aria-label="Rubric score">
      <span className={isScored ? styles.rubricScore : styles.rubricScorePlaceholder}>
        {isScored ? headline.toFixed(2) : "—"}
      </span>
      {rubric.target !== null && (
        <span className={styles.rubricTarget}>{`/ ${rubric.target.toFixed(2)} target`}</span>
      )}
      {rubric.areas.length > 0 && (
        <ul className={styles.rubricAreas}>
          {rubric.areas.map((area, i) => (
            <li
              key={area.area || `area-${i}`}
              className={[styles.rubricArea, AREA_STATUS_CLASS[area.status]].join(" ")}
              title={`${area.area || "—"}: ${area.status}`}
            >
              <span aria-hidden="true">{AREA_STATUS_GLYPH[area.status]}</span>
              {area.area && <span className={styles.rubricAreaName}>{area.area}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface DockedSteeringInputProps {
  messages: ChatMessage[];
  onSend: (content: string) => Promise<void>;
  sending: boolean;
}

/** Docked steering input (alphaXiv screen `_5`'s bottom chat dock) — reuses
 * useSteeringChat's optimistic-message contract, a lighter/compact sibling
 * of the dark lab's SteeringChat (rlm/steering-chat.tsx), built fresh
 * rather than editing it. */
function DockedSteeringInput({ messages, onSend, sending }: DockedSteeringInputProps) {
  const [draft, setDraft] = useState("");
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || sending) return;
    setDraft("");
    await onSend(content);
  }

  return (
    <div className={styles.dock} data-testid="session-steering-dock">
      {messages.length > 0 && (
        <div className={styles.dockLog} ref={logRef} role="log" aria-live="polite">
          {messages.map((m) => (
            <div
              key={m.id}
              className={m.role === "user" ? styles.dockMsgUser : styles.dockMsgAssistant}
              data-optimistic={m.optimistic ? "true" : undefined}
            >
              <span className={styles.dockMsgRole}>{m.role === "user" ? "you" : "agent"}</span>
              <span>{m.content}</span>
            </div>
          ))}
        </div>
      )}
      <form className={styles.dockInputRow} onSubmit={handleSubmit}>
        <input
          className={styles.dockInput}
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Send a message…"
          disabled={sending}
          aria-label="Steer the session"
        />
        <button type="submit" className={styles.dockSendBtn} disabled={sending || !draft.trim()}>
          {sending ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}

/**
 * SessionReasoningView — the live agentic-reasoning session log (alphaXiv
 * screen `_5`). Consumes the SAME folded RlmRunState the dark lab uses
 * (via useRlmRunBatched), rendering ONLY its sanitized fields: repl_iteration
 * responses as reasoning prose (with inline maroon links / mono id pills),
 * and primitive_call entries as ReasoningChip rows grouped per iteration.
 * Never reaches around the hook state into a raw event's corpus-bearing
 * fields (context/locals/prompt) — see the corpus-safe helpers above.
 */
export function SessionReasoningView({ runId, events, className }: SessionReasoningViewProps) {
  // Lazy-init folds a full array synchronously on the first render (fixture
  // replay / tests); the effect below feeds only the new tail as `events`
  // grows, mirroring RlmLab's exact incremental-feed pattern so a live SSE
  // stream and a replayed fixture render identically.
  const { state, addEvent, reset } = useRlmRunBatched(events);
  const fedCountRef = useRef(events.length);
  useEffect(() => {
    const start = fedCountRef.current;
    const end = events.length;
    if (end < start) {
      reset();
      fedCountRef.current = 0;
      for (let i = 0; i < end; i++) addEvent(events[i]);
      fedCountRef.current = end;
    } else {
      for (let i = start; i < end; i++) addEvent(events[i]);
      fedCountRef.current = end;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const { messages, send, sending } = useSteeringChat(runId, events);

  const blocks = useMemo(() => buildLogBlocks(state), [state]);

  const logRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [blocks.length]);

  const classes = [styles.view, className].filter(Boolean).join(" ");

  return (
    <div className={classes} data-testid="session-reasoning-view">
      <div className={styles.log} ref={logRef}>
        <p className={styles.sessionMeta}>
          Session <span className={styles.idPill}>{runId}</span>
        </p>
        <CompactRubricStrip rubric={state.rubric} />
        {blocks.length === 0 ? (
          <p className={styles.empty}>Waiting for the agent to start reasoning…</p>
        ) : (
          blocks.map((block) => (
            <div key={block.key} className={styles.block}>
              {block.iteration && (
                <p className={styles.reasoning}>
                  {renderReasoningText(block.iteration.response, block.key)}
                </p>
              )}
              {block.calls.length > 0 && (
                <div className={styles.chipGroup}>
                  {block.calls.map((call, i) => (
                    <ReasoningChip
                      key={`${block.key}-call-${i}`}
                      label={humanizePrimitiveLabel(call.primitive)}
                      status={chipStatusFor(call.status)}
                    />
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
      <DockedSteeringInput messages={messages} onSend={send} sending={sending} />
    </div>
  );
}
