"use client";

import { usePresentationMode } from "@/lib/presentation-mode";
import type { LaidOutNode } from "@/lib/pipeline/layout";
import { ICONS, type IconKey } from "./icons";
import type { NodeState } from "./node-config";

import "./node-card.css";

export const NODE_W = 200;
export const NODE_H = 80;

export function NodeCard({
  node,
  onClick,
  selected,
  state,
  progress
}: {
  node: LaidOutNode;
  onClick: () => void;
  selected: boolean;
  state: NodeState;
  progress: number;
}) {
  const mode = usePresentationMode();
  const agentLabel = mode === "demo" ? node.demo_label : node.internal_label;

  const tones = {
    info: { icBg: "var(--info-soft)", icFg: "#3b48d1" },
    accent: { icBg: "var(--accent-soft)", icFg: "var(--accent-ink)" },
    hermes: { icBg: "var(--hermes-soft)", icFg: "var(--hermes)" },
    neutral: { icBg: "var(--chip)", icFg: "var(--muted)" }
  } as const;

  const tone = tones[node.tone];
  let borderColor = "var(--line)";
  let glow = "none";
  let opacity = 1;
  let background = "#fff";
  let showProgress = false;

  if (node.tone === "hermes") {
    background = "linear-gradient(180deg,#faf8ff,#fff)";
  }
  if (state === "running") {
    borderColor = node.tone === "hermes" ? "var(--hermes)" : "var(--accent)";
    glow =
      node.tone === "hermes"
        ? "0 0 0 4px rgba(124,92,255,.10), 0 12px 32px -16px rgba(124,92,255,.55)"
        : "0 0 0 4px rgba(22,178,92,.10), 0 12px 32px -16px rgba(22,178,92,.5)";
    showProgress = true;
  }
  if (state === "done") {
    borderColor = "var(--line-2)";
  }
  if (state === "upcoming") {
    opacity = 0.4;
  }
  if (selected) {
    borderColor = "var(--ink)";
    glow = "0 0 0 4px rgba(14,14,16,.06), 0 16px 36px -18px rgba(14,14,16,.5)";
  }

  return (
    <div
      className={state === "upcoming" ? "" : "wf-pop"}
      data-node="1"
      onClick={onClick}
      style={{
        position: "absolute",
        left: node.x,
        top: node.y,
        width: NODE_W,
        height: NODE_H,
        background,
        border: `1px solid ${borderColor}`,
        borderRadius: 14,
        padding: "12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        boxShadow: glow,
        cursor: state === "upcoming" ? "default" : "pointer",
        transition: "border-color .25s ease, box-shadow .25s ease, opacity .3s ease, transform .25s ease",
        opacity,
        transform: selected ? "translateY(-2px) scale(1.015)" : "scale(1)",
        zIndex: selected ? 5 : state === "running" ? 3 : 2
      }}
    >
      <div className="node-head">
        <div className="node-icon" style={{ background: tone.icBg, color: tone.icFg }}>
          {ICONS[node.icon as IconKey]}
          {state === "running" ? <span className="wf-ring node-ring" /> : null}
        </div>
        <div className="node-copy">
          <div className="node-agent">{agentLabel}</div>
          <div className="node-step">{node.step}</div>
        </div>
        {state === "done" ? (
          <div className="node-check">
            <svg width="10" height="10" viewBox="0 0 16 16" aria-hidden="true">
              <path
                d="M3 8.5l3 3 7-7"
                stroke="currentColor"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        ) : null}
      </div>
      {showProgress ? (
        <div className="node-progress">
          <div
            className="wf-bar"
            style={{
              background: node.tone === "hermes" ? "var(--hermes)" : "var(--accent)",
              width: `${Math.max(5, Math.round(progress * 100))}%`,
              transition: "width 0.4s ease"
            }}
          />
        </div>
      ) : null}
    </div>
  );
}
