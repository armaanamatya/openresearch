"use client";

import { useState } from "react";

import type { LeafDetail, RlmRunState } from "../../../hooks/use-rlm-run";
import styles from "./rubric-detail.module.css";

interface RubricDetailProps {
  rubric: RlmRunState["rubric"];
  /** Default-expanded state of the whole panel. Defaults to collapsed so the
   *  ConstellationCanvas is never squeezed on first render. */
  defaultOpen?: boolean;
}

/** Glyph for an area status (matches RubricStrip's vocabulary). */
function areaGlyph(status: "pass" | "partial" | "fail"): string {
  return status === "pass" ? "✓" : status === "partial" ? "◐" : "✗";
}

/** Glyph for a leaf status — adds an explicit "unavailable" marker. */
function leafGlyph(status: LeafDetail["status"]): string {
  switch (status) {
    case "pass":
      return "✓";
    case "partial":
      return "◐";
    case "fail":
      return "✗";
    case "unavailable":
    default:
      return "·";
  }
}

function leafStatusClass(
  status: LeafDetail["status"],
  s: Record<string, string>
): string {
  switch (status) {
    case "pass":
      return s.glyphPass;
    case "partial":
      return s.glyphPartial;
    case "fail":
      return s.glyphFail;
    case "unavailable":
    default:
      return s.glyphUnavailable;
  }
}

function areaStatusClass(
  status: "pass" | "partial" | "fail",
  s: Record<string, string>
): string {
  return status === "pass"
    ? s.glyphPass
    : status === "partial"
    ? s.glyphPartial
    : s.glyphFail;
}

/**
 * LeafRow — a single criterion. Shows status glyph + label + score + the
 * justification (`why`). Long `why` text is clamped to one line and expands on
 * click to the full text.
 */
function LeafRow({ leaf }: { leaf: LeafDetail }) {
  const [expanded, setExpanded] = useState(false);
  const hasWhy = Boolean(leaf.why && leaf.why.trim());
  return (
    <li className={styles.leafRow} data-status={leaf.status}>
      <span
        className={`${styles.leafGlyph} ${leafStatusClass(leaf.status, styles)}`}
        aria-hidden="true"
      >
        {leafGlyph(leaf.status)}
      </span>
      <div className={styles.leafBody}>
        <div className={styles.leafHead}>
          <span className={styles.leafLabel} title={leaf.label || leaf.id}>
            {leaf.label || leaf.id}
          </span>
          <span className={styles.leafScore}>
            {leaf.status === "unavailable" || leaf.score == null
              ? "—"
              : leaf.score.toFixed(2)}
          </span>
        </div>
        {hasWhy && (
          <button
            type="button"
            className={styles.leafWhy}
            data-expanded={expanded ? "true" : "false"}
            title={expanded ? undefined : leaf.why}
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            {leaf.why}
          </button>
        )}
      </div>
    </li>
  );
}

/**
 * AreaRow — one collapsible row per rubric AREA. Header shows the area name,
 * status glyph, and score/weight. Expanding lists the area's leaves (when the
 * enriched event provided them) or an honest "leaf detail not available" note.
 */
function AreaRow({
  area,
  defaultOpen,
}: {
  area: RlmRunState["rubric"]["areas"][number];
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const leaves = area.leaves;
  return (
    <li className={styles.areaRow} data-status={area.status}>
      <button
        type="button"
        className={styles.areaHeader}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        data-testid="rubric-detail-area-header"
      >
        <span className={styles.disclosure} aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
        <span
          className={`${styles.areaGlyph} ${areaStatusClass(area.status, styles)}`}
          aria-hidden="true"
        >
          {areaGlyph(area.status)}
        </span>
        <span className={styles.areaName} title={area.area || "—"}>
          {area.area || "—"}
        </span>
        <span className={styles.areaMeta}>
          {area.score.toFixed(2)}
          <span className={styles.areaWeight}>
            {" · "}
            {(area.weight * 100).toFixed(0)}%
          </span>
        </span>
      </button>
      {open && (
        <div className={styles.areaExpanded}>
          {leaves && leaves.length > 0 ? (
            <ul className={styles.leafList}>
              {leaves.map((l, i) => (
                <LeafRow key={l.id || `${area.area}_leaf_${i}`} leaf={l} />
              ))}
            </ul>
          ) : (
            <p className={styles.unavailableNote}>
              Leaf detail not available for this iteration.
            </p>
          )}
        </div>
      )}
    </li>
  );
}

/**
 * RubricDetail — an expandable, clearly-labeled "Rubric breakdown" panel driven
 * by `state.rubric`. Surfaces which leaves fail, why, and the specific recent
 * errors — without fabricating data when the backend has not yet enriched the
 * rubric_score event.
 *
 * Sections:
 *  1. One collapsible row per AREA → expands to its leaves (or an honest note).
 *  2. "Recent errors" — the specific failures (kind + concise message).
 *
 * Layout: the whole panel is collapsible (default collapsed) and the expanded
 * body is height-bounded (max-height + overflow-y:auto) so it never occludes
 * the ConstellationCanvas. Mirrors the ScorecardPanel / RubricBreakdown idiom.
 */
export function RubricDetail({ rubric, defaultOpen = false }: RubricDetailProps) {
  const [open, setOpen] = useState(defaultOpen);
  const { areas, weakLeaves } = rubric;
  // recentErrors is optional on the type (legacy/fixture states may omit it);
  // normalize to [] so the rest of the component treats it uniformly.
  const recentErrors = rubric.recentErrors ?? [];

  // Nothing scored AND nothing failed → the breakdown has nothing honest to
  // show; render nothing so we don't add an empty band above the graph.
  const hasAreas = areas.length > 0;
  const hasErrors = recentErrors.length > 0;
  if (!hasAreas && !hasErrors) return null;

  const failCount = areas.filter((a) => a.status === "fail").length;
  const partialCount = areas.filter((a) => a.status === "partial").length;
  // Leaves default-expand only on the failing/partial areas so the user lands
  // on what needs attention without manually opening each row.
  const areaDefaultOpen = (status: "pass" | "partial" | "fail") =>
    status === "fail" || status === "partial";

  return (
    <section
      className={styles.panel}
      data-testid="rubric-detail"
      aria-label="Rubric breakdown"
    >
      <button
        type="button"
        className={styles.panelHeader}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        data-testid="rubric-detail-toggle"
      >
        <span className={styles.disclosure} aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
        <span className={styles.panelTitle}>Rubric breakdown</span>
        <span className={styles.panelSummary}>
          {hasAreas ? (
            <>
              {areas.length} {areas.length === 1 ? "area" : "areas"}
              {failCount > 0 && (
                <span className={styles.summaryFail}>{` · ${failCount} failing`}</span>
              )}
              {partialCount > 0 && (
                <span className={styles.summaryPartial}>{` · ${partialCount} partial`}</span>
              )}
            </>
          ) : null}
          {hasErrors && (
            <span className={styles.summaryFail}>
              {`${hasAreas ? " · " : ""}${recentErrors.length} recent error${
                recentErrors.length === 1 ? "" : "s"
              }`}
            </span>
          )}
        </span>
      </button>

      {open && (
        <div className={styles.body}>
          {hasAreas ? (
            <ul className={styles.areaList}>
              {areas.map((a, i) => (
                <AreaRow
                  key={a.area || `__area_${i}`}
                  area={a}
                  defaultOpen={areaDefaultOpen(a.status)}
                />
              ))}
            </ul>
          ) : (
            <p className={styles.unavailableNote}>
              No scored areas yet — the breakdown populates after the first
              verification.
            </p>
          )}

          {/* Weakest-leaf digest — the grader's "what to fix next", when present. */}
          {weakLeaves && weakLeaves.length > 0 && (
            <div className={styles.subSection}>
              <p className={styles.subSectionLabel}>Weakest criteria</p>
              <ul className={styles.weakList}>
                {weakLeaves.map((w, i) => (
                  <li key={w.id || `__weak_${i}`} className={styles.weakRow}>
                    <span className={styles.weakScore}>
                      {w.score == null ? "—" : w.score.toFixed(2)}
                    </span>
                    <span className={styles.weakArea}>{w.area || "—"}</span>
                    <span className={styles.weakWhy} title={w.why}>
                      {w.why || w.id}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Recent errors — the specific failures, clearly labeled. */}
          <div className={styles.subSection}>
            <p className={styles.subSectionLabel}>Recent errors</p>
            {hasErrors ? (
              <ul className={styles.errorList}>
                {recentErrors
                  .slice()
                  .reverse()
                  .map((e, i) => (
                    <li
                      key={`${e.kind}_${e.iteration}_${i}`}
                      className={styles.errorRow}
                      data-testid="rubric-detail-error"
                    >
                      <span className={styles.errorKind}>{e.kind}</span>
                      <span className={styles.errorMessage} title={e.message}>
                        {e.message}
                      </span>
                    </li>
                  ))}
              </ul>
            ) : (
              <p className={styles.unavailableNote}>No errors recorded.</p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
