"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { useRun } from "@/hooks/use-run";
import { useBudgetEstimate } from "@/hooks/use-budget-estimate";
import type { PaperBudgetEstimate, RecipeEstimate } from "@/components/lab/budget/budget-panel";
import { readUserPrefs } from "@/lib/user-prefs";
import { isDevSessionSignedIn } from "@/lib/autoresearch/dev-session";
import { FALLBACK_PAPER_TITLE } from "@/lib/autoresearch/paper-meta";
import { Card } from "./ui/Card";
import { Eyebrow } from "./ui/Eyebrow";
import { PrimaryButton } from "./ui/PrimaryButton";
import { RepoField } from "./ui/RepoField";
import styles from "./RepoConfirm.module.css";

export interface RepoConfirmProps {
  /** The paper's arXiv id, e.g. "2605.15155". */
  arxivId: string;
  /** Real paper title, once a metadata source supplies one (see PaperLanding). */
  title?: string;
  /** Best-effort repo suggestion from GET /papers/{id}/repo (T3b), resolved
   * server-side and handed down as a plain string — "" when none was found
   * (fail-soft: the field just renders blank and editable). */
  initialRepoUrl: string;
  className?: string;
}

// Display-only estimate shown in the launch cost-guard dialog. The frontend
// never reads backend config directly (no client-side env access to it) —
// this mirrors the autonomous profile's actual enforced cap
// (configs/autonomous_reproduction_run_spec.json ->
// OPENRESEARCH_MAX_RUN_GPU_USD: "10.0") as a static display string. Keep
// this in sync if that profile value ever changes. This is a hard CEILING,
// not a prediction — the paper-specific estimate below it (once loaded) is
// the actual prediction.
const ESTIMATED_GPU_CAP_DISPLAY = "$10";

/** 1–8, matching the backend's StartRunRequest/StartArxivRunRequest.gpu_count
 * bound (``Field(default=None, ge=1, le=8)``). `null` == "Auto" (existing
 * auto-resolution — omit the field entirely so the backend's own GPU
 * resolver picks the SKU/count, unchanged from today's behavior). */
const GPU_COUNT_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8] as const;

function fmtUsd(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v < 0.01) return `$${v.toFixed(4)}`;
  return `$${v.toFixed(2)}`;
}

function fmtHours(h: number): string {
  if (!Number.isFinite(h)) return "—";
  if (h < 1) return `${Math.round(h * 60)} min`;
  return `${h.toFixed(1)}h`;
}

/** Picks the recipe the autonomous profile actually runs — "strict" (full
 * fidelity) when present, else whichever recipe the estimator returned
 * first. Autonomous launches don't expose a recipe picker (unlike the lab's
 * BudgetPanel), so there is exactly one number to show here. */
function pickRecipe(estimate: PaperBudgetEstimate): RecipeEstimate | null {
  const keys = Object.keys(estimate.recipes ?? {});
  if (keys.length === 0) return null;
  return estimate.recipes.strict ?? estimate.recipes[keys[0]];
}

/** Best-effort classification of a launch failure so the dialog can show a
 * more useful message + a retry affordance for the (usually transient) case
 * of GPU capacity/quota being unavailable right now, vs. any other error.
 * Pure string matching on the hook's already-surfaced error message — no
 * new backend error taxonomy, so this degrades to "generic" harmlessly if
 * the wording ever changes upstream. */
function isQuotaOrCapacityError(message: string): boolean {
  const lower = message.toLowerCase();
  return /quota|capacity|no (gpu|sku|instance)s? available|out of stock|exhausted|unavailable.*gpu|gpu.*unavailable/.test(
    lower
  );
}

/**
 * Repo-confirm screen (alphaXiv `_4`) — shown at /abs/<arxivId>?confirm=1
 * after the stubbed sign-in step. Lets the user confirm/replace the
 * auto-resolved repository and pick a GPU count, then gates the actual
 * launch behind a cost-guard dialog (D4): shows a real paper-specific cost
 * estimate alongside the autonomous profile's hard GPU cap, and requires
 * the dev session flag before it will call the launcher.
 */
export function RepoConfirm({ arxivId, title, initialRepoUrl, className }: RepoConfirmProps) {
  const router = useRouter();
  const [repoUrl, setRepoUrl] = useState(initialRepoUrl);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [gpuCount, setGpuCount] = useState<number | null>(null);
  const gpuSelectId = useId();
  const hasRepo = repoUrl.trim().length > 0;

  const {
    estimate,
    loading: estimateLoading,
    error: estimateError,
    estimateFromArxiv,
  } = useBudgetEstimate();

  // Preload the real per-paper estimate as soon as this screen is reachable
  // (i.e. once the stubbed sign-in flag is set) so the cost-guard dialog
  // already has a number by the time the user opens it. Gated on sign-in —
  // matching the launcher's own gate — so an unauthenticated visit never
  // fires a network request, only ever the static cap is shown.
  useEffect(() => {
    if (isDevSessionSignedIn()) {
      void estimateFromArxiv(arxivId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arxivId]);

  // Type note: `runOptions` is intentionally NOT annotated as
  // `ProviderRunOptions` here — passing an untyped variable (rather than an
  // inline object literal) to `useRun` sidesteps TS's excess-property check,
  // so this keeps compiling whether or not `ProviderRunOptions.gpuCount`
  // has landed yet in use-run.ts (owned elsewhere; this file must not edit
  // it). Once that field exists, the shapes match exactly and this is a
  // no-op — no follow-up change needed here.
  const runOptions = {
    autonomous: true,
    repoUrl: repoUrl.trim() || undefined,
    ...(gpuCount != null ? { gpuCount } : {}),
  };

  const { startArxivRun, busy, error } = useRun(null, runOptions);

  const handleConfirmLaunch = async () => {
    const model = readUserPrefs().model ?? "sonnet";
    const arxivUrl = `https://arxiv.org/abs/${arxivId}`;
    const runId = await startArxivRun(arxivUrl, model);
    if (runId) {
      router.push(`/sessions/${encodeURIComponent(runId)}`);
    }
  };

  return (
    <Card className={[styles.card, className].filter(Boolean).join(" ")}>
      <Eyebrow>{`arXiv ${arxivId}`}</Eyebrow>
      <h1 className={styles.title}>{title ?? FALLBACK_PAPER_TITLE}</h1>
      <p className={styles.lead}>
        {hasRepo
          ? "We found the code for this paper. Confirm the repository to import, or replace it with another."
          : "We couldn't automatically find a code repository for this paper. Add one below to import it, or leave it blank to start without one."}
      </p>
      <RepoField
        label="GitHub repository"
        value={repoUrl}
        onChange={setRepoUrl}
        placeholder="owner/repo"
        className={styles.field}
      />
      <div className={styles.gpuCountRow}>
        <label htmlFor={gpuSelectId} className={styles.gpuCountLabel}>
          GPU count
        </label>
        <select
          id={gpuSelectId}
          className={styles.gpuCountSelect}
          value={gpuCount ?? "auto"}
          onChange={(event) => {
            const raw = event.target.value;
            setGpuCount(raw === "auto" ? null : Number(raw));
          }}
        >
          <option value="auto">Auto (recommended)</option>
          {GPU_COUNT_OPTIONS.map((n) => (
            <option key={n} value={n}>
              {n === 1 ? "1 GPU" : `${n} GPUs`}
            </option>
          ))}
        </select>
      </div>
      <PrimaryButton onClick={() => setDialogOpen(true)} className={styles.cta}>
        🚀 Start autoresearch
      </PrimaryButton>

      {dialogOpen && (
        <LaunchConfirmDialog
          signedIn={isDevSessionSignedIn()}
          busy={busy}
          error={error}
          estimate={estimate}
          estimateLoading={estimateLoading}
          estimateError={estimateError}
          gpuCount={gpuCount}
          onCancel={() => setDialogOpen(false)}
          onConfirm={handleConfirmLaunch}
        />
      )}
    </Card>
  );
}

interface LaunchConfirmDialogProps {
  signedIn: boolean;
  busy: boolean;
  error: string | null;
  estimate: PaperBudgetEstimate | null;
  estimateLoading: boolean;
  estimateError: string | null;
  gpuCount: number | null;
  onCancel: () => void;
  onConfirm: () => void;
}

/** The D4 launch cost-guard: accessible dialog (role, focus, Escape/cancel)
 * mirroring the shortcut-overlay / node-detail-popup pattern used elsewhere
 * in the lab. */
function LaunchConfirmDialog({
  signedIn,
  busy,
  error,
  estimate,
  estimateLoading,
  estimateError,
  gpuCount,
  onCancel,
  onConfirm,
}: LaunchConfirmDialogProps) {
  const cancelBtnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  useEffect(() => {
    cancelBtnRef.current?.focus();
  }, []);

  const recipe = estimate ? pickRecipe(estimate) : null;
  const isQuotaError = error ? isQuotaOrCapacityError(error) : false;

  return (
    <div
      className={styles.overlay}
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) onCancel();
      }}
    >
      <div
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label="Confirm autoresearch launch"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 className={styles.dialogTitle}>Start autoresearch?</h2>
        <p className={styles.dialogBody}>
          This launches a fully autonomous reproduction run, capped at an
          estimated <strong>{ESTIMATED_GPU_CAP_DISPLAY}</strong> in GPU spend.
        </p>
        {recipe ? (
          <p className={styles.dialogEstimate}>
            Estimated for this paper:{" "}
            <strong>
              {fmtUsd(recipe.gpu_usd + recipe.api_usd_best)}–
              {fmtUsd(recipe.gpu_usd + recipe.api_usd_worst)}
            </strong>{" "}
            · ~{fmtHours(recipe.wall_clock_hours_p50)}
          </p>
        ) : estimateLoading ? (
          <p className={styles.dialogEstimateLoading}>Estimating cost for this paper…</p>
        ) : estimateError ? (
          <p className={styles.dialogEstimateLoading}>
            Estimate unavailable — the {ESTIMATED_GPU_CAP_DISPLAY} cap above still applies.
          </p>
        ) : null}
        <p className={styles.dialogGpuChoice}>
          GPUs: <strong>{gpuCount != null ? gpuCount : "Auto"}</strong>
        </p>
        {!signedIn && (
          <p className={styles.dialogWarning} role="alert">
            Sign in first to start an autonomous run.
          </p>
        )}
        {error && (
          <div className={styles.dialogErrorBlock} role="alert">
            <p className={styles.dialogError}>
              {isQuotaError
                ? "GPU capacity is temporarily unavailable for this request — this is usually transient."
                : error}
            </p>
            {isQuotaError && <p className={styles.dialogErrorDetail}>{error}</p>}
            <button
              type="button"
              className={styles.retryBtn}
              onClick={onConfirm}
              disabled={!signedIn || busy}
            >
              {busy ? "Retrying…" : "Retry"}
            </button>
          </div>
        )}
        <div className={styles.dialogActions}>
          <button
            type="button"
            ref={cancelBtnRef}
            className={styles.cancelBtn}
            onClick={onCancel}
          >
            Cancel
          </button>
          <PrimaryButton onClick={onConfirm} disabled={!signedIn || busy}>
            {busy ? "Launching…" : "Confirm & launch"}
          </PrimaryButton>
        </div>
      </div>
    </div>
  );
}
