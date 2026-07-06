"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { useRun } from "@/hooks/use-run";
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
// this in sync if that profile value ever changes.
const ESTIMATED_GPU_CAP_DISPLAY = "$10";

/**
 * Repo-confirm screen (alphaXiv `_4`) — shown at /abs/<arxivId>?confirm=1
 * after the stubbed sign-in step. Lets the user confirm/replace the
 * auto-resolved repository, then gates the actual launch behind a cost-guard
 * dialog (D4): shows the autonomous profile's GPU cap and requires the dev
 * session flag before it will call the launcher.
 */
export function RepoConfirm({ arxivId, title, initialRepoUrl, className }: RepoConfirmProps) {
  const router = useRouter();
  const [repoUrl, setRepoUrl] = useState(initialRepoUrl);
  const [dialogOpen, setDialogOpen] = useState(false);
  const hasRepo = repoUrl.trim().length > 0;

  const { startArxivRun, busy, error } = useRun(null, {
    autonomous: true,
    repoUrl: repoUrl.trim() || undefined,
  });

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
      <PrimaryButton onClick={() => setDialogOpen(true)} className={styles.cta}>
        🚀 Start autoresearch
      </PrimaryButton>

      {dialogOpen && (
        <LaunchConfirmDialog
          signedIn={isDevSessionSignedIn()}
          busy={busy}
          error={error}
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
  onCancel: () => void;
  onConfirm: () => void;
}

/** The D4 launch cost-guard: accessible dialog (role, focus, Escape/cancel)
 * mirroring the shortcut-overlay / node-detail-popup pattern used elsewhere
 * in the lab. */
function LaunchConfirmDialog({ signedIn, busy, error, onCancel, onConfirm }: LaunchConfirmDialogProps) {
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
        {!signedIn && (
          <p className={styles.dialogWarning} role="alert">
            Sign in first to start an autonomous run.
          </p>
        )}
        {error && (
          <p className={styles.dialogError} role="alert">
            {error}
          </p>
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
