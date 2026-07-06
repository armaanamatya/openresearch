"use client";

import { useRouter } from "next/navigation";

import { markDevSessionSignedIn } from "@/lib/autoresearch/dev-session";
import { FALLBACK_PAPER_TITLE } from "@/lib/autoresearch/paper-meta";
import { Card } from "./ui/Card";
import { Eyebrow } from "./ui/Eyebrow";
import { PrimaryButton } from "./ui/PrimaryButton";
import styles from "./PaperLanding.module.css";

export interface PaperLandingProps {
  /** The paper's arXiv id, e.g. "2605.15155" — the one datum this screen can
   * always rely on (no metadata endpoint exists yet ahead of ingestion). */
  arxivId: string;
  /** Real paper title, once a metadata source supplies one. Omitted today —
   * falls back to an honest, paper-agnostic heading rather than a fabricated
   * title. */
  title?: string;
  className?: string;
}

/**
 * Paper-landing screen (alphaXiv `_3`) — the first thing a visitor sees at
 * /abs/<arxivId>. There is no real auth yet: "Sign in to start" sets a
 * stubbed dev-session flag (`markDevSessionSignedIn`) and advances to the
 * repo-confirm screen via `?confirm=1`, which the /abs/[arxivId] page reads
 * to decide which screen to render.
 */
export function PaperLanding({ arxivId, title, className }: PaperLandingProps) {
  const router = useRouter();

  const handleSignIn = () => {
    markDevSessionSignedIn();
    router.push(`/abs/${encodeURIComponent(arxivId)}?confirm=1`);
  };

  return (
    <Card className={[styles.card, className].filter(Boolean).join(" ")}>
      <Eyebrow>WELCOME TO OPENRESEARCH</Eyebrow>
      <h1 className={styles.title}>{title ?? FALLBACK_PAPER_TITLE}</h1>
      <p className={styles.arxivId}>arXiv {arxivId}</p>
      <div className={styles.body}>
        <p>
          OpenResearch deploys an agent to build a minimal reproduction of
          this paper. If it ships code, that means resolving setup issues
          until the run is error-free.
        </p>
        <p>
          From there you can direct the agent to run larger experiments from
          the paper and write up a report.
        </p>
      </div>
      <PrimaryButton onClick={handleSignIn} className={styles.cta}>
        → Sign in to start
      </PrimaryButton>
    </Card>
  );
}
