"use client";

import { useEffect, useRef, useState } from "react";
import type { PrimitiveCallView, RlmRunState } from "../../../hooks/use-rlm-run";
import { describePrimitiveError } from "./primitive-error";
import styles from "./run-toasts.module.css";

interface RunToastsProps {
  status: RlmRunState["status"];
  iterationCount: number;
  primitiveCalls: PrimitiveCallView[];
  report: RlmRunState["report"];
}

interface Toast {
  id: number;
  tone: "info" | "warn" | "err";
  title: string;
  body: string;
  sticky: boolean;
}

interface Snapshot {
  status: RlmRunState["status"];
  iterationCount: number;
  primitiveCount: number;
  finalReportPath: string | null;
}

function snapshot(props: RunToastsProps): Snapshot {
  return {
    status: props.status,
    iterationCount: props.iterationCount,
    primitiveCount: props.primitiveCalls.length,
    finalReportPath: props.report?.finalReportPath ?? null,
  };
}

export function RunToasts({ status, iterationCount, primitiveCalls, report }: RunToastsProps) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const initializedRef = useRef(false);
  const lastRef = useRef<Snapshot | null>(null);
  const nextIdRef = useRef(1);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    return () => {
      for (const timer of timersRef.current) clearTimeout(timer);
      timersRef.current = [];
    };
  }, []);

  useEffect(() => {
    const current = snapshot({ status, iterationCount, primitiveCalls, report });
    const previous = lastRef.current;
    lastRef.current = current;

    if (!initializedRef.current) {
      initializedRef.current = true;
      return;
    }
    if (!previous) return;

    const pending: Toast[] = [];
    const add = (toast: Omit<Toast, "id">) => {
      pending.push({ id: nextIdRef.current++, ...toast });
    };

    if (iterationCount > previous.iterationCount) {
      add({
        tone: "info",
        title: `Iteration ${iterationCount} emitted`,
        body: "The root REPL produced a new turn. The canvas and state rail are updating from the event stream.",
        sticky: false,
      });
    }

    const newCalls = primitiveCalls.slice(previous.primitiveCount);
    for (const call of newCalls) {
      if (call.status !== "error") continue;
      const detail = describePrimitiveError(call);
      add({
        tone: "warn",
        title: detail.title,
        body: `${detail.reason} Recovery: ${detail.recovery}`,
        sticky: true,
      });
    }

    if (status !== previous.status && (status === "completed" || status === "partial" || status === "failed")) {
      add({
        tone: status === "failed" ? "err" : status === "partial" ? "warn" : "info",
        title:
          status === "failed"
            ? "Run failed"
            : status === "partial"
            ? "Run completed with partial verification"
            : "Run completed",
        body:
          status === "failed"
            ? "The subprocess reached a terminal error. The header banner contains the run-level failure."
            : "The run reached a terminal state. final_report.json is expected on disk when report generation finishes.",
        sticky: status === "failed",
      });
    }

    if (report?.finalReportPath && report.finalReportPath !== previous.finalReportPath) {
      add({
        tone: "info",
        title: "final_report.json ready",
        body: report.finalReportPath,
        sticky: false,
      });
    }

    if (pending.length === 0) return;
    setToasts((existing) => [...pending, ...existing].slice(0, 4));
    for (const toast of pending) {
      if (toast.sticky) continue;
      const timer = setTimeout(() => {
        setToasts((existing) => existing.filter((item) => item.id !== toast.id));
      }, 5_000);
      timersRef.current.push(timer);
    }
  }, [status, iterationCount, primitiveCalls, report]);

  if (toasts.length === 0) return null;

  return (
    <div className={styles.stack} aria-live="polite" aria-label="Run updates">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={[styles.toast, styles[toast.tone]].join(" ")}
          role={toast.tone === "err" || toast.tone === "warn" ? "alert" : "status"}
        >
          <span className={styles.dot} aria-hidden="true" />
          <div className={styles.content}>
            <p className={styles.title}>{toast.title}</p>
            <p className={styles.body}>{toast.body}</p>
          </div>
          <button
            type="button"
            className={styles.dismiss}
            aria-label="Dismiss run update"
            title="Dismiss"
            onClick={() => setToasts((existing) => existing.filter((item) => item.id !== toast.id))}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
