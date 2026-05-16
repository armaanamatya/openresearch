"use client";

import { useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { RecentRunSummary } from "@/lib/runs/server-list";

import "./command-palette.css";

type CommandPaletteProps = {
  open: boolean;
  setOpen: (open: boolean) => void;
  recents: RecentRunSummary[];
  currentRun: LiveDemoRunState | null;
};

function copyText(text: string) {
  try {
    void navigator.clipboard?.writeText(text);
  } catch {
    // non-fatal — clipboard may be blocked in the harness
  }
}

export function CommandPalette({ open, setOpen, recents, currentRun }: CommandPaletteProps) {
  const router = useRouter();

  const close = useCallback(() => setOpen(false), [setOpen]);

  // Defensive Esc handler — cmdk handles input focus, but if the user
  // clicks the overlay we still want Esc on the document to close.
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        close();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, close]);

  if (!open) return null;

  const runAction = (fn: () => void) => () => {
    fn();
    close();
  };

  const onNavigate = (href: string) => runAction(() => router.push(href));

  const onCopyProjectId = currentRun
    ? runAction(() => copyText(currentRun.projectId))
    : null;

  const onCopyShareUrl = currentRun
    ? runAction(() => {
        const origin = typeof window !== "undefined" ? window.location.origin : "";
        copyText(`${origin}/lab?projectId=${currentRun.projectId}`);
      })
    : null;

  const onOpenFinalReport = currentRun
    ? runAction(() => {
        const href = `/api/demo/final-report?projectId=${currentRun.projectId}`;
        if (typeof window !== "undefined") {
          window.open(href, "_blank", "noopener,noreferrer");
        }
      })
    : null;

  return (
    <div
      className="cmdk-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
      role="presentation"
    >
      <Command
        className="cmdk-dialog"
        label="Command palette"
        loop
        onClick={(e) => e.stopPropagation()}
      >
        <Command.Input
          className="cmdk-input"
          placeholder="Type a command or search recent runs..."
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
          autoFocus
        />
        <Command.List className="cmdk-list">
          <Command.Empty className="cmdk-empty">No results.</Command.Empty>

          <Command.Group className="cmdk-group" heading="Navigate">
            <Command.Item className="cmdk-item" onSelect={onNavigate("/lab")}>
              Lab
              <span className="cmdk-item-kind">/lab</span>
            </Command.Item>
            <Command.Item className="cmdk-item" onSelect={onNavigate("/library")}>
              Library
              <span className="cmdk-item-kind">/library</span>
            </Command.Item>
            <Command.Item className="cmdk-item" onSelect={onNavigate("/paperbench")}>
              PaperBench
              <span className="cmdk-item-kind">/paperbench</span>
            </Command.Item>
          </Command.Group>

          {currentRun && onCopyProjectId && onCopyShareUrl && onOpenFinalReport ? (
            <Command.Group className="cmdk-group" heading="Current run">
              <Command.Item className="cmdk-item" onSelect={onCopyProjectId}>
                Copy project id
                <span className="cmdk-item-id">{currentRun.projectId}</span>
              </Command.Item>
              <Command.Item className="cmdk-item" onSelect={onCopyShareUrl}>
                Copy shareable URL
                <span className="cmdk-item-kind">/lab?projectId=…</span>
              </Command.Item>
              <Command.Item className="cmdk-item" onSelect={onOpenFinalReport}>
                Open final report
                <span className="cmdk-item-kind">new tab</span>
              </Command.Item>
            </Command.Group>
          ) : null}

          {recents.length > 0 ? (
            <Command.Group className="cmdk-group" heading="Recent runs">
              {recents.map((entry) => (
                <Command.Item
                  key={entry.projectId}
                  className="cmdk-item"
                  value={`${entry.projectId} ${entry.sourceLabel ?? ""} ${entry.status}`}
                  onSelect={onNavigate(`/lab?projectId=${entry.projectId}`)}
                >
                  <span>{entry.sourceLabel ?? entry.projectId}</span>
                  <span className="cmdk-item-id">{entry.projectId}</span>
                </Command.Item>
              ))}
            </Command.Group>
          ) : null}
        </Command.List>
      </Command>
    </div>
  );
}
