"use client";

import { useRef } from "react";

import type { DemoModelChoice, DemoRunMode } from "@/lib/demo/demo-run-types";
import { RUN_MODE_OPTIONS } from "@/lib/demo/demo-run-types";
import { PRESET_PAPERS } from "@/lib/demo/preset-papers";
import type { ModelChoice } from "@/lib/models/server-fetch";
import { ICONS } from "./icons";

import "./upload-view.css";

export function UploadView({
  arxiv,
  busy,
  error,
  model,
  models,
  runMode,
  onArxivChange,
  onArxivSubmit,
  onFileSelected,
  onModelChange,
  onRunModeChange,
  over,
  setOver
}: {
  arxiv: string;
  busy: boolean;
  error: string | null;
  model: DemoModelChoice;
  models: ModelChoice[];
  runMode: DemoRunMode;
  onArxivChange: (value: string) => void;
  onArxivSubmit: () => void;
  onFileSelected: (file: File) => void;
  onModelChange: (value: DemoModelChoice) => void;
  onRunModeChange: (value: DemoRunMode) => void;
  over: boolean;
  setOver: (value: boolean) => void;
}) {
  const fileInput = useRef<HTMLInputElement | null>(null);

  return (
    <div className="upload-shell">
      <div
        className={`upload-zone${over ? " over" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          const file = event.dataTransfer.files[0];
          if (file) {
            onFileSelected(file);
          }
        }}
        onClick={() => fileInput.current?.click()}
      >
        <input
          ref={fileInput}
          type="file"
          accept=".pdf"
          className="hidden-input"
          aria-label="Upload paper PDF"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onFileSelected(file);
              event.currentTarget.value = "";
            }
          }}
        />
        <div className="upload-icon">{ICONS.upload}</div>
        <h1 className="upload-title">Upload PDF</h1>
        <p className="upload-copy">
          Drop a paper here or click to browse. ReproLab will reproduce, verify, and report -
          independently.
        </p>
        <div className="upload-meta">PDF - max 50 MB - arXiv preprints recommended</div>
      </div>
      <div className="upload-divider">
        <span />
        <span className="upload-divider-label">or paste an arXiv link</span>
        <span />
      </div>
      <form
        className="upload-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy && arxiv.length >= 8) {
            onArxivSubmit();
          }
        }}
      >
        <span className="mono upload-prefix">https://</span>
        <input
          value={arxiv}
          onChange={(event) => {
            // Strip a leading scheme so a user who pastes a full URL doesn't
            // get `https://https://arxiv.org/...` rendered — the prefix span
            // is the visual scheme, the input is the path. Trim whitespace
            // too since clipboard pastes often carry trailing newlines.
            const normalised = event.target.value
              .replace(/^\s*https?:\/\//i, "")
              .trimStart();
            onArxivChange(normalised);
          }}
          placeholder="arxiv.org/abs/2303.04137"
          className="upload-text-input mono"
          disabled={busy}
        />
        <button type="submit" disabled={busy || arxiv.length < 8} className="begin-button">
          {busy ? "Starting..." : "Begin ->"}
        </button>
      </form>
      <div className="preset-row">
        <span className="preset-row-label">Or pick a preset paper</span>
        {PRESET_PAPERS.map((p) => (
          <button
            key={p.id}
            type="button"
            className="preset-chip"
            disabled={busy}
            title={p.title}
            onClick={() => onArxivChange(`arxiv.org/abs/${p.arxivId}`)}
          >
            <span className="preset-chip-short">{p.short}</span>
            <span className="preset-chip-id">{p.arxivId}</span>
          </button>
        ))}
      </div>
      <div className="upload-config-row">
        <label className="upload-config-label" htmlFor="model-select">Model</label>
        <select
          id="model-select"
          className="upload-config-select"
          value={model}
          disabled={busy}
          onChange={(event) => onModelChange(event.target.value as DemoModelChoice)}
        >
          {/* Options come from GET /api/models (proxied to the backend).
              When the list is empty (backend unreachable on the server
              render) we surface the active `model` so the control is
              always selectable. */}
          {models.length > 0
            ? models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))
            : (
                <option key={model} value={model}>
                  {model.charAt(0).toUpperCase() + model.slice(1)}
                </option>
              )}
        </select>
      </div>
      <div className="upload-config-row">
        <label className="upload-config-label" htmlFor="mode-select">Mode</label>
        <select
          id="mode-select"
          className="upload-config-select"
          value={runMode}
          disabled={busy}
          onChange={(event) => onRunModeChange(event.target.value as DemoRunMode)}
          title={RUN_MODE_OPTIONS.find((o) => o.value === runMode)?.description}
        >
          {RUN_MODE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value} title={o.description}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      {error ? <p className="upload-error">{error}</p> : null}
    </div>
  );
}
