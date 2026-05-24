"use client";

import { useRef, useState } from "react";

import type { AuthStatus, RootProvider, SubagentAuth, DemoModelChoice, DemoRunMode, DemoSandboxMode } from "@/lib/demo/demo-run-types";
import { RUN_MODE_OPTIONS } from "@/lib/demo/demo-run-types";

// ---------------------------------------------------------------------------
// Sandbox options — surface the two end-user-relevant choices.
// "auto"/"local" exist in the type but the lab path defaults docker on local
// hardware and runpod on a GPU cloud; they're not meaningful picks for users.
// ---------------------------------------------------------------------------
const SANDBOX_OPTIONS: { value: DemoSandboxMode; label: string; hint: string }[] = [
  { value: "docker", label: "Local (Docker)", hint: "CPU-only on your machine. Fast iteration, no compute cost." },
  { value: "runpod", label: "RunPod GPU",     hint: "GPU pod on RunPod. Requires funded RunPod account (≈$0.34/hr RTX 4090 COMMUNITY)." },
];
import { PRESET_PAPERS } from "@/lib/demo/preset-papers";
import type { ModelChoice } from "@/lib/models/server-fetch";
import { ICONS } from "./icons";

import "./upload-view.css";

// ---------------------------------------------------------------------------
// Provider label + env-var hint table
// ---------------------------------------------------------------------------

const PROVIDER_LABELS: Record<RootProvider, string> = {
  anthropic_api: "Anthropic API",
  anthropic_oauth: "Anthropic OAuth",
  openai_api: "OpenAI",
  azure_openai: "Azure OpenAI",
  featherless: "Featherless",
};

const PROVIDER_ENV_HINTS: Record<RootProvider, string> = {
  anthropic_api: "Set ANTHROPIC_API_KEY and reload",
  anthropic_oauth: "Run `claude login` and reload",
  openai_api: "Set OPENAI_API_KEY and reload",
  azure_openai: "Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT and reload",
  featherless: "Set FEATHERLESS_API_KEY and reload",
};

const SUBAGENT_LABELS: Record<SubagentAuth, string> = {
  anthropic_api: "Anthropic API (key)",
  anthropic_oauth: "Anthropic OAuth (subscription)",
};

// ---------------------------------------------------------------------------
// UploadView component
// Auth status is fetched server-side by lab/page.tsx and passed as a prop.
// This avoids a client-side fetch on render and keeps the component simple.
// ---------------------------------------------------------------------------

export function UploadView({
  arxiv,
  authStatus,
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
  setOver,
  rootProvider,
  subagentAuth,
  dynamicGpu,
  forceSingleGpu,
  maxGpuUsdPerHour,
  vramGb,
  sandbox,
  onRootProviderChange,
  onSubagentAuthChange,
  onDynamicGpuChange,
  onForceSingleGpuChange,
  onMaxGpuUsdPerHourChange,
  onVramGbChange,
  onSandboxChange,
}: {
  arxiv: string;
  authStatus: AuthStatus | null;
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
  rootProvider: RootProvider;
  subagentAuth: SubagentAuth;
  dynamicGpu: boolean;
  forceSingleGpu: boolean;
  maxGpuUsdPerHour: number;
  vramGb: number;
  sandbox: DemoSandboxMode;
  onRootProviderChange: (value: RootProvider) => void;
  onSubagentAuthChange: (value: SubagentAuth) => void;
  onDynamicGpuChange: (value: boolean) => void;
  onForceSingleGpuChange: (value: boolean) => void;
  onMaxGpuUsdPerHourChange: (value: number) => void;
  onVramGbChange: (value: number) => void;
  onSandboxChange: (value: DemoSandboxMode) => void;
}) {
  const fileInput = useRef<HTMLInputElement | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const providers: RootProvider[] = [
    "anthropic_api",
    "anthropic_oauth",
    "openai_api",
    "azure_openai",
    "featherless",
  ];

  const subagentOptions: SubagentAuth[] = ["anthropic_api", "anthropic_oauth"];

  // F3: widen the model picker when the persisted model isn't in the
  // backend-supplied list (e.g. a registry alias resolved server-side, or
  // a value typed via the OAI proxy). Prepend the active model so the
  // <select> always has a matching <option> — otherwise the controlled
  // input falls back to the first option and silently changes the user's
  // choice on first render.
  const currentModelInList = models.some((m) => m.id === model);
  const modelOptions = currentModelInList || models.length === 0
    ? models
    : [
        {
          id: model,
          label: model.charAt(0).toUpperCase() + model.slice(1),
          provider: "custom",
          available: true,
        },
        ...models,
      ];

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
          {modelOptions.length > 0
            ? modelOptions.map((m) => {
                const unavailable = m.available === false;
                const suffix = unavailable ? " (credentials missing)" : "";
                const title = unavailable && m.missingCredentials?.length
                  ? `Missing: ${m.missingCredentials.join(", ")}`
                  : `${m.provider}${m.paperValidated ? " · paper-validated" : ""}`;
                return (
                  <option key={m.id} value={m.id} disabled={unavailable} title={title}>
                    {m.label}{suffix}
                  </option>
                );
              })
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

      {/* ── Sandbox radio group ──────────────────────────────────── */}
      <fieldset className="upload-provider-fieldset" disabled={busy}>
        <legend className="upload-config-label">Sandbox</legend>
        <div className="upload-provider-options">
          {SANDBOX_OPTIONS.map((opt) => (
            <label
              key={opt.value}
              className={`upload-provider-option${sandbox === opt.value ? " selected" : ""}`}
              title={opt.hint}
            >
              <input
                type="radio"
                name="sandbox"
                value={opt.value}
                checked={sandbox === opt.value}
                onChange={() => onSandboxChange(opt.value)}
              />
              <span className="upload-provider-name">{opt.label}</span>
            </label>
          ))}
        </div>
      </fieldset>

      {/* ── LLM provider radio group ─────────────────────────────── */}
      <fieldset className="upload-provider-fieldset" disabled={busy}>
        <legend className="upload-config-label">LLM provider</legend>
        <div className="upload-provider-options">
          {providers.map((p) => {
            const status = authStatus?.providers[p];
            const available = status ? status.available : true; // optimistic until loaded
            const hint = available
              ? (status?.detail ?? "")
              : PROVIDER_ENV_HINTS[p];
            return (
              <label
                key={p}
                className={`upload-provider-option${rootProvider === p ? " selected" : ""}${!available ? " disabled" : ""}`}
                title={hint}
                aria-disabled={!available ? "true" : undefined}
              >
                <input
                  type="radio"
                  name="root_provider"
                  value={p}
                  checked={rootProvider === p}
                  disabled={!available}
                  onChange={() => onRootProviderChange(p)}
                />
                <span className="upload-provider-name">{PROVIDER_LABELS[p]}</span>
                {status ? (
                  <span className={`upload-provider-badge${available ? " ok" : " err"}`}>
                    {available ? "✓" : "✗"}
                  </span>
                ) : null}
              </label>
            );
          })}
        </div>
      </fieldset>

      {/* ── Sub-agent auth radio group ───────────────────────────── */}
      {(() => {
        const noneAvailable = authStatus
          ? !authStatus.subagent_auth.anthropic_api && !authStatus.subagent_auth.anthropic_oauth
          : false;
        return (
          <fieldset className="upload-provider-fieldset" disabled={busy}>
            <legend className="upload-config-label">Sub-agent auth</legend>
            {noneAvailable && (
              <p className="upload-subagent-warn">
                No sub-agent auth available — set <code>ANTHROPIC_API_KEY</code> to enable Sonnet implementation calls.
              </p>
            )}
            <div className="upload-provider-options">
              {subagentOptions.map((s) => {
                const available = authStatus ? authStatus.subagent_auth[s] : true;
                const hint = available
                  ? SUBAGENT_LABELS[s]
                  : (s === "anthropic_api" ? "Set ANTHROPIC_API_KEY and reload" : "Run `claude login` and reload");
                return (
                  <label
                    key={s}
                    className={`upload-provider-option${subagentAuth === s && !noneAvailable ? " selected" : ""}${!available ? " disabled" : ""}`}
                    title={hint}
                    aria-disabled={!available ? "true" : undefined}
                  >
                    <input
                      type="radio"
                      name="subagent_auth"
                      value={s}
                      checked={subagentAuth === s}
                      disabled={!available}
                      onChange={() => onSubagentAuthChange(s)}
                    />
                    <span className="upload-provider-name">{SUBAGENT_LABELS[s]}</span>
                    {authStatus ? (
                      <span className={`upload-provider-badge${available ? " ok" : " err"}`}>
                        {available ? "✓" : "✗"}
                      </span>
                    ) : null}
                  </label>
                );
              })}
            </div>
          </fieldset>
        );
      })()}

      {/* ── Advanced options (collapsible) ──────────────────────── */}
      <details
        className="upload-advanced"
        open={advancedOpen}
        onToggle={(e) => setAdvancedOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary className="upload-advanced-summary upload-config-label">
          Advanced options
        </summary>
        <div className="upload-advanced-body">
          <div className="upload-advanced-row">
            <label className="upload-advanced-label" htmlFor="dynamic-gpu-toggle">
              Dynamic GPU
            </label>
            <input
              id="dynamic-gpu-toggle"
              type="checkbox"
              checked={dynamicGpu}
              disabled={busy}
              onChange={(e) => onDynamicGpuChange(e.target.checked)}
            />
            <span className="upload-advanced-hint">Escalate GPU tier on CUDA OOM</span>
          </div>
          <div className="upload-advanced-row">
            <label className="upload-advanced-label" htmlFor="force-single-gpu-toggle">
              Force single GPU
            </label>
            <input
              id="force-single-gpu-toggle"
              type="checkbox"
              checked={forceSingleGpu}
              disabled={busy}
              onChange={(e) => onForceSingleGpuChange(e.target.checked)}
            />
          </div>
          <div className="upload-advanced-row">
            <label className="upload-advanced-label" htmlFor="max-gpu-usd-input">
              Max GPU $/hr
            </label>
            <input
              id="max-gpu-usd-input"
              type="number"
              min={0}
              step={0.01}
              className="upload-advanced-number"
              value={maxGpuUsdPerHour}
              disabled={busy}
              onChange={(e) => onMaxGpuUsdPerHourChange(parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="upload-advanced-row">
            <label className="upload-advanced-label" htmlFor="vram-gb-input">
              VRAM (GB)
            </label>
            <input
              id="vram-gb-input"
              type="number"
              min={0}
              step={1}
              className="upload-advanced-number"
              value={vramGb}
              disabled={busy}
              onChange={(e) => onVramGbChange(parseInt(e.target.value, 10) || 0)}
            />
          </div>
        </div>
      </details>

      {error ? <p className="upload-error">{error}</p> : null}
    </div>
  );
}
