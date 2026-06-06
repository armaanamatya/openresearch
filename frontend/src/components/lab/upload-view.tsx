"use client";

import { useRef, useState } from "react";

import type { AuthStatus, DemoAccelerator, DemoGpuParallelism, RootProvider, SubagentAuth, DemoModelChoice, DemoRunMode, DemoSandboxMode } from "@/lib/demo/demo-run-types";
import { RUN_MODE_OPTIONS } from "@/lib/demo/demo-run-types";
import type { ProviderCredentialsInput } from "@/hooks/use-run";
import { BudgetPanel, type PaperBudgetEstimate, type RecipeMode } from "./budget/budget-panel";

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
  minimizeCompute,
  sandbox,
  providerCredentials,
  onRootProviderChange,
  onSubagentAuthChange,
  onDynamicGpuChange,
  onForceSingleGpuChange,
  gpuParallelism,
  onGpuParallelismChange,
  accelerator,
  onAcceleratorChange,
  onMaxGpuUsdPerHourChange,
  onVramGbChange,
  onMinimizeComputeChange,
  onSandboxChange,
  onProviderCredentialsChange,
  budgetEstimate,
  budgetLoading,
  budgetError,
  selectedRecipe,
  selectedProvider,
  hasPendingPaper,
  estimateSkipped,
  onSelectRecipe,
  onSelectProvider,
  onSkipEstimate,
  onConfirmRun,
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
  minimizeCompute: boolean;
  sandbox: DemoSandboxMode;
  providerCredentials: ProviderCredentialsInput;
  onRootProviderChange: (value: RootProvider) => void;
  onSubagentAuthChange: (value: SubagentAuth) => void;
  onDynamicGpuChange: (value: boolean) => void;
  onForceSingleGpuChange: (value: boolean) => void;
  gpuParallelism: DemoGpuParallelism;
  onGpuParallelismChange: (value: DemoGpuParallelism) => void;
  accelerator: DemoAccelerator;
  onAcceleratorChange: (value: DemoAccelerator) => void;
  onMaxGpuUsdPerHourChange: (value: number) => void;
  onVramGbChange: (value: number) => void;
  onMinimizeComputeChange: (value: boolean) => void;
  onSandboxChange: (value: DemoSandboxMode) => void;
  onProviderCredentialsChange: (value: ProviderCredentialsInput) => void;
  budgetEstimate: PaperBudgetEstimate | null;
  budgetLoading: boolean;
  budgetError: string | null;
  selectedRecipe: RecipeMode;
  selectedProvider: string | null;
  hasPendingPaper: boolean;
  estimateSkipped: boolean;
  onSelectRecipe: (mode: RecipeMode) => void;
  onSelectProvider: (modelId: string) => void;
  onSkipEstimate: () => void;
  onConfirmRun: () => void;
}) {
  const fileInput = useRef<HTMLInputElement | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // BYO keys: per-session reveal toggle keyed by field name. Defaults to
  // hidden so a casual screenshot won't expose the value. The credentials
  // themselves are kept in `providerCredentials` (parent state, in-memory
  // only — never written to localStorage).
  const [revealKey, setRevealKey] = useState<Record<string, boolean>>({});
  const setCredField = (key: keyof ProviderCredentialsInput, value: string): void => {
    onProviderCredentialsChange({ ...providerCredentials, [key]: value });
  };
  const toggleReveal = (key: string): void => {
    setRevealKey((prev) => ({ ...prev, [key]: !prev[key] }));
  };
  // Show the BYO key block when the picked provider has a typeable key
  // surface. anthropic_oauth + featherless don't surface here:
  // - oauth uses the local `claude login` subscription, no key.
  // - featherless is operator-only (cheapest plan), not a per-run BYO surface.
  const showAnthropicKey = rootProvider === "anthropic_api";
  const showOpenAIKey = rootProvider === "openai_api";
  const showAzureFields = rootProvider === "azure_openai";
  const showCredentialsBlock = showAnthropicKey || showOpenAIKey || showAzureFields;

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

  // The BudgetPanel is shown whenever a paper is pending (file dropped or
  // arXiv URL submitted) AND the user hasn't explicitly skipped a failed
  // estimate. It blocks Begin until the panel's Confirm is clicked, which
  // is the user-mandated "no run without seeing the estimate" gate.
  const showBudgetPanel = hasPendingPaper && !estimateSkipped;

  return (
    <div className="upload-shell">
      {showBudgetPanel ? (
        <BudgetPanel
          estimate={budgetEstimate}
          loading={budgetLoading}
          error={budgetError}
          selectedRecipe={selectedRecipe}
          selectedProvider={selectedProvider}
          onSelectRecipe={onSelectRecipe}
          onSelectProvider={onSelectProvider}
          onSkip={onSkipEstimate}
          onConfirm={onConfirmRun}
          busy={busy}
        />
      ) : null}
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
          Drop a paper here or click to browse. OpenResearch will reproduce, verify, and report -
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

      {/* ── BYO API keys (optional) ─────────────────────────────── */}
      {showCredentialsBlock ? (
        <fieldset className="upload-provider-fieldset upload-byo-fieldset" disabled={busy}>
          <legend className="upload-config-label">API keys (optional)</legend>
          <p className="upload-byo-hint">
            Paste a key to use it for this run only. Values are sent to the
            backend over localhost and injected into the run subprocess&apos;s
            env — never persisted to disk and never echoed in logs.
          </p>
          {showAnthropicKey ? (
            <ByoKeyField
              fieldName="anthropic_api_key"
              label="ANTHROPIC_API_KEY"
              placeholder="sk-ant-…"
              value={providerCredentials.anthropic_api_key ?? ""}
              revealed={revealKey.anthropic_api_key ?? false}
              onChange={(v) => setCredField("anthropic_api_key", v)}
              onToggleReveal={() => toggleReveal("anthropic_api_key")}
              hint={
                authStatus?.providers.anthropic_api?.available
                  ? "Overrides the server-side ANTHROPIC_API_KEY for this run."
                  : "Server-side key not set — provide one to run via Anthropic."
              }
              busy={busy}
            />
          ) : null}
          {showOpenAIKey ? (
            <ByoKeyField
              fieldName="openai_api_key"
              label="OPENAI_API_KEY"
              placeholder="sk-…"
              value={providerCredentials.openai_api_key ?? ""}
              revealed={revealKey.openai_api_key ?? false}
              onChange={(v) => setCredField("openai_api_key", v)}
              onToggleReveal={() => toggleReveal("openai_api_key")}
              hint={
                authStatus?.providers.openai_api?.available
                  ? "Overrides the server-side OPENAI_API_KEY for this run."
                  : "Server-side key not set — provide one to run via OpenAI."
              }
              busy={busy}
            />
          ) : null}
          {showAzureFields ? (
            <>
              <ByoKeyField
                fieldName="azure_openai_api_key"
                label="AZURE_OPENAI_API_KEY"
                placeholder="32-char key from Azure portal"
                value={providerCredentials.azure_openai_api_key ?? ""}
                revealed={revealKey.azure_openai_api_key ?? false}
                onChange={(v) => setCredField("azure_openai_api_key", v)}
                onToggleReveal={() => toggleReveal("azure_openai_api_key")}
                hint=""
                busy={busy}
              />
              <ByoTextField
                fieldName="azure_openai_endpoint"
                label="AZURE_OPENAI_ENDPOINT"
                placeholder="https://my-resource.openai.azure.com"
                value={providerCredentials.azure_openai_endpoint ?? ""}
                onChange={(v) => setCredField("azure_openai_endpoint", v)}
                hint="Required for Azure. The resource URL — without a trailing path."
                busy={busy}
              />
              <ByoTextField
                fieldName="azure_openai_deployment"
                label="AZURE_OPENAI_DEPLOYMENT"
                placeholder="gpt-4o"
                value={providerCredentials.azure_openai_deployment ?? ""}
                onChange={(v) => setCredField("azure_openai_deployment", v)}
                hint="Optional. Defaults to the model name."
                busy={busy}
              />
              <ByoTextField
                fieldName="azure_openai_api_version"
                label="AZURE_OPENAI_API_VERSION"
                placeholder="2024-10-21 (current GA)"
                value={providerCredentials.azure_openai_api_version ?? ""}
                onChange={(v) => setCredField("azure_openai_api_version", v)}
                hint="Optional. Leave blank to use the backend default."
                busy={busy}
              />
            </>
          ) : null}
        </fieldset>
      ) : null}

      {/* ── Minimize compute (top-level — reproduction philosophy choice) ── */}
      <fieldset className="upload-provider-fieldset" disabled={busy}>
        <legend className="upload-config-label">Reproduction style</legend>
        <label
          className={`upload-provider-option${minimizeCompute ? " selected" : ""}`}
          title="When enabled, the agent reproduces the paper's CLAIM (its reported metric) rather than its exact recipe — substituting slow paper schedules (SGD+linear-decay-from-10 × 3000 epochs) with modern fast equivalents (Adam@lr=0.001 × 200-500 epochs). Every substitution is annotated in scope.declared_reductions and the scope-adjusted rubric scores the metric match. Off by default — strict reproduction is the safer baseline."
        >
          <input
            id="minimize-compute-checkbox"
            type="checkbox"
            checked={minimizeCompute}
            disabled={busy}
            onChange={(e) => onMinimizeComputeChange(e.target.checked)}
          />
          <span className="upload-provider-name">Minimize compute</span>
          <span className="upload-provider-badge">
            {minimizeCompute ? "claim-match" : "strict"}
          </span>
        </label>
      </fieldset>

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
            <label className="upload-advanced-label" htmlFor="gpu-parallelism-select">
              GPU parallelism
            </label>
            <select
              id="gpu-parallelism-select"
              className="upload-config-select"
              value={gpuParallelism}
              disabled={busy}
              onChange={(e) => onGpuParallelismChange(e.target.value as DemoGpuParallelism)}
            >
              <option value="auto">Auto</option>
              <option value="single">Single</option>
              <option value="multi">Multi</option>
            </select>
            <span className="upload-advanced-hint">Multi-GPU (DDP) when the paper benefits</span>
          </div>
          <div className="upload-advanced-row">
            <label className="upload-advanced-label" htmlFor="accelerator-select">
              Accelerator (cheap calls)
            </label>
            <select
              id="accelerator-select"
              className="upload-config-select"
              value={accelerator}
              disabled={busy}
              onChange={(e) => onAcceleratorChange(e.target.value as DemoAccelerator)}
            >
              <option value="off">Off</option>
              <option value="auto">Auto</option>
              <option value="local">Local (vLLM)</option>
              <option value="runpod">RunPod</option>
              <option value="azure">Azure</option>
              <option value="endpoint">Endpoint</option>
            </select>
            <span className="upload-advanced-hint">Fast model for navigation/scoring; root stays Sonnet</span>
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

// ---------------------------------------------------------------------------
// BYO key helper components — local to upload-view.
// Kept small + uncontrolled-state-free; parent owns the value, this just
// renders the input + reveal toggle. type="password" by default; "text"
// when the user explicitly clicks the show toggle.
// ---------------------------------------------------------------------------

interface ByoKeyFieldProps {
  fieldName: string;
  label: string;
  placeholder: string;
  value: string;
  revealed: boolean;
  onChange: (value: string) => void;
  onToggleReveal: () => void;
  hint: string;
  busy: boolean;
}

function ByoKeyField({
  fieldName,
  label,
  placeholder,
  value,
  revealed,
  onChange,
  onToggleReveal,
  hint,
  busy,
}: ByoKeyFieldProps) {
  return (
    <div className="upload-byo-row">
      <label className="upload-byo-label" htmlFor={`byo-${fieldName}`}>
        {label}
      </label>
      <div className="upload-byo-input-wrap">
        <input
          id={`byo-${fieldName}`}
          type={revealed ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          className="upload-byo-input mono"
          value={value}
          disabled={busy}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          className="upload-byo-reveal"
          aria-pressed={revealed}
          aria-label={revealed ? "Hide key" : "Show key"}
          disabled={busy}
          onClick={onToggleReveal}
        >
          {revealed ? "Hide" : "Show"}
        </button>
      </div>
      {hint ? <p className="upload-byo-fieldhint">{hint}</p> : null}
    </div>
  );
}

interface ByoTextFieldProps {
  fieldName: string;
  label: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  hint: string;
  busy: boolean;
}

function ByoTextField({
  fieldName,
  label,
  placeholder,
  value,
  onChange,
  hint,
  busy,
}: ByoTextFieldProps) {
  return (
    <div className="upload-byo-row">
      <label className="upload-byo-label" htmlFor={`byo-${fieldName}`}>
        {label}
      </label>
      <input
        id={`byo-${fieldName}`}
        type="text"
        autoComplete="off"
        spellCheck={false}
        className="upload-byo-input mono"
        value={value}
        disabled={busy}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
      {hint ? <p className="upload-byo-fieldhint">{hint}</p> : null}
    </div>
  );
}
