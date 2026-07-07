/**
 * Tests for UploadView — provider picker, subagent auth, advanced options,
 * localStorage persistence (D3), and disabled-state for unavailable providers.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AuthStatus } from "@/lib/demo/demo-run-types";
import { readProviderPrefs, writeProviderPrefs } from "@/lib/user-prefs";
import { UploadView } from "./upload-view";

// ---------------------------------------------------------------------------
// Mock fetch for /api/demo/auth-status
// ---------------------------------------------------------------------------

const ALL_AVAILABLE: AuthStatus = {
  providers: {
    anthropic_api: { available: true, detail: "ANTHROPIC_API_KEY set" },
    anthropic_oauth: { available: true, detail: "claude CLI subscription" },
    openai_api: { available: true, detail: "OPENAI_API_KEY set" },
    azure_openai: { available: true, detail: "Azure credentials set" },
    featherless: { available: true, detail: "FEATHERLESS_API_KEY set" },
    foundry: { available: true, detail: "AZURE_FOUNDRY_API_KEY set" },
  },
  subagent_auth: { anthropic_api: true, anthropic_oauth: true, foundry: true },
  defaults: {
    root_provider: "anthropic_oauth",
    root_model: "sonnet",
    subagent_auth: "anthropic_oauth",
  },
};

const ONLY_OAUTH: AuthStatus = {
  providers: {
    anthropic_api: { available: false, detail: "ANTHROPIC_API_KEY missing" },
    anthropic_oauth: { available: true, detail: "claude CLI subscription" },
    openai_api: { available: false, detail: "OPENAI_API_KEY missing" },
    azure_openai: { available: false, detail: "AZURE_OPENAI_API_KEY missing" },
    featherless: { available: false, detail: "FEATHERLESS_API_KEY missing" },
    foundry: { available: false, detail: "AZURE_FOUNDRY_API_KEY missing" },
  },
  subagent_auth: { anthropic_api: false, anthropic_oauth: true, foundry: false },
  defaults: {
    root_provider: "anthropic_oauth",
    root_model: "sonnet",
    subagent_auth: "anthropic_oauth",
  },
};

// The demo deployment: only Foundry funded, so it's the pre-selected default.
const ONLY_FOUNDRY: AuthStatus = {
  providers: {
    anthropic_api: { available: false, detail: "ANTHROPIC_API_KEY missing" },
    anthropic_oauth: { available: false, detail: "claude login required" },
    openai_api: { available: false, detail: "OPENAI_API_KEY missing" },
    azure_openai: { available: false, detail: "AZURE_OPENAI_API_KEY missing" },
    featherless: { available: false, detail: "FEATHERLESS_API_KEY missing" },
    foundry: { available: true, detail: "AZURE_FOUNDRY_API_KEY set" },
  },
  subagent_auth: { anthropic_api: false, anthropic_oauth: false, foundry: true },
  defaults: {
    root_provider: "foundry",
    root_model: "sonnet",
    subagent_auth: "foundry",
  },
};

// Default props — fill required callbacks with no-ops.
const NOP = () => {};
const DEFAULT_PROPS = {
  arxiv: "",
  authStatus: null as AuthStatus | null,
  busy: false,
  error: null,
  model: "sonnet" as const,
  models: [],
  runMode: "rlm" as const,
  onArxivChange: NOP,
  onArxivSubmit: NOP,
  onFileSelected: NOP,
  onModelChange: NOP,
  onRunModeChange: NOP,
  over: false,
  setOver: NOP,
  rootProvider: "anthropic_oauth" as const,
  subagentAuth: "anthropic_oauth" as const,
  dynamicGpu: false,
  forceSingleGpu: false,
  maxGpuUsdPerHour: 0,
  vramGb: 0,
  gpuCount: 0,
  minimizeCompute: false,
  autonomous: false,
  repoUrl: "",
  sandbox: "docker" as const,
  onRootProviderChange: NOP,
  onSubagentAuthChange: NOP,
  onDynamicGpuChange: NOP,
  onForceSingleGpuChange: NOP,
  gpuParallelism: "auto" as const,
  onGpuParallelismChange: NOP,
  accelerator: "off" as const,
  onAcceleratorChange: NOP,
  onMaxGpuUsdPerHourChange: NOP,
  onVramGbChange: NOP,
  onGpuCountChange: NOP,
  onMinimizeComputeChange: NOP,
  onAutonomousChange: NOP,
  onRepoUrlChange: NOP,
  onSandboxChange: NOP,
  providerCredentials: {},
  onProviderCredentialsChange: NOP,
  budgetEstimate: null,
  budgetLoading: false,
  budgetError: null,
  selectedRecipe: "strict" as const,
  selectedProvider: null,
  hasPendingPaper: false,
  estimateSkipped: false,
  onSelectRecipe: NOP,
  onSelectProvider: NOP,
  onSkipEstimate: NOP,
  onConfirmRun: NOP,
};

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

// ---------------------------------------------------------------------------
// Basic rendering
// ---------------------------------------------------------------------------

describe("UploadView basic rendering", () => {
  it("renders the upload zone heading", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
  });

  it("renders the LLM provider fieldset", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByText("LLM provider")).toBeInTheDocument();
  });

  it("renders the sub-agent auth fieldset", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByText("Sub-agent auth")).toBeInTheDocument();
  });

  it("renders the Advanced options summary", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByText("Advanced options")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Provider radio state
// ---------------------------------------------------------------------------

describe("UploadView provider radios", () => {
  it("checks the active rootProvider radio", () => {
    render(<UploadView {...DEFAULT_PROPS} authStatus={ALL_AVAILABLE} rootProvider="openai_api" />);
    // Use the input value directly — "OpenAI" label text matches openai_api radio value
    const radios = screen.getAllByRole("radio", { name: /OpenAI/ });
    const openaiRadio = radios.find((r) => (r as HTMLInputElement).value === "openai_api");
    expect(openaiRadio).toBeChecked();
  });

  it("calls onRootProviderChange when a provider is selected", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} authStatus={ALL_AVAILABLE} onRootProviderChange={onChange} />);
    const radios = screen.getAllByRole("radio", { name: /OpenAI/ });
    const openaiRadio = radios.find((r) => (r as HTMLInputElement).value === "openai_api")!;
    fireEvent.click(openaiRadio);
    expect(onChange).toHaveBeenCalledWith("openai_api");
  });
});

// ---------------------------------------------------------------------------
// Disabled state for unavailable providers (D8)
// ---------------------------------------------------------------------------

describe("UploadView unavailable providers are disabled", () => {
  it("disables provider radios that authStatus reports as unavailable", () => {
    const { container } = render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_OAUTH} />);

    // Scope queries to the root_provider fieldset to avoid collisions with
    // the subagent_auth fieldset which also has an "Anthropic API" radio.
    const providerFieldset = container.querySelector('fieldset:has(input[name="root_provider"])');

    const anthropicApiRadio = providerFieldset?.querySelector('input[value="anthropic_api"]') as HTMLInputElement;
    expect(anthropicApiRadio?.disabled).toBe(true);

    const openaiRadio = providerFieldset?.querySelector('input[value="openai_api"]') as HTMLInputElement;
    expect(openaiRadio?.disabled).toBe(true);

    const azureRadio = providerFieldset?.querySelector('input[value="azure_openai"]') as HTMLInputElement;
    expect(azureRadio?.disabled).toBe(true);

    const featherlessRadio = providerFieldset?.querySelector('input[value="featherless"]') as HTMLInputElement;
    expect(featherlessRadio?.disabled).toBe(true);
  });

  it("leaves available provider radios enabled", () => {
    const { container } = render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_OAUTH} />);
    const providerFieldset = container.querySelector('fieldset:has(input[name="root_provider"])');

    const oauthRadio = providerFieldset?.querySelector('input[value="anthropic_oauth"]') as HTMLInputElement;
    expect(oauthRadio?.disabled).toBe(false);
  });

  it("disables sub-agent auth radios that are unavailable", () => {
    const { container } = render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_OAUTH} />);
    const subagentFieldset = container.querySelector('fieldset:has(input[name="subagent_auth"])');

    const apiRadio = subagentFieldset?.querySelector('input[value="anthropic_api"]') as HTMLInputElement;
    expect(apiRadio?.disabled).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Sandbox radio group — Local + the three GPU clouds (RunPod, GCP, Azure)
// ---------------------------------------------------------------------------

describe("UploadView sandbox radios", () => {
  it("renders the Local, RunPod, GCP, and Azure sandbox options", () => {
    const { container } = render(<UploadView {...DEFAULT_PROPS} />);
    const fieldset = container.querySelector('fieldset:has(input[name="sandbox"])');
    for (const value of ["docker", "runpod", "gcp", "azure"]) {
      expect(fieldset?.querySelector(`input[value="${value}"]`)).not.toBeNull();
    }
  });

  it("checks the active sandbox radio", () => {
    const { container } = render(<UploadView {...DEFAULT_PROPS} sandbox="gcp" />);
    const gcp = container.querySelector('input[name="sandbox"][value="gcp"]') as HTMLInputElement;
    expect(gcp).toBeChecked();
  });

  it("calls onSandboxChange when a GPU cloud is selected", () => {
    const onChange = vi.fn();
    const { container } = render(<UploadView {...DEFAULT_PROPS} onSandboxChange={onChange} />);
    const azure = container.querySelector('input[name="sandbox"][value="azure"]') as HTMLInputElement;
    fireEvent.click(azure);
    expect(onChange).toHaveBeenCalledWith("azure");
  });
});

// ---------------------------------------------------------------------------
// Foundry provider + sub-agent auth (the funded demo default)
// ---------------------------------------------------------------------------

describe("UploadView Foundry option", () => {
  it("renders the Foundry provider radio, enabled when available", () => {
    const { container } = render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_FOUNDRY} />);
    const foundry = container.querySelector(
      'fieldset:has(input[name="root_provider"]) input[value="foundry"]'
    ) as HTMLInputElement;
    expect(foundry).not.toBeNull();
    expect(foundry.disabled).toBe(false);
  });

  it("calls onRootProviderChange('foundry') when Foundry is selected", () => {
    const onChange = vi.fn();
    const { container } = render(
      <UploadView {...DEFAULT_PROPS} authStatus={ALL_AVAILABLE} onRootProviderChange={onChange} />
    );
    const foundry = container.querySelector(
      'fieldset:has(input[name="root_provider"]) input[value="foundry"]'
    ) as HTMLInputElement;
    fireEvent.click(foundry);
    expect(onChange).toHaveBeenCalledWith("foundry");
  });

  it("renders the Foundry sub-agent auth radio, checked when selected", () => {
    const { container } = render(
      <UploadView {...DEFAULT_PROPS} authStatus={ONLY_FOUNDRY} subagentAuth="foundry" />
    );
    const foundry = container.querySelector(
      'fieldset:has(input[name="subagent_auth"]) input[value="foundry"]'
    ) as HTMLInputElement;
    expect(foundry).not.toBeNull();
    expect(foundry).toBeChecked();
  });

  it("shows a Recommended pill on the available Foundry option", () => {
    render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_FOUNDRY} rootProvider="foundry" subagentAuth="foundry" />);
    expect(screen.getAllByText("Recommended").length).toBeGreaterThanOrEqual(1);
  });

  it("disables the Foundry provider when its creds are missing", () => {
    const { container } = render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_OAUTH} />);
    const foundry = container.querySelector(
      'fieldset:has(input[name="root_provider"]) input[value="foundry"]'
    ) as HTMLInputElement;
    expect(foundry.disabled).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Picker captions + selected-option detail lines (visible, plain-language copy)
// ---------------------------------------------------------------------------

describe("UploadView picker captions + detail", () => {
  it("renders the plain-language group captions", () => {
    render(<UploadView {...DEFAULT_PROPS} authStatus={ALL_AVAILABLE} />);
    expect(screen.getByText("Where the reproduction runs its code.")).toBeInTheDocument();
    expect(screen.getByText("The model that reads the paper and drives the reproduction.")).toBeInTheDocument();
    expect(screen.getByText("How the code-writing sub-agents authenticate.")).toBeInTheDocument();
  });

  it("shows the selected provider's plain-language detail when available", () => {
    render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_FOUNDRY} rootProvider="foundry" />);
    expect(
      screen.getByText("Funded Claude models — Opus 4.8 and Sonnet 5 — via the Azure Foundry endpoint.")
    ).toBeInTheDocument();
  });

  it("shows the env hint as detail when the selected provider is unavailable", () => {
    render(<UploadView {...DEFAULT_PROPS} authStatus={ONLY_OAUTH} rootProvider="foundry" />);
    expect(
      screen.getByText("Set AZURE_FOUNDRY_API_KEY + AZURE_FOUNDRY_ENDPOINT and reload")
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Advanced options (GPU controls)
// ---------------------------------------------------------------------------

describe("UploadView advanced options", () => {
  it("renders Dynamic GPU checkbox", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByLabelText("Dynamic GPU")).toBeInTheDocument();
  });

  it("reflects dynamicGpu prop as checked", () => {
    render(<UploadView {...DEFAULT_PROPS} dynamicGpu={true} />);
    expect(screen.getByLabelText("Dynamic GPU")).toBeChecked();
  });

  it("calls onDynamicGpuChange when toggled", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} onDynamicGpuChange={onChange} />);
    fireEvent.click(screen.getByLabelText("Dynamic GPU"));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("renders VRAM number input", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByLabelText("VRAM (GB)")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// GPU count (user-selectable, 1-8, blank = auto)
// ---------------------------------------------------------------------------

describe("UploadView GPU count", () => {
  it("renders the GPU count number input", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByLabelText("GPU count")).toBeInTheDocument();
  });

  it("renders blank (auto) when gpuCount is unset (0)", () => {
    render(<UploadView {...DEFAULT_PROPS} gpuCount={0} />);
    expect(screen.getByLabelText("GPU count")).toHaveValue(null);
  });

  it("reflects a set gpuCount value", () => {
    render(<UploadView {...DEFAULT_PROPS} gpuCount={4} />);
    expect(screen.getByLabelText("GPU count")).toHaveValue(4);
  });

  it("calls onGpuCountChange with the typed value", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} onGpuCountChange={onChange} />);
    fireEvent.change(screen.getByLabelText("GPU count"), { target: { value: "4" } });
    expect(onChange).toHaveBeenCalledWith(4);
  });

  it("clamps values above 8 down to 8", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} onGpuCountChange={onChange} />);
    fireEvent.change(screen.getByLabelText("GPU count"), { target: { value: "20" } });
    expect(onChange).toHaveBeenCalledWith(8);
  });

  it("clamps values below 1 up to 1", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} onGpuCountChange={onChange} />);
    fireEvent.change(screen.getByLabelText("GPU count"), { target: { value: "0" } });
    expect(onChange).toHaveBeenCalledWith(1);
  });

  it("calls onGpuCountChange with 0 (auto) when the field is cleared", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} gpuCount={4} onGpuCountChange={onChange} />);
    fireEvent.change(screen.getByLabelText("GPU count"), { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith(0);
  });
});

// ---------------------------------------------------------------------------
// T10 — autonomous toggle (top-level, mirrors Minimize compute) + repoUrl
// (Advanced section)
// ---------------------------------------------------------------------------

describe("UploadView autonomous toggle", () => {
  it("renders the autonomous toggle, off by default", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByLabelText(/autonomous/i)).not.toBeChecked();
  });

  it("reflects autonomous=true prop as checked", () => {
    render(<UploadView {...DEFAULT_PROPS} autonomous={true} />);
    expect(screen.getByLabelText(/autonomous/i)).toBeChecked();
  });

  it("calls onAutonomousChange when toggled", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} onAutonomousChange={onChange} />);
    fireEvent.click(screen.getByLabelText(/autonomous/i));
    expect(onChange).toHaveBeenCalledWith(true);
  });
});

describe("UploadView repoUrl field (Advanced)", () => {
  it("renders a repo URL input", () => {
    render(<UploadView {...DEFAULT_PROPS} />);
    expect(screen.getByLabelText(/repo/i)).toBeInTheDocument();
  });

  it("reflects the repoUrl prop value", () => {
    render(<UploadView {...DEFAULT_PROPS} repoUrl="https://github.com/foo/bar" />);
    expect(screen.getByLabelText(/repo/i)).toHaveValue("https://github.com/foo/bar");
  });

  it("calls onRepoUrlChange when edited", () => {
    const onChange = vi.fn();
    render(<UploadView {...DEFAULT_PROPS} onRepoUrlChange={onChange} />);
    fireEvent.change(screen.getByLabelText(/repo/i), { target: { value: "https://github.com/foo/bar" } });
    expect(onChange).toHaveBeenCalledWith("https://github.com/foo/bar");
  });
});

// ---------------------------------------------------------------------------
// localStorage persistence (D3)
// ---------------------------------------------------------------------------

describe("localStorage persistence via providerPrefs", () => {
  it("readProviderPrefs returns empty object when localStorage is clean", () => {
    expect(readProviderPrefs()).toEqual({});
  });

  it("writeProviderPrefs + readProviderPrefs round-trips", () => {
    writeProviderPrefs({ root_provider: "openai_api", subagent_auth: "anthropic_oauth" });
    const restored = readProviderPrefs();
    expect(restored.root_provider).toBe("openai_api");
    expect(restored.subagent_auth).toBe("anthropic_oauth");
  });

  it("writeProviderPrefs merges with existing values", () => {
    writeProviderPrefs({ root_provider: "openai_api" });
    writeProviderPrefs({ ...readProviderPrefs(), dynamic_gpu: true });
    const prefs = readProviderPrefs();
    expect(prefs.root_provider).toBe("openai_api");
    expect(prefs.dynamic_gpu).toBe(true);
  });

  it("writeProviderPrefs + readProviderPrefs round-trips autonomous (T10)", () => {
    writeProviderPrefs({ autonomous: true });
    expect(readProviderPrefs().autonomous).toBe(true);
  });
});
