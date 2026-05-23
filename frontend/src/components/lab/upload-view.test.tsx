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
  },
  subagent_auth: { anthropic_api: true, anthropic_oauth: true },
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
  },
  subagent_auth: { anthropic_api: false, anthropic_oauth: true },
  defaults: {
    root_provider: "anthropic_oauth",
    root_model: "sonnet",
    subagent_auth: "anthropic_oauth",
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
  onRootProviderChange: NOP,
  onSubagentAuthChange: NOP,
  onDynamicGpuChange: NOP,
  onForceSingleGpuChange: NOP,
  onMaxGpuUsdPerHourChange: NOP,
  onVramGbChange: NOP,
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
});
