import type { DemoAccelerator, DemoGpuParallelism } from "@/lib/demo/demo-run-types";

const KEY = "openresearch:user-prefs";
// Provider selection is stored under its own key (D3) so it doesn't
// collide with the existing user-prefs shape.
const PROVIDER_KEY = "reprolab.lab.providerSelection";

export interface UserPrefs {
  model?: string;
  sandbox?: "auto" | "local" | "docker" | "runpod" | "azure" | "gcp";
  executionMode?: "efficient" | "max";
  splitRatio?: number;
}

export interface ProviderPrefs {
  root_provider?: string;
  subagent_auth?: string;
  dynamic_gpu?: boolean;
  force_single_gpu?: boolean;
  max_gpu_usd_per_hour?: number;
  vram_gb?: number;
  // Lane Q — minimize-compute mode. When true, the agent's prompt gets a
  // "reproduce the CLAIM, not the recipe" block that swaps slow paper
  // schedules for modern fast equivalents and annotates the substitutions
  // in scope.declared_reductions.
  minimize_compute?: boolean;
  gpu_parallelism?: DemoGpuParallelism;
  accelerator?: DemoAccelerator;
}

export function readProviderPrefs(): ProviderPrefs {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(PROVIDER_KEY);
    return raw ? (JSON.parse(raw) as ProviderPrefs) : {};
  } catch {
    return {};
  }
}

export function writeProviderPrefs(prefs: ProviderPrefs): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PROVIDER_KEY, JSON.stringify(prefs));
  } catch {
    // non-fatal
  }
}

export function readUserPrefs(): UserPrefs {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as UserPrefs) : {};
  } catch {
    return {};
  }
}

export function writeUserPref<K extends keyof UserPrefs>(key: K, value: UserPrefs[K]): void {
  if (typeof window === "undefined") return;
  try {
    const prefs = readUserPrefs();
    prefs[key] = value;
    window.localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    // localStorage may be disabled (private mode) — non-fatal.
  }
}
