import "server-only";
import { backendBaseUrl } from "@/lib/demo/server-run";

export interface ModelChoice {
  id: string;
  label: string;
  provider: string;
}

/**
 * Server-side fetcher for the Sonnet/Opus dropdown source. Used by the
 * lab and demo page server components so the upload view ships its
 * options with the first paint. Returns `[]` when the backend is
 * unavailable; UploadView falls back to a "sonnet" default in that case.
 */
export async function fetchModels(): Promise<ModelChoice[]> {
  try {
    const response = await fetch(`${backendBaseUrl()}/models`, { cache: "no-store" });
    if (!response.ok) return [];
    const body = (await response.json()) as ModelChoice[];
    return Array.isArray(body) ? body : [];
  } catch {
    return [];
  }
}
