import "server-only";
import { backendBaseUrl } from "@/lib/demo/server-run";
import type { PipelineTopology } from "./topology";

export async function fetchTopology(): Promise<PipelineTopology | null> {
  try {
    const response = await fetch(`${backendBaseUrl()}/pipeline/topology`, {
      cache: "no-store"
    });
    if (!response.ok) return null;
    return (await response.json()) as PipelineTopology;
  } catch {
    return null;
  }
}
