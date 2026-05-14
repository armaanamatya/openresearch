import { ReproLabClient } from "@/components/lab/repro-lab-client";
import { fetchRunById } from "@/lib/demo/server-run";

// The current run is identified by the `?projectId=` query param — that
// makes the URL the single source of truth, so a refresh or a shared
// link restores the exact run instead of dropping back to the upload
// view. No projectId → fresh upload view (the client may still
// auto-resume from localStorage).
export const dynamic = "force-dynamic";

export default async function LabPage({
  searchParams
}: {
  searchParams: Promise<{ projectId?: string | string[] }>;
}) {
  const params = await searchParams;
  const raw = params.projectId;
  const projectId = Array.isArray(raw) ? raw[0] : raw;
  const initialRun = projectId ? await fetchRunById(projectId) : null;
  return <ReproLabClient initialRun={initialRun} />;
}
