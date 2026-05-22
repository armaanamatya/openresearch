import { LabShell } from "@/components/lab/lab-shell";
import { fetchRunById } from "@/lib/demo/server-run";
import { fetchRecentRuns } from "@/lib/runs/server-list";
import { fetchModels } from "@/lib/models/server-fetch";

// The `/demo` route mirrors `/lab` but presents the workflow with the
// mythological agent names (Reader, Forge, …) and a step-through tour
// overlay. The unlock gate (proxy.ts) is scoped to `/demo` and
// `/api/demo`, so this route remains the secret-protected presentation
// surface while `/lab` and `/library` are open for internal use.
export const dynamic = "force-dynamic";

export default async function DemoPage({
  searchParams
}: {
  searchParams: Promise<{ projectId?: string | string[] }>;
}) {
  const params = await searchParams;
  const raw = params.projectId;
  const projectId = Array.isArray(raw) ? raw[0] : raw;
  const [initialRun, initialRecents, models] = await Promise.all([
    projectId ? fetchRunById(projectId) : Promise.resolve(null),
    fetchRecentRuns(8),
    fetchModels()
  ]);
  return (
    <LabShell
      initialRun={initialRun}
      initialRecents={initialRecents}
      initialModels={models}
      presentationMode="demo"
    />
  );
}
