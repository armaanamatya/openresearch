import { PaperBenchClient } from "@/components/paperbench/paperbench-client";
import { listBundles, listRuns } from "@/lib/paperbench/runner";

export const dynamic = "force-dynamic";

export default async function PaperBenchPage() {
  const [bundleListing, runs] = await Promise.all([
    listBundles().catch(() => ({ bundles_root: "", bundles: [] })),
    listRuns().catch(() => []),
  ]);

  return <PaperBenchClient initialBundles={bundleListing} initialRuns={runs} />;
}
