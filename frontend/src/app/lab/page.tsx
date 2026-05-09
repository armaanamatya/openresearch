import { LiveDemoClient } from "@/components/lab/live-demo-client";
import { loadDemoRun } from "@/lib/demo/node-runner";

export const dynamic = "force-dynamic";

export default async function LabPage() {
  const initialRun = await loadDemoRun();

  return <LiveDemoClient initialRun={initialRun} />;
}
