import { DashboardShell } from "../components/dashboard/dashboard-shell";
import { sampleDashboardSnapshot } from "../lib/dashboard/fixtures";

export default function HomePage() {
  return <DashboardShell snapshot={sampleDashboardSnapshot} />;
}
