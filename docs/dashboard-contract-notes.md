# Dashboard Contract Notes

This branch implements the Issue `#21` dashboard as a safe scaffold because the codebase on `main` does not yet include the upstream dependencies listed in the issue body.

## What Is Missing On `main`

- No merged Next.js or React app shell from issue `#20`
- No backend event stream transport or API module from issues `#16`, `#18`, or `#11`
- No existing UI component system, data hooks, or app routing structure to extend

## Safe Integration Boundary

- The dashboard consumes a typed `DashboardSnapshot` with PRD-derived task and event records.
- `lib/dashboard/normalize.ts` is the only place that translates raw payloads into view-friendly topology, feed, citation, and lineage models.
- The panel components do not know whether data came from fixtures, SSE, or WebSocket transport.

## Contract Sources

- `docs/reprolab-agent-prd.md`
  - Shared State And Blackboard Model
  - Shared State Write Protocol
  - Agent Task Lifecycle
  - Citation Explorer
  - Real-Time Architecture

## Follow-Up Integration Work

- Replace `sampleDashboardSnapshot` with a transport-backed loader once the backend contract lands.
- Preserve the current task and event union shape unless backend work reveals a conflicting canonical schema.
- Add richer source drill-down and streaming state management after the upstream app shell exists.
