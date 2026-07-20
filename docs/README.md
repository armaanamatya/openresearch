<!-- doc-meta: status=current; last-verified=2026-07-20 -->
# Documentation index

Start with [../ONBOARDING.md](../ONBOARDING.md). This index deliberately
separates current operating guidance from historical evidence.

## Current documentation

| Need | Canonical document |
|---|---|
| Understand the product | [../README.md](../README.md), [../system_overview.md](../system_overview.md) |
| Set up and run locally | [reproduction.md](reproduction.md), [guides/setup-guide.md](guides/setup-guide.md) |
| Understand the system | [architecture.md](architecture.md), [design/](design/) |
| Operate cloud or local runs | [infra.md](infra.md), [runbooks/running-the-project.md](runbooks/running-the-project.md) |
| Test and troubleshoot | [runbooks/e2e-testing.md](runbooks/e2e-testing.md), [troubleshooting.md](troubleshooting.md) |
| Review enabled behavior | [reference/flags.md](reference/flags.md) (generated) |
| Apply durable engineering lessons | [guides/reliability-rules.md](guides/reliability-rules.md) |
| Retain run evidence | [policies/artifacts.md](policies/artifacts.md) |

## Historical material

- `runbooks/2026-*` and `superpowers/` are dated decisions and session
  handoffs. Consult them for provenance, not as the default operating path.
- `archive/` is frozen historical material.
- `runbooks/artifacts/`, `runs/`, and `best_runs/` are point-in-time evidence.

Documentation policy and freshness enforcement live in
[policies/documentation.md](policies/documentation.md). When prose disagrees
with code, code wins.
