# Agent Provider Resilience

ReproLab agent invocations now run through a typed resilience layer instead of
one-off provider string matching.

## Runtime Recovery

Each agent attempt is classified into provider-independent failures:

- `QuotaExhausted`
- `RateLimited`
- `TransientError`
- `TurnBudgetExhausted`
- `ToolBudgetExhausted`
- `WallClockExceeded`
- `AuthenticationError`
- `GuardViolation`
- `BudgetExhausted`

The default chain is bidirectional: an Anthropic primary falls back to OpenAI,
and an OpenAI primary falls back to Anthropic. Fallback carries a continuation
preamble with the previous provider's partial output and uses the same working
directory, so files already written to disk remain visible.

## Artifacts

Each SDK run writes:

- `runs/<project>/agent_telemetry.jsonl`
- `runs/<project>/cost_ledger.jsonl`
- `runs/<project>/fallback_summary.json`

The cost ledger is append-only. Token counts are preserved even when a model has
no known pricing entry; estimated USD is omitted for unknown models.

## Budget Flags

`python -m backend.cli reproduce ...` supports:

- `--max-usd <float>`
- `--max-wall-clock <seconds>`
- `--max-invocations agent=count[,agent=count]`

Budgets are checked before every provider attempt, including fallback attempts.

## Hermes Audit Codex Fallback

Hermes audits can fall back to the Codex CLI via ChatGPT OAuth after the API-key
OpenAI provider. The provider never reads OAuth tokens. It only checks that
`codex` is on PATH and that `~/.codex/auth.json` exists, then shells out to:

```bash
codex exec --skip-git-repo-check --ephemeral --ignore-user-config --ignore-rules --output-last-message <tmp-file> "<prompt>"
```

Verified locally with `codex-cli 0.125.0`.

Optional overrides:

- `REPROLAB_CODEX_CLI_PATH`
- `REPROLAB_CODEX_AUTH_PATH`

## RunPod Compatibility

This resilience layer wraps only provider invocations. Experiment execution
backends remain unchanged: Docker, local process, and RunPod still flow through
the existing `SandboxMode` and experiment runner paths. RunPod provisioning is
not triggered by fallback itself; provider fallback only affects agent text/tool
invocations before or around experiment execution.
