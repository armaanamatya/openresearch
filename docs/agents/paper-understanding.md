# `paper-understanding`

> **Stage:** builder (1 of 6) — runs first in the pipeline.
> **Registry entry:** [`AGENT_REGISTRY["paper-understanding"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/paper_understanding.py`](../../backend/agents/paper_understanding.py).

## Purpose

Extract a canonical structured summary of a research paper — its claims,
datasets, metrics, model architecture, training recipe, evaluation protocol,
hardware clues, and ambiguities — into a single Pydantic model
([`PaperClaimMap`](../../backend/agents/schemas.py)) that every downstream
agent reads.

Two execution modes share the same I/O contract:

- `run_with_sdk(...)` — invokes the configured agent runtime (Anthropic /
  OpenAI / etc.) to do LLM-powered extraction.
- `run_offline(...)` — deterministic, no-LLM heuristic extraction over the
  workspace `claim_map`. Used for tests, CI, and offline demos.

Both produce a `PaperClaimMap`, write it to
`<runs_root>/<project_id>/paper_claim_map.json`, and return it.

## Accepts

The Python signatures are:

```python
def run_offline(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
) -> PaperClaimMap: ...

async def run_with_sdk(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
    *,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
) -> PaperClaimMap: ...
```

### Positional arguments

| Argument | Type | Notes |
|---|---|---|
| `project_id` | `str` | Run identifier, e.g. `"prj_01a6d176008af0b3"`. Used as the directory name under `runs_root`. Must already exist. |
| `runs_root` | `pathlib.Path` | Absolute path to the runs directory. Honors `$REPROLAB_RUNS_ROOT`; see [`docs/design/unified-logging-launcher.md`](../design/unified-logging-launcher.md). |
| `workspace_claim_map` | `dict[str, Any]` | Parsed-PDF excerpt structure (see below). Produced upstream by the workspace ingestion step. |

### Keyword-only arguments (`run_with_sdk` only)

| Argument | Type | Default | Notes |
|---|---|---|---|
| `model` | `str \| None` | `None` | Override the registry's default model for this agent. |
| `provider` | `ProviderName \| str \| None` | `None` | `"anthropic"` or `"openai"`. Falls back to `get_settings().agent_provider`. |
| `runtime` | `AgentRuntime \| None` | `None` | Pre-built runtime instance — useful for tests that want to inject a fake. |

### Shape of `workspace_claim_map`

This is a plain `dict` (not a Pydantic model). It comes from the workspace
ingester and is **not** the same thing as the agent's output `PaperClaimMap`.
The agent reads the `entries` list and uses titles + excerpts as context.

| Key | Type | Meaning |
|---|---|---|
| `entries` | `list[dict]` | One entry per parsed paper section. |
| `entries[i].title` | `str` | Section heading, e.g. `"Section 6.1 Comparison of Surrogate Objectives"`. |
| `entries[i].excerpt` | `str` | Text excerpt (truncated to ~600 chars by `pipeline._truncate_excerpt`). |
| `entries[i]` (other keys) | `Any` | Ignored by `paper-understanding`; may be used by `artifact-discovery`. |

**Synthetic example input:**

```json
{
  "entries": [
    {
      "title": "Abstract",
      "excerpt": "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a surrogate objective..."
    },
    {
      "title": "Section 6.1 Comparison of Surrogate Objectives",
      "excerpt": "We compare several different surrogate objectives under different hyperparameters... Table 1 shows results across 7 MuJoCo environments."
    }
  ]
}
```

## Emits

### Return value

A populated [`PaperClaimMap`](../../backend/agents/schemas.py) instance.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `core_contribution` | `str` | — | One-paragraph plain-English summary of the paper's main claim. |
| `claims` | `list[dict[str, str]]` | `[]` | Each claim: `method`, `dataset`, `metric`, `expected_result`. |
| `datasets` | `list[DatasetRequirement]` | `[]` | See [io-contracts.md](io-contracts.md#datasetrequirement). |
| `metrics` | `list[MetricSpec]` | `[]` | See [io-contracts.md](io-contracts.md#metricspec). |
| `model_architecture` | `str` | `""` | Prose description (layers, sizes, activations, sharing). |
| `training_recipe` | `TrainingRecipe` | empty | See [io-contracts.md](io-contracts.md#trainingrecipe). |
| `evaluation_protocol` | `str` | `""` | How the paper evaluates (seeds, episodes, splits). |
| `hardware_clues` | `list[str]` | `[]` | Mentions of GPU count, training time, parallelism. |
| `ambiguities` | `list[Ambiguity]` | `[]` | See [io-contracts.md](io-contracts.md#ambiguity). |

`model_config = {"extra": "ignore"}` — extra LLM-generated keys are silently dropped.

### Side effect — file written

`<runs_root>/<project_id>/paper_claim_map.json` (UTF-8, indent=2). This is the
contract surface downstream agents read; the in-memory return value is a
convenience.

### `AgentOutput` envelope

When called through the orchestrator (not directly), the result is wrapped:

```json
{
  "agent_id": "paper-understanding",
  "status": "completed",
  "structured_outputs": { "paper_claim_map": { /* PaperClaimMap fields */ } },
  "summary": "Extracted 4 claims, 3 datasets, 4 metrics.",
  "exploration_log": { "..." : "..." }
}
```

### Real example output (redacted from `runs/prj_01a6d176008af0b3/paper_claim_map.json`)

```json
{
  "core_contribution": "PPO proposes a clipped surrogate objective L^CLIP that prevents excessively large policy updates without requiring second-order optimization...",
  "claims": [
    {
      "method": "PPO with clipped surrogate objective (epsilon=0.2)",
      "dataset": "7 MuJoCo environments via OpenAI Gym, 1M timesteps each",
      "metric": "Average normalized score across 21 runs (3 seeds x 7 envs)",
      "expected_result": "0.82 (best among all variants tested)"
    }
  ],
  "datasets": [
    {
      "name": "MuJoCo (OpenAI Gym)",
      "source": "OpenAI Gym with MuJoCo physics engine",
      "download_method": "pip install gym; requires MuJoCo license and mujoco-py",
      "size_estimate": "7 environments, 1M timesteps per run, 3 seeds",
      "notes": "Used for ablation study (Section 6.1) and comparison (Section 6.2)"
    }
  ],
  "metrics": [
    {
      "name": "Average Normalized Score",
      "definition": "For each environment, average total reward of last 100 episodes; shifted/scaled so random=0, best=1; averaged over 21 runs.",
      "target_value": "0.82 for PPO with clipping epsilon=0.2",
      "source_section": "Section 6.1, Table 1"
    }
  ],
  "model_architecture": "MuJoCo: MLP, 2 hidden layers of 64 units each, tanh; separate policy/value networks. Atari: same CNN as Mnih et al. 2016, shared policy/value heads.",
  "training_recipe": {
    "optimizer": "Adam",
    "learning_rate": "3e-4 (MuJoCo); 2.5e-4 linear annealed (Atari)",
    "batch_size": "2048 timesteps (MuJoCo); 128 actors x 128 steps (Atari)",
    "epochs_or_steps": "10 epochs per update; 1M timesteps total (MuJoCo)",
    "scheduler": "Linear annealing of LR and clip range (Atari only)",
    "other_hparams": { "clip_epsilon": "0.2", "gae_lambda": "0.95", "discount": "0.99" }
  },
  "evaluation_protocol": "3 random seeds per env; report mean over last 100 episodes for MuJoCo, full-training and last-100-episode means for Atari.",
  "hardware_clues": ["32-128 parallel actors mentioned for Roboschool", "Atari uses 8 parallel actors"],
  "ambiguities": [
    {
      "assumption_id": "A001",
      "detail": "Paper does not specify exact MuJoCo version.",
      "chosen_value": "MuJoCo 1.31 with mujoco-py 0.5.x",
      "evidence": ["Section 6.1 implicit via gym -v1 environments"],
      "risk": "medium"
    }
  ]
}
```

## Source

| What | Where |
|---|---|
| Pydantic models | [`backend/agents/schemas.py`](../../backend/agents/schemas.py) (`PaperClaimMap`, `Ambiguity`, `DatasetRequirement`, `MetricSpec`, `TrainingRecipe`) |
| Implementation | [`backend/agents/paper_understanding.py`](../../backend/agents/paper_understanding.py) |
| Prompt | `PAPER_UNDERSTANDING_PROMPT` in [`backend/agents/prompts/`](../../backend/agents/prompts/) |
| Registry entry | [`backend/agents/registry.py`](../../backend/agents/registry.py) (`AGENT_REGISTRY["paper-understanding"]`) |
| Tests | `tests/agents/test_paper_understanding.py` (if present) |

## Failure modes

See [errors.md](errors.md). Common shapes:

- **Workspace claim map empty** (`entries == []`) → offline mode returns a
  `PaperClaimMap` with empty lists; SDK mode raises before LLM call.
- **LLM JSON parse failure** → `_extract_json` falls back to `{}`, the agent
  emits a `PaperClaimMap` with empty fields and a non-empty `ambiguities` log.
- **Disk write failure** → propagates `OSError` to caller; orchestrator marks
  stage `failed` and records to the assumption ledger.

## Dashboard events

See [events.md](events.md). `paper-understanding` typically fires:

- `agent.started` (with `agent_id="paper-understanding"`)
- `agent.progress` (mid-run model-call updates)
- `agent.completed` or `agent.failed`
