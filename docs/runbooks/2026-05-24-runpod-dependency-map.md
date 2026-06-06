# RunPod dependency map — what the agent installs, why it sometimes fails, how to keep it working

**Status:** living doc as of 2026-05-24. Update on every infra-related fix.

## The five layers of RunPod dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│  L5 — Agent code: train.py + commands.json + requirements.txt       │
├─────────────────────────────────────────────────────────────────────┤
│  L4 — Backend auto-bootstrap (since e614593 / 3f210c6)              │
│        Backend prefixes `python -m pip install -r requirements.txt` │
│        to commands.json on runpod sandbox                            │
├─────────────────────────────────────────────────────────────────────┤
│  L3 — Pre-baked RunPod image (default: runpod/pytorch:               │
│        2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04)                   │
│        Includes: torch 2.1.0 (cu118), python 3.10, CUDA dev headers │
├─────────────────────────────────────────────────────────────────────┤
│  L2 — Pod hardware: GPU SKU resolved by dynamic GPU resolver         │
│        Default ladder: A6000 → L40S → A100_40 → A100_80 → H100_80   │
├─────────────────────────────────────────────────────────────────────┤
│  L1 — Pod creation: RunPod API, requires funded account             │
└─────────────────────────────────────────────────────────────────────┘
```

## Each layer's failure modes and the fix that landed

### L1 — Pod creation
| Failure | Symptom | Fix landed | Commit |
|---|---|---|---|
| Balance too low | HTTP 500 `"Your account balance is too low"` | Raised as `RUNPOD_BALANCE_TOO_LOW:` (non-retryable) | `aae89ad` |
| Capacity exhausted | HTTP 500 `"There are no instances currently available"` | Auto-escalates ladder via `gpu_escalated` SSE event with `reason=runpod_capacity` | `aae89ad` |
| SSH never ready | Pod boots but `_wait_for_pod_ssh` times out at 900s | Currently NOT auto-escalated. Job fails. **Open follow-up.** | — |
| Auth failure | HTTP 401/403 | Non-retryable, fail-fast | `aae89ad` |

### L2 — Hardware mismatch
| Failure | Symptom | Fix landed |
|---|---|---|
| CUDA OOM | Exit 137 or `RuntimeError: CUDA error: out of memory` | Auto-escalate ladder up to `dynamic_gpu_max_escalations` (default 2) | `6119e42` |
| Paper needs more VRAM than ladder offers | `GpuResolutionError` with cheapest SKU named in error | Fail fast with actionable error |
| Wrong SKU picked | Resolver chose A6000 when paper needs A100 | Override via `--vram-gb N` CLI flag | dynamic-gpu spec |

### L3 — Image issues
| Failure | Symptom | Fix landed |
|---|---|---|
| Missing CUDA dev headers | `bitsandbytes` / `flash-attn` / `deepspeed` fail to compile from source | Default reverted to `cuda-devel` image | `88c45b0` |
| torch version mismatch | Agent pins `torch==2.2.0`, base image has `2.1.0` — pip re-downloads 3GB | **Open follow-up.** Either pin agent to base image OR upgrade image |
| Python version mismatch | Agent expects 3.11, image has 3.10 | Doc the constraint in prompt; minor papers only |

### L4 — Backend bootstrap
| Failure | Symptom | Fix landed |
|---|---|---|
| Agent forgets `pip install` in commands.json | `ModuleNotFoundError: transformers` (or other dep) | Backend now auto-prefixes the install — paper-agnostic | `e614593` |
| Enum-vs-string sandbox_mode | Auto-install never fired (gate always false) | Substring match: `"runpod" in str(...).lower()` | `3f210c6` |
| `pip install -q -r requirements.txt && python …` chain | -q silences pip failure, `&&` propagates, train.py runs against half-installed env | New POD SETUP prompt block forbids `-q` and chained `&&`; backend auto-install IS the canonical path | `aac68fe` + `e614593` |
| Pip output not captured | Logs only show train.py output | `_combine_command_output` joins all stdout/stderr; captured correctly | (since `e614593`) |

### L5 — Agent code
| Failure | Symptom | Fix landed |
|---|---|---|
| Surrogate TinyLM instead of real Qwen | All metrics = 0 because synthetic | NO STUB prompt block forbids surrogates | `4b6798f` |
| Agent forgets `assert os.path.exists(...)` after download | ALFWorld/SearchQA setup silently produces empty data | DATASET SETUP prompt requires asserts after each download | `62c98e6` |
| β=2 written instead of paper's β=10 | Hyperparams diverge from rubric leaves | Rubric auto-checklist surfaces leaves; per-paper YAML override (e.g. SDAR's) carries exact hyperparams | `62c98e6` |
| Agent runs full eval on single GPU → timeout | 32-task × 2-env × 2-model eval takes >2hr | Scope guidance env var `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`; smaller `--max-wall-clock` | `9f5233c` |
| Metrics not written incrementally | Timeout produces zero metrics | Prompt instructs agent to write metrics.json in try/finally — **partially adopted; verify on next attempt** | (in extra-guidance) |

## Canonical `requirements.txt` pattern for any LLM paper

```txt
# DO NOT pin torch — base image already has compatible CUDA torch; pip will
# re-download 3GB on version mismatch. Use a range OR omit.
torch>=2.1.0,<2.3.0

# Real-Qwen / HF stack
transformers>=4.40.0
accelerate>=0.27.0
datasets>=2.18.0
huggingface_hub>=0.20.0

# Common training utils
numpy
tqdm
scipy
pyyaml
sentencepiece
einops

# Paper-specific (alfworld for SDAR, etc.)
alfworld>=0.3.5
```

## Canonical `commands.json` pattern on runpod

```json
[
  "alfworld-download 2>&1 || true",
  "python train.py"
]
```

**Why this is enough:**
- Backend auto-runs `python -m pip install --no-cache-dir -r requirements.txt` BEFORE the first entry
- `alfworld-download` is the ALFWorld setup step (idempotent — `|| true` tolerates re-runs)
- `python train.py` is the actual experiment

Anything else in commands.json is either redundant (pip install — backend handles) or unsafe (silenced errors, chained `&&` masking pip exits).

## Debugging recipe when a run fails

1. **Check `runs/<id>/final_report.json`** — `verdict` + `rubric.overall_score` + `reproduction_summary`.
2. **Check `runs/<id>/experiment_runs.jsonl`** — tail the last few entries. Look for:
   - `Successfully installed` → pip succeeded
   - `Traceback` → train.py failure
   - `timed out after Ns` → run_experiment wall-clock
3. **Check pod state**: `bash scripts/runpod_check.sh` — confirms account funded, 0 dangling pods.
4. **Check the resolved SKU**: `cat runs/<id>/rlm_state/gpu_plan.json | jq '.short_name, .vram_gb, .ladder_remaining'`.
5. **Check the agent's actual code**: `cat runs/<id>/code/commands.json && head -50 runs/<id>/code/train.py && head runs/<id>/code/requirements.txt`.

## Cost guard for follow-up sessions

Each failed Lane F-style attempt on A100 = ~$0.50–4 depending on duration. **Hard rule:** if the same failure mode appears twice on a single paper, STOP iterating on prompts — invest in a backend fix or a paper-specific yaml override instead. Burning $/hr to test prompt persuasion is the wrong gradient.

## Open follow-ups for the next session

1. **Pre-baked image with `transformers + accelerate + alfworld + tqdm`** — eliminates per-run pip cost (~5–20 min/attempt savings).
2. **SSH-wait-timeout escalation** — pod boots but never accepts SSH should advance the ladder.
3. **Multi-attempt continuation** — resume from `attempts/<prior_ts>/code/train.py` instead of agent rebuilding from scratch every time.
4. **Pin agent's `torch` to the base image version automatically** — backend could rewrite requirements.txt to replace `torch==X.Y.Z` with `torch>=2.0,<3.0` when running on runpod, avoiding the 3GB re-download.
5. **Incremental metrics.json writer** — backend helper that listens for partial metrics writes, so a timeout still produces a usable score.
6. **Pin runpod_image to a version with newer torch** — `runpod/pytorch:2.4.0-...` would match most papers' expectations.
