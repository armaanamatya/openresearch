# OmniZip (2511.14582) end-to-end run — Opus quality routing + sharded 7B hosting

Operator decisions (2026-06-11): launch after the All-CNN A/B pair frees its GPUs
(~17:00 worst case); Opus on quality-critical sub-agents only; BES off (single
clean attempt); core-claim scope. Pre-launch forensics on the Adam v6 + All-CNN
families confirmed the 06-09/06-11 remediation held (scope-shape repair in-loop,
graceful timeout recovery, no silent cell loss); no launch blockers found.

## Paper

OmniZip — training-free, audio-guided audio-video token compression for
OmniLLMs (Qwen2.5-Omni 7B/3B). Claims 2.51–3.42× prefill speedup and ~1.4×
lower peak memory at ~maintained accuracy on AVUT / VideoMME / ShortVid-Bench /
WorldSense. Paper hardware is an A6000 48 GB — matched here by 2×24 GB A5000
slots (`OPENRESEARCH_GPUS_PER_CELL=2`), so the 7B device_map-shards exactly into the
paper's budget. Official repo (github.com/KD-TAO/OmniZip) is blacklisted via the
hint's `blocked_resources` (clean-room rule). Core-claim scope pinned by the new
`PAPER_HINTS["2511.14582"]`: uncompressed vs OmniZip-r45 vs random-pruning on
WorldSense + ShortVid-Bench bounded subsets, 7B first, 3B mirror; AVUT/VideoMME
and the 35 %-retained setting are declared `scope.gaps`.

## Model-surface routing (this run)

| Surface | Mechanism | Model |
|---|---|---|
| Root REPL loop | `ROOT_MODELS["claude-oauth"].backend_kwargs` | claude-sonnet-4-6 |
| Navigation (`llm_query`/`rlm_query`) | `sub_backend_kwargs` | claude-haiku-4-5 |
| Implementer (`implement_baseline`, patch mode) | `OPENRESEARCH_ANTHROPIC_DEFAULT_MODEL` → `ctx.agent_model` | **claude-opus-4-8** |
| `plan_reproduction` / `propose_improvements` / rubric generation / repro-spec | **new** `OPENRESEARCH_PRIMITIVE_LLM_MODEL` → `_build_llm_client` ClaudeLlmClient pin | **claude-opus-4-8** |
| `verify_against_rubric` | deterministic PaperBench leaf scorer | (no LLM) |

`OPENRESEARCH_PRIMITIVE_LLM_MODEL` (added 2026-06-11, `run.py::_build_llm_client`)
is default-off: unset/blank preserves the legacy behavior (claude CLI default
model). It touches ONLY the Claude client branches; OpenAI/OpenRouter/Azure
branches and the rlm sub-backend are unaffected. Auth: `ANTHROPIC_API_KEY=` is
EMPTY in `.env` (verified in the live runs' `/proc` env), so both SDK surfaces
ride the OAuth subscription — Opus costs quota, not dollars.

## Launch mechanics

- Gate: `/tmp/omnizip_gate.sh` waits for the All-CNN batch parents (PIDs 2612103,
  2612251) to exit — their shutdown releases 4 GPU leases — then settles 120 s and
  execs `/tmp/launch_omnizip.sh`. (Needed because `batch_reproduce` gives up on
  lease acquisition after 30 min, far less than the hours-away GPU release.)
- Launch: `batch_reproduce.py 2511.14582 --gpus-per-run 4 --model claude-oauth
  --sandbox local --mode rlm --extra --paper-hint 2511.14582 --max-wall-clock 72000`
  plus env: `OPENRESEARCH_GPUS_PER_CELL=2`, `OPENRESEARCH_MATRIX_FINALIZE_RESERVE_S=7200`,
  `OPENRESEARCH_EXPERIMENT_STALL_S=7200`, `OPENRESEARCH_EXPERIMENT_GPU_LIVENESS=1`, the
  operator's validated feature-flag set, and the two Opus vars above. BES vars
  intentionally absent. Watchdog follows the explicit 20 h wall clock
  (`_arm_watchdog` uses the 14 h ceiling only when no `--max-wall-clock` is given
  — verified, not a live-run risk).
- Opus preflight in the launch script (`claude --model claude-opus-4-8 --print`)
  aborts before any GPU work if the model stops resolving.

## Rollback

Unset `OPENRESEARCH_PRIMITIVE_LLM_MODEL` + `OPENRESEARCH_ANTHROPIC_DEFAULT_MODEL`
(or set the latter back to claude-sonnet-4-6) — everything else is the
already-validated flag set from the 2026-06-10/11 A/B runs.
