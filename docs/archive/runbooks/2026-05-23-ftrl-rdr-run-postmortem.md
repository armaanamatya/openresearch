# Postmortem — FTRL RDR run scoring 3.2% (2026-05-23)

End-to-end RDR-mode reproduction of the **FTRL paper** (Wolczyk et al.,
ICML 2024 Spotlight: *"Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem"*) on a vendored PaperBench
bundle. Project id `pb_ftrl_1779576942`.

The run completed end-to-end and produced a real grading number. The
**rubric score landed at 3.2% (15 / 178 leaves graded as passing)** —
not because the system was broken, but because of three compounding
issues that each leaked downstream into the grade. This document is the
record of what each issue was, what it cost the run, and the fix
status.

## Run config

| field | value |
|---|---|
| paper | `ftrl` (PaperBench bundle, ICML 2024 spotlight) |
| project_id | `pb_ftrl_1779576942` |
| mode | `rdr` (rubric-driven, no RLM repair) |
| sandbox | `runpod` COMMUNITY (RTX 4090) |
| coding agents | `claude-agent-sdk` via OAuth (Claude Code subscription) |
| grading | initial: Claude (failed silently); re-graded: GPT-4o (`--provider openai`) |
| total wall time | ~16 hr (multiple iterations + waits + manual fixes) |
| RunPod cost | ~$0.10 (one ~18 min successful experiment) |
| OpenAI grading | ~$0.50–1.00 (178 leaves @ batch_size=15) |
| final rubric_score | **0.0322** (15 / 178 leaves passed) |

## Final result

```json
{
  "verdict": "failed",
  "rubric": {
    "overall_score": 0.0322,
    "leaf_count": 178,
    "graded": 177,
    "degraded": false
  }
}
```

15 leaves scored 1.0, all related to the **Robotic Sequence SAC agent
architecture** (4-layer MLP, LeakyReLU, LayerNorm-after-first-only,
SAC, auto-alpha, 100k replay buffer, per-stage heads, etc.). 162 leaves
scored 0 — these covered the paper's NetHack and Montezuma's Revenge
fine-tuning experiments which the run never actually ran.

## D1 — `SandboxConfig.workdir = "/work"` mismatched the codegen contract's `/code` convention

- **Symptom:** First experiment attempt failed instantly with
  `python3: can't open file '/code/collect_saves.py': [Errno 2] No
  such file or directory`. Subsequent attempts hit the same path on
  every command in `commands.json`.
- **Cause:** `backend/services/runtime/interface.py` defaulted
  `SandboxConfig.workdir = "/work"`, so all three sandbox backends
  (Docker, RunPod, local) mounted the project at `/work` and ran
  commands with `cwd=/work`. But `_sandbox_contract.py` told codegen
  agents only that the project was "mounted at the container working
  dir" without naming the path — so agents emitted `/code`-prefixed
  absolute paths (the standard Docker convention). The runtime didn't
  resolve them and shells reported "No such file or directory".
- **Fix:**
  - `backend/services/runtime/interface.py:48` — changed default
    `workdir: str = "/work"` → `"/code"`.
  - `backend/agents/prompts/_sandbox_contract.py` — explicit
    "mounted READ-ONLY at `/code`" so future codegen never has to
    guess.
- **Verified:** After the fix, `run_experiment` resolved
  `/code/collect_saves.py` and other agent-generated paths.

## D2 — Claude Code SDK threw `"Claude Code returned an error result: success"` on 57 / 63 clusters

- **Symptom:** 57 of 63 cluster coding agents died mid-run. Every
  failure logged the same exception:
  ```
  Exception: Claude Code returned an error result: success
  run_rdr[pb_ftrl_1779576942]: cluster <id> failed: Exception: Claude Code returned an error result: success
  ```
  In `experiment_runs.jsonl` this manifested as 9 of the 12 experiment
  attempts returning `success=False, metrics_count=0`.
- **Cause:** Likely a transient Claude Code subscription / rate-limit
  state — the SDK reported `is_error=True` *while also* sending
  `subtype="success"` in the result envelope. A working
  `claude --print "ping"` proves only the *subscription* works; the
  SDK quota has its own ceiling and was exhausted partway through the
  63-cluster batch.
- **Impact:** The 6 clusters that *did* complete only covered the
  Robotic Sequence environment (`train.py`, `environment.py`,
  `replay_buffer.py`, `sac.py`, `models.py`, `bc_train.py`). The
  paper's NetHack and Montezuma experiments — which together dominate
  the rubric — were never coded. This alone caps the achievable score
  at ≈10–15% of leaves.
- **Status:** Not yet fixed. The SDK error path needs a retry / exponential
  backoff loop, and the `_finalize_cluster_failure` helper should mark
  clusters as recoverable rather than terminal so a later resume can
  retry them. Note for future runs: when the operator's Claude Code
  subscription is fresh, run with `--provider openai` to use the
  OpenAI runtime path which is per-token-billed and does not hit this
  error mode.

## D3 — codegen produced syntactically-broken Python (missing `i` in `import`, `""` instead of `"""`)

- **Symptom:** Even after D1 was fixed and the experiment found the
  files, every run aborted at module import:
  ```
  File "/workspace/.../work/train.py", line 1
      mport os, json, time, argparse, numpy as np, torch
            ^^
  SyntaxError: invalid syntax
  ```
  Then:
  ```
  File "/workspace/.../work/replay_buffer.py", line 15
      """
      ^
  SyntaxError: unterminated triple-quoted string literal (detected at line 55)
  ```
  Then `models.py`, `sac.py`, `bc_sac.py`, `expert_buffer.py`,
  `montezuma_knowledge_retention.py`, `nethack_env_mock.py`,
  `robotic_sequence_env.py`, `verify_appo_params.py`, `verify_env.py`
  — 9 files in total.
- **Cause:** Two distinct codegen artifacts:
  1. `train.py` line 1 was literally `mport os, ...` — the leading
     `i` was missing. Looks like a tool-call truncation by the Claude
     Code SDK during file write.
  2. 8 other files opened with `""\nDocstring text\n"""` instead of
     `"""\nDocstring text\n"""` — the opening triple-quote was
     truncated to a 2-char empty string, so Python parsed `""` as an
     empty string literal, then choked on the next non-import line.
- **Fix:** Manual `StrReplace` edits to fix line 1 of each broken
  file. See git history of `runs/pb_ftrl_1779576942/code/*.py`. CRLF
  → LF normalisation pass on all 21 `.py` files for safety.
- **Status:** The codegen path needs a syntax-validation gate (the
  artifact-merge step in `rdr/controller.py:_merge_cluster_files`
  is the natural place to add `ast.parse()` per-file and reject
  artifacts that don't parse).

## D4 — initial scoring silently produced 0 / 178 graded leaves because the LLM client was Claude (also rate-limited)

- **Symptom:** First successful experiment run ended with
  `rubric.graded = 0, overall_score = 0.0` even though the experiment
  produced 27 metrics and `degraded = false`.
- **Cause:** `score_reproduction(rubric, project_dir, ctx.llm_client,
  ...)` uses whatever LLM client was configured for the run. The
  default for this run was Claude (Anthropic), which was the same
  rate-limited path that killed the codegen step. Each batch of 15
  leaves silently swallowed an exception via
  ```
  try:
      raw = llm_client.complete(...)
  except Exception:
      logger.warning(...)
  ```
  in `backend/evals/paperbench/leaf_scorer.py`. Result: every leaf
  got the default `"ungraded"` justification with score 0.0, and the
  surface-level rubric showed `graded=0`.
- **Fix:** Re-ran with `--provider openai` which routes the LLM
  client to GPT-4o. Got `graded=177/178` and the real
  `overall_score=0.0322`.
- **Status:** Not yet fixed structurally. The leaf scorer should
  warn-and-promote: if the per-batch grader exception count exceeds
  some threshold, fail the entire scoring pass loudly rather than
  return a silent zero. Right now a transient grader outage looks
  identical to a legitimate "experiment produced nothing" outcome.

## What the 15 passed leaves tell us

The 15 leaves that scored 1.0 all describe the **architecture and
training setup** of the Robotic Sequence SAC agent — not the actual
research claims of the paper. Examples:

- "The models are confirmed to be implemented as a 4-layer MLP with 256 neurons each."
- "The implementation uses Leaky-ReLU activations in the policy and Q-value function."
- "The replay buffer is confirmed to have a capacity of 100,000 trajectories."
- "The entropy coefficient in SAC is confirmed to be tuned automatically."

These are static-code-inspection leaves — the grader confirms them by
reading the file contents, not by checking the experiment's measured
behaviour. The leaves that *would* require measured behaviour
(e.g. "BC fine-tuning achieves average return 6000 on Montezuma's
Revenge", "EWC converges around 5e7 steps") all scored 0 because the
NetHack and Montezuma experiments never ran (D2).

## Pipeline that worked

For the record, this is the path that actually produced a graded
report:

1. Initial run (failed at experiment step due to D1)
2. Code-level fix to D1 (`workdir = "/code"`)
3. Resume #1 — found D3 (`mport`); fixed manually
4. Resume #2 — found `replay_buffer.py` syntax error; fixed
5. Resume #3 — found `models.py` syntax error; fixed
6. Resume #4 (CRLF→LF pass) — experiment ran 18 min, produced 27
   metrics, but D4 silenced the grader
7. Resume #5 with `--provider openai --max-repair-iterations 0` —
   final score 0.0322

## Logs

### Final report extract

```text
verdict: failed
rubric.overall_score: 0.0322
rubric.leaf_count: 178
rubric.graded: 177
rubric.degraded: false
clusters_total: 63
clusters_failed: 57
repair_iterations: 0
total_cost_usd (from ledger): 0.0000  # Claude OAuth subscription, not API-billed
```

### One canonical D2 traceback (Claude Code SDK)

```text
File "C:\Users\Armaan\Desktop\openresearch\.venv\Lib\site-packages\claude_agent_sdk\_internal\query.py", line 852, in receive_messages
    raise Exception(message.get("error", "Unknown error"))
Exception: Claude Code returned an error result: success

run_rdr[pb_ftrl_1779576942]: cluster 45c909a4-75fc-4c43-94a6-9cfd055979e5 failed: Exception: Claude Code returned an error result: success
```

### One canonical D3 syntax error from RunPod

```text
File "/workspace/openresearch/pb-ftrl-1779576942/pb-ftrl-1779576942-f6ad9847/work/models.py", line 154
    """
    ^
SyntaxError: unterminated triple-quoted string literal (detected at line 157)
```

### Successful experiment (12th and final attempt)

`runs/pb_ftrl_1779576942/experiment_runs.jsonl[-1].metrics`:

```json
{
  "final_mean_return_last100ep": -88.83,
  "success_rate": 0.0,
  "total_episodes": 500,
  "total_steps": 100000,
  "final_alpha": 0.0231,
  "num_hidden_layers": 4,
  "hidden_dim": 256,
  "activation": "leaky_relu",
  "layer_norm_count_in_policy": 1,
  "layer_norm_after_first_only": 1,
  "num_stage_heads_policy": 3,
  "num_stages": 3,
  "replay_buffer_capacity": 100000,
  "start_steps": 10000,
  "critic_weight_decay": 0,
  "algorithm": "SAC",
  "env": "RoboticSequence",
  "req1_4layer_256_mlp": 1,
  "req2_leaky_relu": 1,
  "req3_layernorm_first_only": 1,
  "req4_sac_implemented": 1,
  "req5_start_steps_uniform_random": 1,
  "req6_replay_buffer_100k": 1,
  "req7_terminal_no_bootstrap": 1,
  "req8_auto_entropy_tuning": 1,
  "req9_per_stage_heads": 1,
  "req10_critic_not_regularized": 1
}
```

The agent trained for 100k steps on the Robotic Sequence env, hit
`success_rate=0.0` (didn't solve the task), but verified all 10
self-reported architectural requirements.

## Action items (for the next FTRL attempt)

1. **D2 (codegen reliability):** add retry + exponential backoff
   around the Claude Code SDK call site at `backend/agents/runtime/claude_runtime.py:74`. Promote `is_error=True` results that carry `subtype="success"` to a typed retryable error rather than raw `Exception`.
2. **D3 (codegen syntax safety):** add `ast.parse()` validation in
   `rdr/controller.py:_merge_cluster_files` and refuse to write files
   that don't parse. Failing fast at write time gives a clear repair
   target.
3. **D4 (silent grader):** in `backend/evals/paperbench/leaf_scorer.py`
   `score_reproduction`, count batch grader exceptions and abort with
   a typed error if they exceed `len(leaves) * 0.5` rather than
   returning a silent zero.
4. **For the next manual run:** prefer `--provider openai` end-to-end
   when the operator's Claude Code subscription is depleted —
   per-token billing avoids D2 entirely. Or wait for subscription
   refresh and run with default Claude. Use `--max-repair-iterations 0`
   to skip the repair loop when the goal is system validation, not
   convergence.
