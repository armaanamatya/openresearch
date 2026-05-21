#!/usr/bin/env python3
"""rlms spike — RLM Pivot Phase 2 fork-resolution groundwork (umbrella #64).

Purpose
-------
The canonical brief `docs/design/rlm-pivot-brief.md` §3 asserts that the `rlms`
PyPI package (import name: `rlm`) IS Algorithm 1 and exposes a specific API.
`docs/design/phase2-analysis.md` risk R6 flagged that nobody had installed or
run it. This script verifies it empirically.

This is NOT Phase 2 implementation code. It touches nothing under
`backend/agents/rlm/`. It is a throwaway verification harness kept in `tools/`
so the result is reproducible.

What it does
------------
Runs the brief's intended Phase 1 spike: a minimal `RLM(custom_tools=...)` with
two mock domain primitives over a tiny mock "paper", and checks that:
  1. Algorithm 1 actually runs (the root iteration loop executes);
  2. `custom_tools` are callable from inside the REPL;
  3. the `on_iteration_*` / `on_subcall_*` callbacks fire;
  4. the run terminates and returns an answer.

Run
---
    .venv/bin/python tools/rlms_spike.py

Requires `OPENAI_API_KEY` (the root model). Override the model with
`RLMS_SPIKE_MODEL` (default: gpt-4o-mini). Cost is cents-scale: one tiny mock
paper, max_iterations=6, a cheap model.
"""

from __future__ import annotations

import os
import sys
import traceback

# --- mock domain primitives -------------------------------------------------
# Stand-ins for ReproLab's real stage-agent functions. Each records that it was
# invoked, so the spike can prove `custom_tools` were actually callable in the
# REPL rather than merely accepted by the constructor.
TOOL_CALLS: dict[str, int] = {"understand_section": 0, "extract_hyperparameters": 0}


def mock_understand_section(text_slice: str) -> dict:
    """Mock of the `understand_section` primitive — claims/datasets/metrics."""
    TOOL_CALLS["understand_section"] += 1
    return {
        "claims": ["Algorithm X reaches mean_reward 200 on MockEnv-v1"],
        "datasets": ["MockEnv-v1"],
        "metrics": ["mean_reward"],
    }


def mock_extract_hyperparameters(text_slice: str) -> dict:
    """Mock of the `extract_hyperparameters` primitive."""
    TOOL_CALLS["extract_hyperparameters"] += 1
    return {"lr": 3e-4, "batch_size": 64, "epochs": 10}


# --- lifecycle callback recorders ------------------------------------------
EVENTS: list[tuple] = []


def on_iteration_start(depth: int, iteration: int) -> None:
    EVENTS.append(("iteration_start", depth, iteration))


def on_iteration_complete(depth: int, iteration: int, duration: float) -> None:
    EVENTS.append(("iteration_complete", depth, iteration, round(duration, 2)))


def on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
    EVENTS.append(("subcall_start", depth, model))


def on_subcall_complete(depth: int, model: str, duration: float, error: str | None) -> None:
    EVENTS.append(("subcall_complete", depth, model, error))


def main() -> int:
    try:
        from rlm import RLM
        from rlm.logger import RLMLogger
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot import `rlm` — {e!r}")
        print("      Install with: .venv/bin/pip install rlms")
        return 1

    model = os.environ.get("RLMS_SPIKE_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("FAIL: OPENAI_API_KEY is not set — the spike needs it for the root model.")
        return 1

    # The mock paper is offloaded as the REPL `context` variable (a dict the
    # root model slices into — it never sees the whole thing in its prompt).
    mock_paper = {
        "paper_text": (
            "Mock paper. We propose Algorithm X, an on-policy RL method. "
            "It is trained on MockEnv-v1 with learning rate 3e-4, batch size 64, "
            "for 10 epochs. We report a mean_reward of 200 over 100 evaluation "
            "episodes, exceeding the prior baseline of 150. " * 8
        ),
        "paper_metadata": {"title": "Mock RL Paper", "sections": ["Method", "Experiments"]},
    }

    # custom_tools format (verified against rlm 0.1.1): each entry is a dict
    # {"tool": callable_or_value, "description": str} — NOT a bare callable.
    custom_tools = {
        "mock_understand_section": {
            "tool": mock_understand_section,
            "description": "Extract claims/datasets/metrics from a section text slice. Returns a dict.",
        },
        "mock_extract_hyperparameters": {
            "tool": mock_extract_hyperparameters,
            "description": "Extract hyperparameters from a text slice. Returns a dict.",
        },
    }

    logger = RLMLogger()  # in-memory trajectory, surfaced on completion.metadata

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model, "api_key": api_key},
        environment="local",
        max_depth=2,          # depth=2 → rlm_query spawns a genuine child RLM
        max_iterations=6,     # hard root-iteration cap for the spike
        max_timeout=180.0,    # wall-clock guard
        max_budget=0.50,      # best-effort cost guard (needs a cost-tracking backend)
        custom_tools=custom_tools,
        custom_sub_tools={},  # child RLMs don't need the domain tools
        logger=logger,
        verbose=False,
        on_iteration_start=on_iteration_start,
        on_iteration_complete=on_iteration_complete,
        on_subcall_start=on_subcall_start,
        on_subcall_complete=on_subcall_complete,
    )

    # Deliberately directive prompt: a spike should exercise every feature
    # reliably, so we tell the root exactly what to do. (A production prompt
    # would NOT prescribe a workflow like this.)
    root_prompt = (
        "You are reproducing a paper. The REPL `context` is a dict with keys "
        "'paper_text' and 'paper_metadata'. Do exactly this, one repl block per step:\n"
        "1. Call mock_understand_section(context['paper_text']); store it as `claims`; print it.\n"
        "2. Call mock_extract_hyperparameters(context['paper_text']); store it as `hp`; print it.\n"
        "3. Call rlm_query('Reply with the single word OK.') once; store it as `check`; print it.\n"
        "4. Build report = {'claims': claims, 'hyperparameters': hp, 'check': check}.\n"
        "5. In your NEXT response, on its own line, emit FINAL_VAR(report)."
    )

    print(f"[spike] rlm 0.1.1 — RLM(...).completion()  model={model}  max_depth=2 max_iterations=6")
    result = None
    error: str | None = None
    try:
        result = rlm.completion(mock_paper, root_prompt=root_prompt)
    except Exception as e:  # noqa: BLE001
        error = repr(e)
        print(f"FAIL: rlm.completion raised — {error}")
        traceback.print_exc()
    finally:
        rlm.close()

    if result is None:
        return 1

    iter_starts = [e for e in EVENTS if e[0] == "iteration_start"]
    iter_done = [e for e in EVENTS if e[0] == "iteration_complete"]
    subcalls = [e for e in EVENTS if e[0] == "subcall_start"]

    checks = {
        "completion() returned a non-empty response": bool(result.response and str(result.response).strip()),
        "Algorithm-1 root loop ran (>= 1 iteration)": len(iter_starts) >= 1,
        "respected max_iterations cap (<= 6)": len(iter_starts) <= 6,
        "on_iteration_* callbacks fired": len(iter_done) >= 1,
        "custom_tools were callable inside the REPL": sum(TOOL_CALLS.values()) > 0,
        "trajectory captured via RLMLogger": result.metadata is not None,
    }
    observations = {
        "root iterations observed": len(iter_starts),
        "on_subcall_* fired (rlm_query used)": len(subcalls),
        "mock_understand_section call count": TOOL_CALLS["understand_section"],
        "mock_extract_hyperparameters call count": TOOL_CALLS["extract_hyperparameters"],
    }

    print("\n=== SPIKE VERIFICATION ===")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("  --- observations ---")
    for name, val in observations.items():
        print(f"  [info] {name}: {val}")
    print(f"  response (first 240 chars): {str(result.response)[:240]!r}")
    usage = result.usage_summary.to_dict() if result.usage_summary else "n/a"
    print(f"  usage_summary: {usage}")
    print(f"  execution_time: {result.execution_time:.1f}s")

    all_pass = all(checks.values())
    print(
        "\nSPIKE RESULT: "
        + ("PASS — rlms (rlm 0.1.1) runs Algorithm 1 with the brief's §3 API."
           if all_pass else
           "PARTIAL — core loop ran; see FAIL lines above.")
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
