#!/usr/bin/env python3
"""rlms spike — RLM Pivot Phase 2 fork-resolution groundwork (umbrella #64).

Purpose
-------
The canonical brief `docs/design/rlm-pivot-brief.md` §3 asserts the `rlms` PyPI
package (import name: `rlm`) IS Algorithm 1 and exposes a specific API.
`docs/design/phase2-analysis.md` risk R6 flagged that nobody had installed or
run it. This script verifies it empirically.

This is NOT Phase 2 implementation code. It touches nothing under
`backend/agents/rlm/`. It is a verification harness kept in `tools/` so the
result is reproducible.

What it checks
--------------
Runs the brief's intended Phase 1 spike: a minimal `RLM(custom_tools=...)` with
two mock domain primitives over a tiny mock "paper", and confirms that
Algorithm 1 runs, `custom_tools` are callable inside the REPL, and the run
terminates by returning a REPL variable via `FINAL_VAR`.

Two modes
---------
* mock  (default) — monkeypatches the rlm client factory with a deterministic
  scripted fake model. No API key, no network, no cost. This is the
  reproducible verification.
* live  (`--live`)  — runs a real `RLM` against OpenAI/Anthropic. Needs a valid
  key. (When this spike was authored the only key in the environment was an
  invalid `OPENAI_API_KEY`, so the live path 401'd at the API boundary — see
  docs/design/rlms-spike-report.md.)

Run
---
    .venv/bin/python tools/rlms_spike.py            # mock (default)
    .venv/bin/python tools/rlms_spike.py --live     # real API, needs a key
"""

from __future__ import annotations

import os
import sys
import traceback

# --- mock domain primitives -------------------------------------------------
# Stand-ins for ReproLab's real stage-agent functions. Each records that it was
# invoked, so the spike can prove `custom_tools` were callable IN THE REPL
# rather than merely accepted by the constructor.
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
    EVENTS.append(("iteration_complete", depth, iteration))


def on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
    EVENTS.append(("subcall_start", depth, model))


def on_subcall_complete(depth: int, model: str, duration: float, error: str | None) -> None:
    EVENTS.append(("subcall_complete", depth, model, error))


# --- mock paper + custom_tools (shared by both modes) -----------------------
MOCK_PAPER = {
    "paper_text": (
        "Mock paper. We propose Algorithm X, an on-policy RL method. It is "
        "trained on MockEnv-v1 with learning rate 3e-4, batch size 64, for 10 "
        "epochs. We report a mean_reward of 200 over 100 evaluation episodes. " * 8
    ),
    "paper_metadata": {"title": "Mock RL Paper", "sections": ["Method", "Experiments"]},
}

# custom_tools format verified against rlm 0.1.1: each entry is a dict
# {"tool": callable_or_value, "description": str} — NOT a bare callable.
CUSTOM_TOOLS = {
    "mock_understand_section": {
        "tool": mock_understand_section,
        "description": "Extract claims/datasets/metrics from a section text slice. Returns a dict.",
    },
    "mock_extract_hyperparameters": {
        "tool": mock_extract_hyperparameters,
        "description": "Extract hyperparameters from a text slice. Returns a dict.",
    },
}


# --- deterministic scripted fake model (mock mode) --------------------------
def _make_scripted_lm():
    """Build a fake BaseLM that drives a fixed two-iteration Algorithm-1 run."""
    from rlm.clients.base_lm import BaseLM
    from rlm.core.types import ModelUsageSummary, UsageSummary

    class ScriptedLM(BaseLM):
        """Deterministic fake root model. Ignores the prompt; replays a fixed
        trajectory keyed on the call count, so the RLM loop runs with zero
        network/cost while still exercising the real REPL + custom_tools."""

        def __init__(self) -> None:
            super().__init__(model_name="scripted-mock")
            self.turns = 0

        def completion(self, prompt) -> str:  # noqa: ANN001
            self.turns += 1
            if self.turns == 1:
                # Turn 1: write REPL code that calls the injected custom_tools.
                return (
                    "I will extract the paper's claims and hyperparameters.\n"
                    "```repl\n"
                    "claims = mock_understand_section(context['paper_text'])\n"
                    "hp = mock_extract_hyperparameters(context['paper_text'])\n"
                    "report = {'claims': claims, 'hyperparameters': hp}\n"
                    "print('built report in REPL:', report)\n"
                    "```\n"
                )
            # Turn 2+: terminate by returning the `report` REPL variable.
            return "Extraction complete.\nFINAL_VAR(report)"

        async def acompletion(self, prompt) -> str:  # noqa: ANN001
            return self.completion(prompt)

        def _usage(self):
            return ModelUsageSummary(
                total_calls=self.turns,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost=0.0,
            )

        def get_usage_summary(self):
            return UsageSummary(model_usage_summaries={self.model_name: self._usage()})

        def get_last_usage(self):
            return self._usage()

    return ScriptedLM()


def run_mock() -> int:
    """Deterministic spike: monkeypatch the client factory, run a scripted RLM."""
    try:
        import rlm.core.rlm as rlm_core
        from rlm import RLM
        from rlm.logger import RLMLogger
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot import `rlm` — {e!r}\n      Install with: .venv/bin/pip install rlms")
        return 1

    fake = _make_scripted_lm()
    # Inject the fake: rlm.core.rlm calls module-level get_client() to build the
    # client that LMHandler wraps. Replacing it makes the root model scripted.
    original_get_client = rlm_core.get_client
    rlm_core.get_client = lambda backend, backend_kwargs: fake  # noqa: ARG005

    logger = RLMLogger()
    max_iterations = 6
    try:
        rlm = RLM(
            backend="openai",  # irrelevant — get_client is patched
            backend_kwargs={"model_name": "scripted-mock"},
            environment="local",
            max_depth=2,
            max_iterations=max_iterations,
            custom_tools=CUSTOM_TOOLS,
            custom_sub_tools={},
            logger=logger,
            verbose=False,
            on_iteration_start=on_iteration_start,
            on_iteration_complete=on_iteration_complete,
            on_subcall_start=on_subcall_start,
            on_subcall_complete=on_subcall_complete,
        )
        print("[spike/mock] RLM(...).completion() with a scripted fake root model")
        result = None
        try:
            result = rlm.completion(MOCK_PAPER, root_prompt="Extract claims and hyperparameters.")
        finally:
            rlm.close()
    finally:
        rlm_core.get_client = original_get_client

    if result is None:
        print("FAIL: completion() returned nothing")
        return 1
    return _report(result, fake_turns=fake.turns, max_iterations=max_iterations, mode="mock")


def run_live() -> int:
    """Real spike against a hosted model. Requires a valid API key."""
    try:
        from rlm import RLM
        from rlm.logger import RLMLogger
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot import `rlm` — {e!r}")
        return 1

    model = os.environ.get("RLMS_SPIKE_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("FAIL: --live needs OPENAI_API_KEY (or set RLMS_SPIKE_MODEL + a key).")
        return 1

    logger = RLMLogger()
    max_iterations = 6
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model, "api_key": api_key},
        environment="local",
        max_depth=2,
        max_iterations=max_iterations,
        max_timeout=180.0,
        custom_tools=CUSTOM_TOOLS,
        custom_sub_tools={},
        logger=logger,
        verbose=False,
        on_iteration_start=on_iteration_start,
        on_iteration_complete=on_iteration_complete,
        on_subcall_start=on_subcall_start,
        on_subcall_complete=on_subcall_complete,
    )
    root_prompt = (
        "The REPL `context` is a dict with keys 'paper_text' and 'paper_metadata'. "
        "Call mock_understand_section and mock_extract_hyperparameters on "
        "context['paper_text'], assemble a dict `report`, then emit FINAL_VAR(report)."
    )
    print(f"[spike/live] RLM(...).completion()  model={model}")
    result = None
    try:
        result = rlm.completion(MOCK_PAPER, root_prompt=root_prompt)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: rlm.completion raised — {e!r}")
        traceback.print_exc()
    finally:
        rlm.close()
    if result is None:
        return 1
    return _report(result, fake_turns=None, max_iterations=max_iterations, mode="live")


def _report(result, fake_turns, max_iterations: int, mode: str) -> int:  # noqa: ANN001
    """Print the verification checks and return a process exit code."""
    iters = fake_turns if fake_turns is not None else len(
        [e for e in EVENTS if e[0] == "iteration_start"]
    )
    response_ok = bool(result.response and str(result.response).strip())
    terminated_before_cap = iters is not None and 0 < iters < max_iterations

    checks = {
        "completion() returned a non-empty response": response_ok,
        "Algorithm-1 root loop ran (>= 1 iteration)": bool(iters and iters >= 1),
        "respected the root-iteration cap": iters is not None and iters <= max_iterations,
        "custom_tools were callable inside the REPL": sum(TOOL_CALLS.values()) > 0,
        "terminated via FINAL_VAR before the cap": terminated_before_cap,
        "trajectory captured via RLMLogger": result.metadata is not None,
    }
    print("\n=== SPIKE VERIFICATION (mode=%s) ===" % mode)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("  --- observations ---")
    print(f"  [info] root iterations: {iters}")
    print(f"  [info] mock_understand_section calls: {TOOL_CALLS['understand_section']}")
    print(f"  [info] mock_extract_hyperparameters calls: {TOOL_CALLS['extract_hyperparameters']}")
    print(f"  [info] on_subcall_* events fired: {len([e for e in EVENTS if 'subcall' in e[0]])}")
    print(f"  [info] on_iteration_* events fired: {len([e for e in EVENTS if 'iteration' in e[0]])}"
          "  (rlm 0.1.1 declares these callbacks but never invokes them)")
    print(f"  [info] response (first 240 chars): {str(result.response)[:240]!r}")
    all_pass = all(checks.values())
    print(
        "\nSPIKE RESULT: "
        + ("PASS — rlms (rlm 0.1.1) runs Algorithm 1: custom_tools callable in the "
           "REPL, FINAL_VAR termination, trajectory logged."
           if all_pass else "PARTIAL/FAIL — see the FAIL lines above.")
    )
    return 0 if all_pass else 2


def main(argv: list[str]) -> int:
    if "--live" in argv:
        return run_live()
    return run_mock()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
