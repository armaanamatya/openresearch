"""TDD regression: forced-iteration guidance must reach the root model via stdout.

Bug (pre-fix): _intercepted_final_var returned the block string on refusal but
never printed the guidance.  execute_code() captures stdout via _capture_output;
the rlm core builds the root's next-turn context from REPLResult.stdout/stderr.
So the guidance reached the SSE/UI callback but NOT the root model's next turn —
the root was flying blind and degenerated into a FINAL_VAR loop.

Fix: print(f"[forced-iteration] {message}") at both refusal return sites in
_intercepted_final_var so the guidance is captured into REPLResult.stdout during
the root's code-block exec and surfaced to the root's next turn.
"""

from __future__ import annotations

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)

apply_forced_iteration_patch()


def _make_repl():
    from rlm.environments.local_repl import LocalREPL
    return LocalREPL()


# ---------------------------------------------------------------------------
# Helpers — minimal refusing policies
# ---------------------------------------------------------------------------


def _no_experiment_policy() -> ForcedIterationPolicy:
    """Policy that refuses because _total_run_experiments == 0 (BUG-NEW-046).

    This is the simplest refusing shape: the check fires at position 0.7,
    BEFORE the rubric snapshot is even consulted.  No rubric callable needed.
    """
    return ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (None, None, 0),
        current_iteration=lambda: 0,
        remaining_s=lambda: 9999,
        on_refusal=None,
        # _total_run_experiments defaults to 0 — refusal fires immediately.
    )


def _rubric_below_target_policy() -> ForcedIterationPolicy:
    """Policy that refuses on the standard below-target / below-floor check (#4)."""
    p = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.1, 0.9, 0),
        current_iteration=lambda: 0,
        remaining_s=lambda: 9999,
        on_refusal=None,
    )
    p._total_run_experiments = 1  # passes the no-experiment check
    return p


# ---------------------------------------------------------------------------
# Core regression: guidance in result.stdout (the failing test)
# ---------------------------------------------------------------------------


def test_refusal_guidance_present_in_stdout_no_experiment() -> None:
    """[FAILING pre-fix] The recovery guidance must appear in result.stdout.

    When the no-experiment branch (BUG-NEW-046) fires, the root's
    execute_code result must carry the guidance text in stdout so the rlm core
    surfaces it in the next-turn context.  Before the fix stdout is empty.
    """
    repl = _make_repl()
    repl.execute_code("report = {'x': 1}")

    policy = _no_experiment_policy()

    with forced_iteration_policy(policy):
        result = repl.execute_code('FINAL_VAR("report")')

    # The fix: a [forced-iteration] line must appear in stdout.
    assert "forced-iteration" in result.stdout, (
        f"Guidance absent from stdout (was: {result.stdout!r}). "
        "The root model never saw the recovery directive."
    )
    # At least one concrete next-step directive.
    assert "implement_baseline" in result.stdout or "run_experiment" in result.stdout, (
        f"Expected a concrete next-step in stdout, got: {result.stdout!r}"
    )


def test_refusal_guidance_present_in_stdout_below_target() -> None:
    """[FAILING pre-fix] Same assertion for the standard below-target refusal (#4).

    Covers the main refusal return site (different code path from the
    no-experiment branch above).
    """
    repl = _make_repl()
    repl.execute_code("report = {'score': 0.1}")

    policy = _rubric_below_target_policy()

    with forced_iteration_policy(policy):
        result = repl.execute_code('FINAL_VAR("report")')

    assert "forced-iteration" in result.stdout, (
        f"Guidance absent from stdout (was: {result.stdout!r}). "
        "The root model never saw the recovery directive."
    )
    assert "propose_improvements" in result.stdout or "implement_baseline" in result.stdout, (
        f"Expected a concrete next-step in stdout, got: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# find_final_answer contract is intact after the fix
# ---------------------------------------------------------------------------


def test_find_final_answer_still_returns_none_on_refusal() -> None:
    """The returned block string still satisfies the 'no final answer' contract.

    After the fix, _intercepted_final_var PRINTS the guidance (captured into
    stdout) AND RETURNS the block string unchanged.  The rlm core's
    find_final_answer must still see the returned value as "keep going".

    Simulates the rlm code path: execute_code("print(FINAL_VAR(...))") wraps
    the call in an outer print, so find_final_answer sees the RETURN VALUE
    printed to stdout — the block string containing the three required substrings.
    Our inner guidance line is also in stdout, but find_final_answer only needs
    the three substrings present somewhere; it does NOT need them in isolation.
    """
    from rlm.utils.parsing import find_final_answer

    repl = _make_repl()
    repl.execute_code("report = {'x': 1}")

    policy = _no_experiment_policy()

    with forced_iteration_policy(policy):
        # find_final_answer calls execute_code("print(FINAL_VAR('report'))")
        result = find_final_answer('FINAL_VAR("report")', environment=repl)

    assert result is None, (
        f"find_final_answer should return None on refusal, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Accept path is unchanged (no spurious print on success)
# ---------------------------------------------------------------------------


def test_accept_path_has_no_guidance_print() -> None:
    """On the accept path the interceptor must NOT print anything extra.

    A spurious [forced-iteration] line on every accepted FINAL_VAR would
    pollute the root's context and waste tokens.
    """
    repl = _make_repl()
    repl.execute_code("report = {'score': 0.95}")

    # Policy that accepts immediately (score >= target).
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.95, 0.7, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 9999,
    )
    policy._total_run_experiments = 1

    with forced_iteration_policy(policy):
        result = repl.execute_code('FINAL_VAR("report")')

    assert "forced-iteration" not in result.stdout, (
        f"Accept path must not print guidance. stdout={result.stdout!r}"
    )
    assert result.final_answer == "{'score': 0.95}"


# ---------------------------------------------------------------------------
# Two-experiment anti-pattern refusal also gets guidance in stdout
# ---------------------------------------------------------------------------


def test_two_experiment_refusal_guidance_in_stdout() -> None:
    """The two-experiment (PR-μ) branch also needs guidance in stdout."""
    repl = _make_repl()
    repl.execute_code("report = {'score': 0.1}")

    # Build a policy whose should_refuse_final_var (two-exp check) fires:
    # >=2 run_experiment calls in this iteration, last one a failure outcome.
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.1, 0.9, 1),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999,
    )
    policy._total_run_experiments = 2
    # Seed two experiments in the current iteration, last one "repairable".
    policy._experiments_in_iteration = ["ok", "repairable"]

    with forced_iteration_policy(policy):
        result = repl.execute_code('FINAL_VAR("report")')

    assert "forced-iteration" in result.stdout, (
        f"Two-experiment refusal guidance absent from stdout: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# Integration: guidance surfaces into root's next-turn prompt via format_iteration
# ---------------------------------------------------------------------------


def test_format_iteration_surfaces_guidance_to_next_prompt() -> None:
    """Guidance from a refused FINAL_VAR appears in the root's next-turn prompt.

    Chain under test:
      refused FINAL_VAR
        → [forced-iteration] line printed into REPLResult.stdout
          → CodeBlock(code=..., result=result) wraps that REPLResult
            → RLMIteration holds the code block
              → format_iteration() emits a user-role message whose content
                 contains the guidance — this is what the root model reads next.

    Closing the last link: the existing stdout tests prove guidance reaches
    REPLResult.stdout; this test proves that stdout is forwarded into the
    next-turn prompt the rlm core feeds back to the root model.
    """
    from rlm.core.types import CodeBlock, RLMIteration
    from rlm.utils.parsing import format_iteration

    repl = _make_repl()
    repl.execute_code("report = {'x': 1}")

    policy = _no_experiment_policy()

    with forced_iteration_policy(policy):
        result = repl.execute_code('FINAL_VAR("report")')

    # Guidance must already be in stdout (sanity-check the upstream link).
    assert "forced-iteration" in result.stdout

    # Build the iteration the way rlm.core.rlm does (lines 601-608).
    code = 'FINAL_VAR("report")'
    block = CodeBlock(code=code, result=result)
    iteration = RLMIteration(
        prompt="reproduce the paper",
        response=code,
        code_blocks=[block],
        final_answer=None,
        iteration_time=None,
    )

    messages = format_iteration(iteration)

    # format_iteration produces one assistant message + one user message per
    # code block.  The guidance must appear in at least one user-role message.
    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert user_contents, "format_iteration produced no user-role messages"

    combined = "\n".join(user_contents)
    assert "[forced-iteration]" in combined, (
        f"[forced-iteration] tag absent from next-turn prompt messages.\n"
        f"User message contents: {user_contents!r}"
    )
    assert "implement_baseline" in combined or "run_experiment" in combined, (
        f"Next-step directive absent from next-turn prompt messages.\n"
        f"User message contents: {user_contents!r}"
    )
