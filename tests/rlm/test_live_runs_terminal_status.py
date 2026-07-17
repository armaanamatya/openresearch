

def test_live_run_state_coerces_structured_error_dict():
    """The 2026-07-03 degenerate-abort path wrote a structured dict into
    demo_status.json's `error` field, and /runs/latest 500'd on validation
    (BUG-NEW-045 family: reading a run's status must never 500). A dict error
    must coerce to its human-readable string."""
    from backend.services.events.live_runs import LiveRunState

    state = LiveRunState(
        projectId="prj_x",
        outputDir="runs/prj_x",
        runMode="rlm",
        status="failed",
        error={
            "primitive": "root_degenerate_loop",
            "outcome": "fatal",
            "error": "root emitted 3 consecutive iterations with no code block",
            "failure_class": "root_degenerate_loop",
            "suggested_fix": "Use a validated agentic root.",
            "metrics_present": False,
        },
    )
    assert isinstance(state.error, str)
    assert "root emitted 3 consecutive iterations" in state.error
    assert "root_degenerate_loop" in state.error


def test_live_run_state_plain_string_error_unchanged():
    from backend.services.events.live_runs import LiveRunState

    state = LiveRunState(
        projectId="prj_x", outputDir="runs/prj_x", runMode="rlm",
        status="failed", error="plain failure",
    )
    assert state.error == "plain failure"
