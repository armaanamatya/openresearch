"""Tests for backend.agents.rlm.root_progress — pure stage helper."""
from backend.agents.rlm.root_progress import infer_required_stage, REQUIRED_STAGES


# ---------------------------------------------------------------------------
# Canonical ladder cases (plan-mandated five)
# ---------------------------------------------------------------------------

def test_no_code_returns_need_baseline():
    stage = infer_required_stage(
        primitives=[],
        code_path_exists=False,
        env_built=False,
        total_run_experiments=0,
        total_verifications=0,
    )
    assert stage == "need_baseline"


def test_code_no_env_returns_need_environment():
    stage = infer_required_stage(
        primitives=["understand_section", "implement_baseline"],
        code_path_exists=True,
        env_built=False,
        total_run_experiments=0,
        total_verifications=0,
    )
    assert stage == "need_environment"


def test_code_and_env_no_experiment_returns_need_experiment():
    stage = infer_required_stage(
        primitives=["understand_section", "implement_baseline", "build_environment"],
        code_path_exists=True,
        env_built=True,
        total_run_experiments=0,
        total_verifications=0,
    )
    assert stage == "need_experiment"


def test_experiment_no_score_returns_need_verification():
    stage = infer_required_stage(
        primitives=["implement_baseline", "build_environment", "run_experiment"],
        code_path_exists=True,
        env_built=True,
        total_run_experiments=1,
        total_verifications=0,
    )
    assert stage == "need_verification"


def test_score_present_returns_can_finalize():
    stage = infer_required_stage(
        primitives=["implement_baseline", "build_environment", "run_experiment", "verify_against_rubric"],
        code_path_exists=True,
        env_built=True,
        total_run_experiments=1,
        total_verifications=1,
    )
    assert stage == "can_finalize"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_multiple_experiments_and_verifications_still_can_finalize():
    stage = infer_required_stage(
        primitives=[],
        code_path_exists=True,
        env_built=True,
        total_run_experiments=5,
        total_verifications=3,
    )
    assert stage == "can_finalize"


def test_return_value_always_in_required_stages():
    """All branches produce a value in the canonical set."""
    cases = [
        dict(code_path_exists=False, env_built=False, total_run_experiments=0, total_verifications=0),
        dict(code_path_exists=True,  env_built=False, total_run_experiments=0, total_verifications=0),
        dict(code_path_exists=True,  env_built=True,  total_run_experiments=0, total_verifications=0),
        dict(code_path_exists=True,  env_built=True,  total_run_experiments=1, total_verifications=0),
        dict(code_path_exists=True,  env_built=True,  total_run_experiments=1, total_verifications=1),
    ]
    for kwargs in cases:
        result = infer_required_stage(primitives=[], **kwargs)
        assert result in REQUIRED_STAGES, f"{result!r} not in REQUIRED_STAGES for {kwargs}"


def test_ladder_precedence_code_missing_wins_over_everything():
    """Even if counts > 0, missing code path triggers need_baseline."""
    stage = infer_required_stage(
        primitives=["run_experiment", "verify_against_rubric"],
        code_path_exists=False,
        env_built=True,
        total_run_experiments=3,
        total_verifications=2,
    )
    assert stage == "need_baseline"


def test_ladder_precedence_no_env_wins_over_run_counts():
    """Even if counts > 0, missing env triggers need_environment (ladder step 2)."""
    stage = infer_required_stage(
        primitives=[],
        code_path_exists=True,
        env_built=False,
        total_run_experiments=2,
        total_verifications=1,
    )
    assert stage == "need_environment"
