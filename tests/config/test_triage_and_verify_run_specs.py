"""tests/config/test_triage_and_verify_run_specs.py

Guards the two-tier triage-funnel run-spec profiles
(``configs/triage_screen_run_spec.json`` Tier 1, ``configs/verify_deep_run_spec.json``
Tier 2) against the exact silently-inert-profile failure mode the repo's
2026-07-07 reliability postmortem flagged (``learn.md``): a reliability fix
that ships default-OFF protects nothing until it is turned on in a run-spec,
and a renamed/typo'd key in that run-spec fails silently (INIT-time, at $0,
per ``run_spec_contract``) rather than loudly.

Four checks per profile, mirroring ``tests/config/test_sdar_execute_run_spec.py``
and ``tests/rlm/test_run_spec_contract.py``:
  1. The file parses as a non-empty JSON object.
  2. Every key passes ``run_spec_key_applies`` (the F15 campaign-INIT round-trip
     predicate) — a rejected key would silently no-op for the whole campaign.
  3. No driver-owned per-attempt key is present — the campaign driver sets these
     itself per attempt; a run-spec that also sets one makes INIT fail-close on
     overlap (``backend/agents/rlm/CLAUDE.md`` "Reproduction campaign" section).
  4. Every flag name literally appears somewhere in ``backend/`` source — a
     grep-based check that the flag is real and not a typo of a real one.

Plus a handful of design-intent assertions: both tiers enable the full common
guard list; Tier 1 never carries a Tier-2-only guard; Tier 2 has a configured
external validator (REPRODUCED is unreachable without one) and
``OPENRESEARCH_GRADER_SAMPLES=3`` (the σ-gate memo's fidelity-critical
recommendation).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from backend.agents.rlm.models import resolve_root_model
from backend.agents.rlm.role_models import resolve_role_models
from backend.agents.rlm.run import assert_no_foundry_oauth_coresidency
from backend.agents.rlm.run_spec_contract import run_spec_key_applies

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

PROFILE_PATHS = {
    "triage_screen": REPO_ROOT / "configs" / "triage_screen_run_spec.json",
    "verify_deep": REPO_ROOT / "configs" / "verify_deep_run_spec.json",
}

# Driver-owned per-attempt keys (backend/agents/rlm/CLAUDE.md, "Reproduction
# campaign" -> "Profile configs/campaign_run_spec.json" bullet): the campaign
# driver sets these itself per attempt; a run-spec that also carries one makes
# `_load_run_spec`/INIT fail-closed on overlap.
_DRIVER_OWNED_KEYS = (
    "OPENRESEARCH_SEED_BEST_ATTEMPT",
    "OPENRESEARCH_TARGET_BEST_FLOOR",
    "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE",
    "OPENRESEARCH_MAX_RUN_GPU_USD",
)

# The two run_spec_contract special keys are meta-keys (they map to a
# DIFFERENT env var name), not OPENRESEARCH_* flag literals -- exempt from the
# "appears in backend source" grep since the check below greps for the key
# text itself, not its redirect target.
_SPECIAL_NON_FLAG_KEYS = {"models", "baseline_extra_guidance"}

# The 17 reliability/fabrication guards both tiers must enable (task spec):
# pure-stdlib, conservative, fail-soft, currently default-OFF.
COMMON_GUARDS = (
    "OPENRESEARCH_ZERO_METRICS_GUARD",
    "OPENRESEARCH_STUB_METRICS_GUARD",
    "OPENRESEARCH_EVAL_PROVENANCE_GUARD",
    "OPENRESEARCH_EVIDENCE_GATE",
    "OPENRESEARCH_PER_MODEL_STATUS_GATE",
    "OPENRESEARCH_EVIDENCE_FINGERPRINT",
    "OPENRESEARCH_METRICS_COMPLETENESS_CHECK",
    "OPENRESEARCH_FAILURE_CAPSULES",
    "OPENRESEARCH_CELL_RESUME_AUTO",
    "OPENRESEARCH_HARDEXIT_CLEANUP",
    "OPENRESEARCH_IMPL_ABANDON_GUARD",
    "OPENRESEARCH_PREFLIGHT_UNION_SCOPE",
    "OPENRESEARCH_GKE_SYNTH_CELL",
    "OPENRESEARCH_ORPHAN_GUARD",
    "OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE",
    "OPENRESEARCH_ENV_LIVENESS_GATE",
    "OPENRESEARCH_NO_LEARNING_SIGNAL_GATE",
)

# Tier-2-only trust/memory machinery; Tier 1 (cheap screen) must never carry it.
TIER2_ONLY_KEYS = (
    "OPENRESEARCH_TWO_AXIS_VERDICT",
    "OPENRESEARCH_EXTERNAL_VALIDATOR",
    "OPENRESEARCH_VALIDATOR_BACKEND",
    "OPENRESEARCH_VALIDATOR_MODEL",
    "OPENRESEARCH_VALIDATOR_PANEL_N",
    "OPENRESEARCH_CHAMPION_ARTIFACT",
    "OPENRESEARCH_LEAF_ACTUATE",
    "OPENRESEARCH_LEAF_ACTUATE_MAX_COST",
    "OPENRESEARCH_LEAF_ACTUATE_SEEDS",
    "OPENRESEARCH_POSITIVE_RECIPES",
    "OPENRESEARCH_NEGATIVE_LESSONS",
    "OPENRESEARCH_EXPERIENCE_MEMORY",
)

# "Execute-mode first" is the locked product decision: run the AUTHORS' published
# code behind a verified metrics shim rather than have the LLM re-implement the
# paper (from-scratch SDAR scored 0.0; the authors' trainer scored 0.456). The
# three capability flags that make that real, required in BOTH tiers.
#
# The mode is `auto`, NOT a pinned `execute`: `auto` resolves PER-PAPER (usable
# author repo => execute; none => a disclosed from-scratch attempt). A pinned
# `execute` makes `assert_execute_mode_stamped` RAISE on any paper that published
# no code -- which for a recall-critical triage funnel is the worst possible
# outcome: a hard crash on a paper that might well have reproduced, i.e. a FALSE
# NEGATIVE manufactured by the harness itself.
EXECUTE_MODE_KEYS = {
    "OPENRESEARCH_USE_AUTHOR_REPO": "1",
    "OPENRESEARCH_REPRODUCTION_MODE": "auto",
    # Without the deterministic understand->implement->run->verify FSM the run
    # depends on the root LLM self-sequencing -- the documented degenerate-loop
    # failure mode.
    "OPENRESEARCH_LIFECYCLE_PRIMARY": "1",
}

# Root tokens CLAUDE.md documents as unreliable harness drivers (they degenerate,
# need OPENRESEARCH_ARG_CONTRACTS, and are not paper-validated). A weak root that
# fails to drive the harness yields a spurious "couldn't reproduce" -- a FALSE
# NEGATIVE, the one error the recall-critical screen tier must not make. If a
# profile ever pins one anyway, ARG_CONTRACTS must be on alongside it.
WEAK_ROOT_TOKENS = frozenset({
    "qwen3-coder",
    "qwen3-coder-featherless",
    "kimi-k2.5",
    "azure-foundry",
    "grok",
    "grok-4.3",
    "foundry",
})


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def backend_source_corpus() -> str:
    """Concatenated text of every ``backend/**/*.py`` file.

    A single-read corpus so the per-flag existence check is O(files) once,
    not O(flags * files); mirrors a plain ``grep -r`` over ``backend/``.
    """
    chunks: list[str] = []
    for p in BACKEND_DIR.rglob("*.py"):
        try:
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def _flag_referenced(name: str, corpus: str) -> bool:
    """True iff *name* appears as a quoted string literal anywhere in *corpus*.

    Quoted-literal matching (not bare substring) is deliberate, and it screens
    out TWO distinct silently-inert-key classes:

    1. A typo'd/renamed flag -- appears nowhere.
    2. A pydantic-``Settings``-backed key (``backend/config.py`` resolves it via
       ``env_prefix="OPENRESEARCH_"``, and the field is only *mentioned* in a
       trailing comment, never read through ``os.environ.get("<NAME>")``). These
       PASS ``run_spec_key_applies`` (the prefix is valid) yet do nothing from a
       run-spec: ``backend.config``'s settings cache is already warm by the time
       ``cli.py::_load_run_spec`` writes into ``os.environ``, and nothing
       force-reloads it. Verified empirically for
       ``OPENRESEARCH_REPO_CLONE_TIMEOUT_S``/``_MAX_MB``, which is why they are
       NOT in these profiles. Such a key only takes effect as a SHELL EXPORT set
       before the process starts.

    So a key belongs in a run-spec only if some module actually reads it back
    out of ``os.environ`` -- which is exactly what a quoted-literal hit proves.
    """
    return f'"{name}"' in corpus or f"'{name}'" in corpus


# ---------------------------------------------------------------------------
# Per-profile structural checks (parametrized over both tiers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_name", sorted(PROFILE_PATHS))
def test_run_spec_profile_parses_as_nonempty_json_object(profile_name):
    path = PROFILE_PATHS[profile_name]
    assert path.exists(), f"missing profile: {path}"
    spec = _load(path)
    assert isinstance(spec, dict) and spec, f"{profile_name}: must be a non-empty JSON object"


@pytest.mark.parametrize("profile_name", sorted(PROFILE_PATHS))
def test_run_spec_profile_keys_all_pass_contract(profile_name):
    """Every key must pass run_spec_contract.run_spec_key_applies -- the same
    predicate campaign INIT calls before any money moves (F15). A rejected key
    here would silently no-op for the whole campaign instead of failing at $0.
    """
    spec = _load(PROFILE_PATHS[profile_name])
    rejected = [k for k in spec if not run_spec_key_applies(k)]
    assert not rejected, f"{profile_name}: key(s) rejected by run_spec_key_applies: {rejected}"


@pytest.mark.parametrize("profile_name", sorted(PROFILE_PATHS))
def test_run_spec_profile_excludes_driver_owned_keys(profile_name):
    spec = _load(PROFILE_PATHS[profile_name])
    present = [k for k in _DRIVER_OWNED_KEYS if k in spec]
    assert not present, f"{profile_name}: carries driver-owned per-attempt key(s): {present}"


@pytest.mark.parametrize("profile_name", sorted(PROFILE_PATHS))
def test_run_spec_profile_flag_names_exist_in_backend_source(profile_name, backend_source_corpus):
    spec = _load(PROFILE_PATHS[profile_name])
    missing = [
        k
        for k in spec
        if k not in _SPECIAL_NON_FLAG_KEYS and not _flag_referenced(k, backend_source_corpus)
    ]
    assert not missing, (
        f"{profile_name}: flag name(s) not found anywhere in backend/ source "
        f"(typo'd/renamed -> silently-inert profile): {missing}"
    )


# ---------------------------------------------------------------------------
# Design-intent assertions
# ---------------------------------------------------------------------------


def test_both_tiers_enable_the_full_common_guard_list():
    for profile_name, path in PROFILE_PATHS.items():
        spec = _load(path)
        for guard in COMMON_GUARDS:
            assert spec.get(guard) == "1", f"{profile_name}: {guard} must be enabled ('1')"


def test_triage_screen_never_carries_a_tier2_only_key():
    """Tier 1 is the cheap/broad screen and must never enable the Tier-2-only
    trust/memory machinery (external validator, champion-artifact, leaf-actuate,
    recipe/lesson/experience memory, two-axis verdict)."""
    spec = _load(PROFILE_PATHS["triage_screen"])
    present = [k for k in TIER2_ONLY_KEYS if k in spec]
    assert not present, f"triage_screen: unexpected Tier-2-only key(s): {present}"


def test_verify_deep_has_a_configured_external_validator():
    """REPRODUCED is unreachable without a configured validator (grader_transport
    .build_validator_client fail-closes when OPENRESEARCH_EXTERNAL_VALIDATOR is on
    but OPENRESEARCH_VALIDATOR_BACKEND is unset/misconfigured) -- Tier 2 REQUIRES
    a real backend, not just the master flag."""
    spec = _load(PROFILE_PATHS["verify_deep"])
    assert spec.get("OPENRESEARCH_EXTERNAL_VALIDATOR") == "1"
    assert spec.get("OPENRESEARCH_VALIDATOR_BACKEND"), (
        "verify_deep: OPENRESEARCH_VALIDATOR_BACKEND must be set alongside "
        "OPENRESEARCH_EXTERNAL_VALIDATOR, else the validator fail-closes at runtime"
    )


def test_verify_deep_grader_samples_is_three():
    """The sigma-gate memo (backend/agents/rlm/CLAUDE.md, grader-fidelity
    remediation section) recommends samples=3 for fidelity-critical papers."""
    spec = _load(PROFILE_PATHS["verify_deep"])
    assert spec.get("OPENRESEARCH_GRADER_SAMPLES") == "3"


def test_verify_deep_leaf_actuate_companions_are_wired():
    """OPENRESEARCH_LEAF_ACTUATE alone only activates its zero-cost repair arms
    (aggregation-gap audit, figure-sidecar render); the result-quality re-run
    and the multi-seed variance plan need their own sub-gates -- else those two
    arms are silently inert even with the master flag on."""
    spec = _load(PROFILE_PATHS["verify_deep"])
    assert spec.get("OPENRESEARCH_LEAF_ACTUATE") == "1"
    assert spec.get("OPENRESEARCH_LEAF_ACTUATE_MAX_COST") == "targeted_rerun"
    assert spec.get("OPENRESEARCH_LEAF_ACTUATE_SEEDS") == "1"


def test_both_tiers_are_execute_mode_first():
    """The locked product decision: run the authors' published code behind a
    verified metrics shim, driven by the deterministic lifecycle FSM -- not an
    LLM re-implementation from scratch."""
    for profile_name, path in PROFILE_PATHS.items():
        spec = _load(path)
        for key, expected in EXECUTE_MODE_KEYS.items():
            assert spec.get(key) == expected, (
                f"{profile_name}: {key} must be {expected!r} (execute-mode-first)"
            )


def test_neither_tier_pins_reproduction_mode_to_execute():
    """RECALL GUARD (the false-negative red line).

    A hard-pinned ``REPRODUCTION_MODE=execute`` makes ``assert_execute_mode_stamped``
    raise a RuntimeError for any paper with no usable author repo -- a great many
    papers. In a needle-in-a-haystack triage funnel that converts "this paper
    published no code" into "this paper crashed the run", which reads downstream as
    a false negative: a paper that might well have reproduced from scratch is
    discarded. ``auto`` is the only mode that both prefers the authors' code AND
    still gives a no-code paper a real, honestly-labelled attempt."""
    for profile_name, path in PROFILE_PATHS.items():
        mode = _load(path).get("OPENRESEARCH_REPRODUCTION_MODE")
        assert mode != "execute", (
            f"{profile_name}: REPRODUCTION_MODE=execute hard-fails every no-code paper "
            "(assert_execute_mode_stamped raises). Use 'auto' -- it resolves to execute "
            "when a repo exists and to a disclosed from-scratch attempt when it does not."
        )


def test_neither_tier_hard_pins_execute_owns_deps():
    """``OPENRESEARCH_EXECUTE_OWNS_DEPS=1`` is a HARD opt-in: ``_execute_owns_deps``
    returns True on the literal flag WITHOUT consulting repo_spec.json. Under an
    ``auto`` profile that falls back to from-scratch there IS no authors' conda env
    to own the dependencies, so a pinned 1 would suppress the harness's own pip/torch
    bootstrap and the training cell would die on `import torch`.

    Left UNSET it auto-derives per-paper from the resolved repo_spec mode (execute =>
    True, scratch => False) -- which is exactly the desired behavior on both branches
    of `auto`."""
    for profile_name, path in PROFILE_PATHS.items():
        assert "OPENRESEARCH_EXECUTE_OWNS_DEPS" not in _load(path), (
            f"{profile_name}: do not pin EXECUTE_OWNS_DEPS under REPRODUCTION_MODE=auto -- "
            "unset lets it derive from the resolved repo_spec mode, which is correct on "
            "both the execute and the from-scratch-fallback branch."
        )


@pytest.mark.parametrize("profile_name", sorted(PROFILE_PATHS))
def test_root_model_token_resolves(profile_name, monkeypatch):
    """A root-model pin must be a real, registered token -- a typo would silently
    fall through resolve_root_model's env/default chain at run time (picking some
    other model) instead of failing the profile validation.

    The subject here is REGISTRY MEMBERSHIP, not credential availability, but
    resolve_root_model also enforces a fail-closed credential gate. Satisfy that
    gate with fake keys for every backend the registry can require, so the test
    asserts what it claims to and stays green on a clean checkout. (It previously
    borrowed real keys from the developer's .env, so it silently only worked on
    a machine that happened to have the pinned model's provider configured.)"""
    for env_name in (
        "ANTHROPIC_API_KEY",
        "AZURE_FOUNDRY_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "FEATHERLESS_API_KEY",
        "OPENAI_ADMIN_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.setenv(env_name, "fake-key-for-test")

    spec = _load(PROFILE_PATHS[profile_name])
    token = spec["OPENRESEARCH_RLM_ROOT_MODEL"]
    assert resolve_root_model(token).key == token


@pytest.mark.parametrize("profile_name", sorted(PROFILE_PATHS))
def test_weak_root_requires_arg_contracts(profile_name):
    """Recall guard. A weak/degeneration-prone root (qwen/kimi/grok) turns a
    harness-driving failure into a spurious "couldn't reproduce" -- a false
    negative. Neither tier should pin one; if a future edit does,
    OPENRESEARCH_ARG_CONTRACTS must be enabled alongside it."""
    spec = _load(PROFILE_PATHS[profile_name])
    root = str(spec.get("OPENRESEARCH_RLM_ROOT_MODEL", "")).strip().lower()
    if root in WEAK_ROOT_TOKENS:
        assert spec.get("OPENRESEARCH_ARG_CONTRACTS") == "1", (
            f"{profile_name}: weak root {root!r} pinned without "
            "OPENRESEARCH_ARG_CONTRACTS=1"
        )


def test_verify_deep_role_models_parse_and_avoid_foundry_oauth_coresidency():
    """Tier 2 pins an Anthropic-on-Foundry root (opus-foundry, the best-known
    config). ``assert_no_foundry_oauth_coresidency`` RAISES if a run mixes
    anthropic-foundry with claude-oauth, so every Claude sub-role must also be
    pinned to Foundry -- exactly what configs/sdar_execute_run_spec.json does.
    This test runs the real resolver + the real assertion, so a future edit that
    drops the ``models`` pin fails here instead of at run start."""
    spec = _load(PROFILE_PATHS["verify_deep"])
    root = spec["OPENRESEARCH_RLM_ROOT_MODEL"]
    selection = resolve_role_models(planner_token=root, cli_models=spec["models"])

    assert_no_foundry_oauth_coresidency(root, selection)  # must not raise

    # And the validator stays cross-family from the executor, so the external
    # panel is a genuinely independent judge (role_models.separation_strength).
    assert selection.executor is not None
    assert selection.executor.family == "claude"
    assert spec["OPENRESEARCH_VALIDATOR_BACKEND"] == "openai"  # family "gpt"
