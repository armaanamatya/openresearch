"""tests/config/test_flag_decision_manifest.py

THE FORCING FUNCTION. Makes shipping an integrity guard "dark" impossible.

The repo's #1 systemic weakness (learn.md 2026-07-07): a reliability fix ships
default-OFF and protects NOTHING until it is wired into a run-spec profile --
and nobody notices, because there is no promotion pipeline. This test IS the
pipeline:

  1. ENUMERATION -- every integrity-token OPENRESEARCH_* flag in ``backend/``
     (a name carrying GUARD/GATE/KILL/VERDICT/VALIDATOR/CHECK/DETECTION as a
     ``_``-delimited segment) MUST appear in ``configs/flag_decisions.json``.
     A new guard therefore cannot merge without SOMEONE deciding about it.

  2. CONSISTENCY (both directions) -- a flag the manifest marks "on" for a
     profile MUST actually be enabled in that profile's JSON, an "off" MUST be
     absent/disabled, and a literal-value decision MUST match exactly. So the
     manifest can never drift from what ships -- drift would re-open the hole.

  3. TYPO GUARD -- every manifest flag literal must appear in ``backend/``
     source (mirrors ``test_triage_and_verify_run_specs.py``), so a renamed
     flag cannot silently rot in the manifest either.

Failure messages are deliberately actionable: they name the flag, point at the
manifest, and say what to do.

Env-hermetic: reads files off disk only; injects nothing into ``os.environ``.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
MANIFEST_PATH = REPO_ROOT / "configs" / "flag_decisions.json"

PROFILE_PATHS = {
    "triage_screen": REPO_ROOT / "configs" / "triage_screen_run_spec.json",
    "verify_deep": REPO_ROOT / "configs" / "verify_deep_run_spec.json",
}

# A flag is "integrity-shaped" iff its name carries one of these as a WHOLE
# ``_``-delimited segment. Segment (not substring) matching is deliberate and
# load-bearing: it INCLUDES WATCHDOG_KILL_SECONDS / DOOMED_KILL (KILL segment)
# and CImetric CHECK/VALIDATOR flags, while correctly EXCLUDING the advisory
# skill library (OPENRESEARCH_SKILLS / _SKILL_SELECT -- "sKILLs" only contains
# KILL as a substring, never as a segment) and CELL_CHECKPOINT_* / CELL_GRAD_-
# CHECKPOINT (CHECKPOINT != CHECK). Those were exactly the false positives in
# the audit's substring scan.
_INTEGRITY_TOKENS = frozenset(
    {"GUARD", "GATE", "KILL", "VERDICT", "VALIDATOR", "CHECK", "DETECTION"}
)

# Names that segment-match an integrity token but are NOT real feature flags.
# Each entry is reviewed and reasoned. Keep this list SHORT and justified --
# it is the one escape hatch from the enumeration, so an unjustified addition
# is how a real guard would slip through.
_ARTIFACT_EXCLUSIONS = frozenset(
    {
        # A frozen-surface PREFIX literal in harness_self_edit.py used as a
        # ``startswith("OPENRESEARCH_VALIDATOR_")`` guard over protected env
        # prefixes -- not a flag anything reads.
        "OPENRESEARCH_VALIDATOR_",
    }
)

_FLAG_RE = re.compile(r"OPENRESEARCH_[A-Z0-9_]+")

_TRUTHY = {"1", "true", "yes", "on"}
_DISABLING = {"", "0", "false", "no", "off"}
# Reserved manifest decision words (everything else is a literal-value decision).
_ON = "on"
_OFF = "off"


def _segments(flag: str) -> list[str]:
    return flag[len("OPENRESEARCH_") :].split("_")


def _is_integrity_shaped(flag: str) -> bool:
    return bool(_INTEGRITY_TOKENS.intersection(_segments(flag)))


@pytest.fixture(scope="module")
def backend_source_corpus() -> str:
    chunks: list[str] = []
    for p in BACKEND_DIR.rglob("*.py"):
        try:
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


@pytest.fixture(scope="module")
def integrity_flags_in_backend(backend_source_corpus) -> set[str]:
    """Every integrity-shaped OPENRESEARCH_* flag referenced in backend/ source,
    minus the reviewed artifact exclusions."""
    found = {
        m
        for m in _FLAG_RE.findall(backend_source_corpus)
        if _is_integrity_shaped(m)
    }
    return found - _ARTIFACT_EXCLUSIONS


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"missing manifest: {MANIFEST_PATH}"
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and isinstance(data.get("flags"), dict), (
        f"{MANIFEST_PATH}: must be a JSON object with a 'flags' object"
    )
    return data


def _load_profile(name: str) -> dict:
    return json.loads(PROFILE_PATHS[name].read_text(encoding="utf-8"))


def _flag_referenced(name: str, corpus: str) -> bool:
    return f'"{name}"' in corpus or f"'{name}'" in corpus


# ---------------------------------------------------------------------------
# 1. ENUMERATION -- no integrity flag may exist in backend/ without a decision
# ---------------------------------------------------------------------------


def test_every_integrity_flag_in_backend_has_a_manifest_decision(
    manifest, integrity_flags_in_backend
):
    decided = set(manifest["flags"])
    undecided = sorted(integrity_flags_in_backend - decided)
    assert not undecided, (
        "Integrity-shaped flag(s) exist in backend/ with NO decision in "
        f"configs/flag_decisions.json:\n  {undecided}\n"
        "A guard that ships without a profile decision protects NOTHING "
        "(learn.md 2026-07-07). Add each flag to configs/flag_decisions.json "
        "and DECIDE: 'on' (and wire it into that profile's run-spec JSON) or "
        "'off' (with a stated reason). If a name is a regex artifact / prefix "
        "literal and not a real flag, add it to _ARTIFACT_EXCLUSIONS here with "
        "a justification."
    )


def test_manifest_has_no_stale_integrity_entries(manifest, integrity_flags_in_backend):
    """A manifest key that segment-matches an integrity token but is NOT found
    in backend/ is a typo or a deleted flag -- fail so the manifest can't rot.
    (Non-integrity entries -- e.g. the advisory SKILL_* family -- are allowed
    and checked by the typo guard below instead.)"""
    stale = sorted(
        f
        for f in manifest["flags"]
        if _is_integrity_shaped(f)
        and f not in _ARTIFACT_EXCLUSIONS
        and f not in integrity_flags_in_backend
    )
    assert not stale, (
        f"Manifest lists integrity flag(s) not found in backend/ source: {stale}. "
        "Renamed or deleted? Update configs/flag_decisions.json."
    )


# ---------------------------------------------------------------------------
# 2. CONSISTENCY -- manifest decisions must match the live profile JSON
# ---------------------------------------------------------------------------


def _decision_matches_profile(decision: str, present: bool, raw_value) -> bool:
    val = str(raw_value).strip() if present else None
    low = val.lower() if val is not None else None
    if decision == _ON:
        return present and low in _TRUTHY
    if decision == _OFF:
        return (not present) or (low in _DISABLING)
    # literal-value decision (e.g. "partial", "openai", "gpt-5", "2")
    return present and val == decision


@pytest.mark.parametrize("profile_name", sorted(PROFILE_PATHS))
def test_manifest_decisions_match_profile(profile_name, manifest):
    spec = _load_profile(profile_name)
    violations: list[str] = []
    for flag, entry in manifest["flags"].items():
        decision = str(entry[profile_name])
        present = flag in spec
        if not _decision_matches_profile(decision, present, spec.get(flag)):
            actual = repr(spec.get(flag)) if present else "ABSENT"
            violations.append(
                f"  {flag}: manifest says {decision!r} but profile has {actual}"
            )
    assert not violations, (
        f"[{profile_name}] manifest<->profile DRIFT "
        f"(configs/flag_decisions.json vs {PROFILE_PATHS[profile_name].name}):\n"
        + "\n".join(violations)
        + "\nFix ONE of them so they agree: mark the flag 'on' and add \""
        + "<FLAG>\": \"1\" to the profile, or mark it 'off' and remove it from "
        "the profile (with a rationale). Drift re-opens the exact hole the "
        "manifest closes."
    )


def test_every_manifest_flag_declares_both_profiles(manifest):
    missing: list[str] = []
    for flag, entry in manifest["flags"].items():
        for profile_name in PROFILE_PATHS:
            if profile_name not in entry:
                missing.append(f"{flag}.{profile_name}")
        if not str(entry.get("rationale", "")).strip():
            missing.append(f"{flag}.rationale")
    assert not missing, (
        "Manifest entries missing a required field (each flag needs a decision "
        f"for BOTH profiles + a rationale): {missing}"
    )


# ---------------------------------------------------------------------------
# 3. TYPO GUARD -- every manifest flag literal must exist in backend/ source
# ---------------------------------------------------------------------------


def test_manifest_flag_names_exist_in_backend_source(manifest, backend_source_corpus):
    missing = [
        f for f in manifest["flags"] if not _flag_referenced(f, backend_source_corpus)
    ]
    assert not missing, (
        "Manifest flag name(s) not found anywhere in backend/ source "
        f"(typo'd/renamed -> deciding about a flag that doesn't exist): {missing}"
    )


# ---------------------------------------------------------------------------
# Anchor: the flags this task promoted out of the dark MUST stay wired.
# A regression here means a specific integrity guard went dark again.
# ---------------------------------------------------------------------------


def test_leaf_evidence_gate_is_wired_in_both_tiers(manifest):
    """The A7 per-leaf veto -- the corpus's highest-value correctness lever --
    was DARK in both profiles (they set the distinct verdict-level EVIDENCE_GATE).
    It must stay explicitly on in both tiers and enabled in both profiles."""
    entry = manifest["flags"]["OPENRESEARCH_LEAF_EVIDENCE_GATE"]
    assert entry["triage_screen"] == "on" and entry["verify_deep"] == "on"
    for profile_name in PROFILE_PATHS:
        assert _load_profile(profile_name).get("OPENRESEARCH_LEAF_EVIDENCE_GATE") == "1"


def test_spec_validator_block_stays_off_on_certification_tier(manifest):
    """SPEC_VALIDATOR_BLOCK drops rubric leaves => EASES the bar. Auto-lowering
    its own bar is misaligned with a fail-closed certification tier, so it must
    stay off and absent from verify_deep."""
    assert manifest["flags"]["OPENRESEARCH_SPEC_VALIDATOR_BLOCK"]["verify_deep"] == "off"
    assert "OPENRESEARCH_SPEC_VALIDATOR_BLOCK" not in _load_profile("verify_deep")
