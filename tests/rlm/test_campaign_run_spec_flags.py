"""Phase-0 config-integrity: campaign_run_spec.json must enable the reliability
flags AND every key must pass the run-spec contract (a rejected key fails
campaign INIT at $0). Guards against typos + driver-owned-key overlap."""
import json
from pathlib import Path

from backend.agents.rlm.run_spec_contract import run_spec_key_applies

_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "configs" / "campaign_run_spec.json"

# Reliability flags this plan enables (byte-identical when off; only alter
# broken-case paths — see backend/agents/rlm/CLAUDE.md "2026-07-07" section).
_RELIABILITY_FLAGS = (
    "OPENRESEARCH_PREFLIGHT_UNION_SCOPE",
    "OPENRESEARCH_GKE_SYNTH_CELL",
    "OPENRESEARCH_IMPL_ABANDON_GUARD",
)

# Driver-owned per-attempt keys the profile must NEVER carry (campaign INIT
# fail-closes on overlap — backend/agents/rlm/CLAUDE.md campaign section).
_DRIVER_OWNED = frozenset({
    "OPENRESEARCH_SEED_BEST_ATTEMPT",
    "OPENRESEARCH_TARGET_BEST_FLOOR",
    "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE",
    "OPENRESEARCH_MAX_RUN_GPU_USD",
})


def _load_spec() -> dict:
    return json.loads(_SPEC.read_text(encoding="utf-8"))


def test_every_spec_key_passes_the_contract():
    spec = _load_spec()
    for key in spec:
        assert run_spec_key_applies(key), f"{key} would be rejected at campaign INIT"


def test_no_driver_owned_keys_in_profile():
    spec = _load_spec()
    overlap = set(spec) & _DRIVER_OWNED
    assert not overlap, f"driver-owned keys must not live in the profile: {overlap}"


def test_reliability_flags_enabled():
    spec = _load_spec()
    for flag in _RELIABILITY_FLAGS:
        assert spec.get(flag) == "1", f"{flag} must be enabled (=\"1\") for campaigns"
