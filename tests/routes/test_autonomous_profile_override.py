"""Task 3: `apply_autonomous_profile_override` + the autonomous run-spec config.

When `StartRunRequest.autonomous` is True, this override forces GKE dispatch +
Opus-4.8-Foundry root + the canonical run-spec. OFF (default False) is the
identity function, mirroring `apply_sandbox_override`/`apply_provider_override`.

Grounded override vs. the original task brief: the brief's snippet used
`sandbox="gke"`, but `StartRunRequest.sandbox` is typed
`Literal["auto","docker","local","runpod","azure","gcp"]` — "gke" is not a
member of that Literal. The gke->gcp alias lives only in the separate
`backend.agents.execution` enum. "gcp" IS in the Literal and is exactly what
already selects `GkeJobBackend` byte-for-byte, so the override sets the
in-Literal value directly rather than depending on a downstream remap of an
invalid value that only survives today because `model_copy` skips
validation. Hence every assertion below checks `"gcp"`, not `"gke"`.
"""

from __future__ import annotations

import json
import pathlib

from backend.agents.rlm.run_spec_contract import run_spec_key_applies
from backend.services.events.live_runs import StartRunRequest, apply_autonomous_profile_override

_CONFIG_PATH = pathlib.Path("configs/autonomous_reproduction_run_spec.json")


def test_override_off_is_identity():
    r = StartRunRequest(sandbox="runpod", model="sonnet")
    assert apply_autonomous_profile_override(r) is r or apply_autonomous_profile_override(r) == r


def test_override_forces_gcp_opus_runspec():
    r = apply_autonomous_profile_override(StartRunRequest(autonomous=True))
    assert r.sandbox == "gcp"
    assert r.model == "opus-foundry"
    assert r.run_spec == "configs/autonomous_reproduction_run_spec.json"


def test_override_preserves_explicit_run_spec():
    """The `request.run_spec or _AUTONOMOUS_RUN_SPEC` fallback: a caller that
    already picked a run_spec keeps it — the override only fills the gap, it
    never clobbers an explicit choice."""
    r = apply_autonomous_profile_override(
        StartRunRequest(autonomous=True, run_spec="configs/sdar_execute_run_spec.json")
    )
    assert r.run_spec == "configs/sdar_execute_run_spec.json"
    # sandbox/model are still forced — autonomous wins on those two fields.
    assert r.sandbox == "gcp"
    assert r.model == "opus-foundry"


def test_autonomous_run_spec_keys_all_apply():
    spec = json.loads(_CONFIG_PATH.read_text())
    bad = [k for k in spec if not run_spec_key_applies(k) and k not in ("models", "baseline_extra_guidance")]
    assert bad == [], f"run-spec keys the contract rejects: {bad}"


def test_autonomous_run_spec_has_no_sdar_pins():
    spec = json.loads(_CONFIG_PATH.read_text())
    for forbidden in ("OPENRESEARCH_REPO_LOCAL_PATH", "OPENRESEARCH_REPO_COMMIT", "HF_HOME"):
        assert forbidden not in spec
