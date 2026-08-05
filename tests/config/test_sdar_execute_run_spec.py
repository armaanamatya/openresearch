import json
import pathlib


def test_run_spec_selects_foundry_and_execute_and_primary():
    spec = json.loads(pathlib.Path("configs/sdar_execute_run_spec.json").read_text())
    assert spec["OPENRESEARCH_REPRODUCTION_MODE"] == "execute"
    assert spec["OPENRESEARCH_USE_AUTHOR_REPO"] == "1"
    assert spec["OPENRESEARCH_LIFECYCLE_PRIMARY"] == "1"
    # profile-level enable of the deterministic execute-synth floor (E4); the
    # GLOBAL default stays off pending paired A/B evidence -- do not read this
    # as a default flip.
    assert spec["OPENRESEARCH_EXECUTE_SYNTH"] == "1"
    # foundry routing (root + roles); accept either dedicated keys or ROLE_MODELS json
    assert "opus-foundry" in json.dumps(spec)
    assert "sonnet-foundry" in json.dumps(spec)
    for guard in ("OPENRESEARCH_ZERO_METRICS_GUARD", "OPENRESEARCH_EVAL_PROVENANCE_GUARD",
                  "OPENRESEARCH_ENV_LIVENESS_GATE", "OPENRESEARCH_EXTERNAL_VALIDATOR"):
        assert spec[guard] == "1"
    # driver-owned per-attempt keys must NOT be present (run_spec contract)
    for banned in ("OPENRESEARCH_SEED_BEST_ATTEMPT", "OPENRESEARCH_TARGET_BEST_FLOOR"):
        assert banned not in spec
