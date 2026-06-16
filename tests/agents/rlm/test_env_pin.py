"""D6a — env_pin: the harness owns core packages and neutralizes the agent's
conflicting re-pins on EVERY backend (not just runpod). Pure/stdlib, LLM-free.
"""
from __future__ import annotations

from backend.agents.rlm import env_pin


def test_strips_core_pins_on_local_cu121():
    reqs = [
        "torch==2.2.0",            # agent's conflicting torch — the redundancy bug
        "torchvision==0.17.0",
        "numpy==1.24.0",
        "transformers>=4.40",      # NOT core — must survive
        "datasets",
    ]
    kept, dropped = env_pin.harden_requirements(reqs, base_tag="cu121")
    assert "transformers>=4.40" in kept and "datasets" in kept
    assert any("torch==2.2.0" in d for d in dropped)
    assert any("torchvision" in d for d in dropped)
    assert any("numpy" in d for d in dropped)
    assert all("torch" not in k and "numpy" not in k for k in kept)


def test_allow_override_is_noop():
    reqs = ["torch==2.2.0", "flash-attn==2.5.0"]
    kept, dropped = env_pin.harden_requirements(reqs, base_tag="cu121", allow_override=True)
    assert kept == reqs and dropped == []


def test_unknown_base_tag_is_noop():
    # No PinSet → the harness installs no pin → it doesn't own the package → keep all.
    reqs = ["torch==2.2.0", "numpy==1.24.0"]
    kept, dropped = env_pin.harden_requirements(reqs, base_tag="azure-future")
    assert kept == reqs and dropped == []


def test_comments_and_options_preserved():
    reqs = [
        "# pinned for the paper",
        "-r base.txt",
        "--index-url https://example/simple",
        "torch==2.2.0",
        "scipy",
    ]
    kept, dropped = env_pin.harden_requirements(reqs, base_tag="cu121")
    assert "# pinned for the paper" in kept
    assert "-r base.txt" in kept
    assert "--index-url https://example/simple" in kept
    assert "scipy" in kept
    assert any("torch" in d for d in dropped)


def test_canonicalization_catches_name_variants():
    # Case-folding catches Torch / TORCHVISION; a substring like 'my-torch-helper'
    # is a DIFFERENT package and must be kept (no false-positive strip).
    reqs = ["Torch>=2.0", "TORCHVISION", "torchaudio==2.5.1", "my-torch-helper"]
    kept, dropped = env_pin.harden_requirements(reqs, base_tag="cu121")
    assert len(dropped) == 3  # Torch, TORCHVISION, torchaudio
    assert "my-torch-helper" in kept


def test_pin_install_specs():
    specs = env_pin.pin_install_specs("cu121")
    assert "torch==2.5.1" in specs and "torchvision==0.20.1" in specs
    assert env_pin.pin_install_specs("runpod") == []   # image carries torch
    assert env_pin.pin_install_specs("unknown") == []  # fail-soft


def test_base_tag_for():
    assert env_pin.base_tag_for("local", None) == "cu121"
    # C4 (2026-06-16): docker must NOT claim cu121. The .pth-following
    # LD_LIBRARY_PATH prepend that makes the cu121 pin dlopen-able lives only in
    # LocalProcessBackend; docker exec runs inside a container and never sees the
    # host venv. Claiming cu121 here would strip + re-pin torch WITHOUT the
    # lib-fix, so strip and lib-fix would disagree on scope. Fall back to "".
    assert env_pin.base_tag_for("docker", "ubuntu:22.04") == ""
    assert env_pin.base_tag_for("runpod", "runpod/pytorch:2.1.0-cuda11.8.0-runtime") == "runpod"
    assert env_pin.base_tag_for("azure", "some/image") == ""


def test_docker_tag_makes_strip_and_pin_both_noop():
    # The consequence of C4: under docker the harness owns NO core package, so
    # neither half of the cu121 mechanism fires — harden_requirements keeps the
    # agent's torch verbatim AND pin_install_specs installs nothing. Scope agrees.
    tag = env_pin.base_tag_for("docker", "ubuntu:22.04")
    reqs = ["torch==2.2.0", "numpy==1.24.0", "transformers>=4.40"]
    kept, dropped = env_pin.harden_requirements(reqs, base_tag=tag)
    assert kept == reqs and dropped == []          # nothing stripped
    assert env_pin.pin_install_specs(tag) == []    # nothing pinned
