"""
Unit tests for Phase 1e Unit B — global cross-paper infra memory +
ExperienceMemory (backend/agents/rlm/experience_memory.py).
Pure unit tests — no network, no subprocess (filesystem-backed via tmp_path).
"""

from backend.agents.rlm.experience_memory import ExperienceMemory
from backend.agents.rlm.failure_attribution import attribute_failure


def test_infra_attribution_writes_global_store(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=True)
    att = attribute_failure({"failure_class": "cuda_shlib_load", "error": "libcupti.so.12"})
    mem.record(att, arxiv_id="2605.15155", hint="prepend venv CUDA lib dirs to LD_LIBRARY_PATH")
    mem.record(att, arxiv_id="2512.99999", hint="prepend venv CUDA lib dirs to LD_LIBRARY_PATH")  # recurrence>=2
    hints = mem.infra_hints()
    assert any("LD_LIBRARY_PATH" in h for h in hints)
    # cross-paper: the SAME signature from a DIFFERENT arxiv_id contributed.
    assert (tmp_path / "_memory" / "infra").exists()


def test_method_attribution_never_enters_global_store(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=True)
    att = attribute_failure({"failure_class": "scope_shape_violation"})
    mem.record(att, arxiv_id="2605.15155", hint="emit explicit model_key/env/baseline per cell")
    assert mem.infra_hints() == []                       # method scope NEVER global (the routing invariant)
    infra_dir = tmp_path / "_memory" / "infra"
    assert not infra_dir.exists() or not any(infra_dir.iterdir())


def test_disabled_is_noop(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=False)
    mem.record(attribute_failure({"failure_class": "cuda_shlib_load"}), arxiv_id="x", hint="h")
    assert mem.infra_hints() == [] and mem.guidance_block(arxiv_id="x") == ""


def test_infra_hints_bounded_and_deduped(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=True)
    for i in range(12):
        att = attribute_failure({"failure_class": "network_flake", "error": f"conn reset {i}"})
        mem.record(att, arxiv_id="a"); mem.record(att, arxiv_id="b")
    assert len(mem.infra_hints()) <= 5                   # cap enforced
