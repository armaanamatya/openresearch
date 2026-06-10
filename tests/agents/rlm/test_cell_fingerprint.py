"""Tests for cell_fingerprint — the resume re-run predicate's content hash.

Pure: no GPUs, no torch, no network.  The suite proves the load-bearing
guarantees the resume scheduler relies on:

  * Determinism — same inputs → same hash, across calls and key order.
  * Split — editing the env-specific helper flips THAT env's fingerprint but
    NOT an unrelated env's (the env-vs-shared split that lets resume re-run only
    the cells an edit touched).
  * Sensitivity — a cell-param change or an allow-listed flag change flips it.
  * Shared-helper sensitivity — editing a shared module flips every env.
  * Fail-soft — a missing helper contributes the sentinel, never raises, and a
    present→absent transition flips the hash.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm.cell_fingerprint import (
    ENV_HELPER_FILES,
    FLAG_ALLOWLIST,
    MISSING_SENTINEL,
    SHARED_HELPER_FILES,
    _hash_file,
    compute_fingerprint,
    dep_files_for_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _seed_helpers(code_dir: Path, *, marker: str = "v1") -> None:
    """Write all env-specific + shared helper files with distinct content."""
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in {*ENV_HELPER_FILES.values(), *SHARED_HELPER_FILES}:
        (code_dir / name).write_text(f"# {name} {marker}\n", encoding="utf-8")


def _cell(env: str, **over) -> dict:
    """An alfworld/search_qa/webshop cell mirroring code/cells.json entries."""
    cell: dict = {
        "id": f"qwen3_1_7b__sdar__{env}__s42",
        "model_id": "Qwen/Qwen3-1.7B",
        "model_key": "qwen3_1_7b",
        "baseline": "sdar",
        "env": env,
        "seed": 42,
        "steps": 150,
        "group_size": 8,
        "tasks_per_batch": 16,
        "max_new_tokens": 256,
        "max_turns": 8,
        "est_vram_gb": 18.0,
    }
    cell.update(over)
    return cell


# ---------------------------------------------------------------------------
# _hash_file
# ---------------------------------------------------------------------------

class TestHashFile:
    def test_hashes_bytes_deterministically(self, tmp_path):
        f = tmp_path / "h.py"
        f.write_text("hello", encoding="utf-8")
        assert _hash_file(f) == _hash_file(f)
        assert _hash_file(f) != MISSING_SENTINEL

    def test_distinct_content_distinct_hash(self, tmp_path):
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("alpha", encoding="utf-8")
        b.write_text("beta", encoding="utf-8")
        assert _hash_file(a) != _hash_file(b)

    def test_missing_file_returns_sentinel(self, tmp_path):
        assert _hash_file(tmp_path / "nope.py") == MISSING_SENTINEL

    def test_directory_returns_sentinel(self, tmp_path):
        # A directory is not a readable file — fail-soft to the sentinel.
        assert _hash_file(tmp_path) == MISSING_SENTINEL


# ---------------------------------------------------------------------------
# dep_files_for_env
# ---------------------------------------------------------------------------

class TestDepFilesForEnv:
    def test_alfworld_includes_its_helper_plus_shared(self, tmp_path):
        deps = dep_files_for_env("alfworld", tmp_path)
        names = [p.name for p in deps]
        assert "alfworld_env.py" in names
        for shared in SHARED_HELPER_FILES:
            assert shared in names

    def test_unknown_env_only_shared(self, tmp_path):
        deps = dep_files_for_env("mystery", tmp_path)
        names = [p.name for p in deps]
        assert names == sorted(SHARED_HELPER_FILES)

    def test_case_insensitive_env(self, tmp_path):
        deps = dep_files_for_env("ALFWorld", tmp_path)
        assert any(p.name == "alfworld_env.py" for p in deps)


# ---------------------------------------------------------------------------
# compute_fingerprint — determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_hash(self, tmp_path):
        _seed_helpers(tmp_path)
        cell = _cell("alfworld")
        h1 = compute_fingerprint(cell, tmp_path, env={})
        h2 = compute_fingerprint(cell, tmp_path, env={})
        assert h1 == h2

    def test_is_64_char_hex(self, tmp_path):
        _seed_helpers(tmp_path)
        h = compute_fingerprint(_cell("search_qa"), tmp_path, env={})
        assert len(h) == 64
        int(h, 16)  # parses as hex

    def test_cell_key_order_irrelevant(self, tmp_path):
        _seed_helpers(tmp_path)
        base = _cell("alfworld")
        reordered = dict(reversed(list(base.items())))
        assert compute_fingerprint(base, tmp_path, env={}) == compute_fingerprint(
            reordered, tmp_path, env={}
        )

    def test_volatile_keys_do_not_affect_hash(self, tmp_path):
        """id / est_vram_gb are placement-only — must not perturb the fingerprint."""
        _seed_helpers(tmp_path)
        a = _cell("alfworld", id="different-id", est_vram_gb=99.0)
        b = _cell("alfworld")
        assert compute_fingerprint(a, tmp_path, env={}) == compute_fingerprint(
            b, tmp_path, env={}
        )


# ---------------------------------------------------------------------------
# compute_fingerprint — the env-vs-shared SPLIT guarantee
# ---------------------------------------------------------------------------

class TestSplitGuarantee:
    def test_alfworld_helper_edit_flips_alfworld_only(self, tmp_path):
        _seed_helpers(tmp_path)
        alf = _cell("alfworld")
        sqa = _cell("search_qa")
        alf_before = compute_fingerprint(alf, tmp_path, env={})
        sqa_before = compute_fingerprint(sqa, tmp_path, env={})

        # Edit ONLY the alfworld helper bytes.
        (tmp_path / "alfworld_env.py").write_text("# alfworld_env.py v2\n", encoding="utf-8")

        alf_after = compute_fingerprint(alf, tmp_path, env={})
        sqa_after = compute_fingerprint(sqa, tmp_path, env={})

        assert alf_after != alf_before, "alfworld fingerprint must change"
        assert sqa_after == sqa_before, "search_qa fingerprint must be unaffected"

    def test_shared_helper_edit_flips_every_env(self, tmp_path):
        _seed_helpers(tmp_path)
        alf_before = compute_fingerprint(_cell("alfworld"), tmp_path, env={})
        sqa_before = compute_fingerprint(_cell("search_qa"), tmp_path, env={})
        web_before = compute_fingerprint(_cell("webshop"), tmp_path, env={})

        # Edit a SHARED helper — affects all cells.
        (tmp_path / "agentic_rollout.py").write_text("# agentic_rollout.py v2\n", encoding="utf-8")

        assert compute_fingerprint(_cell("alfworld"), tmp_path, env={}) != alf_before
        assert compute_fingerprint(_cell("search_qa"), tmp_path, env={}) != sqa_before
        assert compute_fingerprint(_cell("webshop"), tmp_path, env={}) != web_before


# ---------------------------------------------------------------------------
# compute_fingerprint — param + flag sensitivity
# ---------------------------------------------------------------------------

class TestSensitivity:
    def test_seed_change_flips(self, tmp_path):
        _seed_helpers(tmp_path)
        before = compute_fingerprint(_cell("alfworld", seed=42), tmp_path, env={})
        after = compute_fingerprint(_cell("alfworld", seed=7), tmp_path, env={})
        assert before != after

    def test_steps_change_flips(self, tmp_path):
        _seed_helpers(tmp_path)
        before = compute_fingerprint(_cell("alfworld", steps=150), tmp_path, env={})
        after = compute_fingerprint(_cell("alfworld", steps=300), tmp_path, env={})
        assert before != after

    def test_model_id_change_flips(self, tmp_path):
        _seed_helpers(tmp_path)
        before = compute_fingerprint(_cell("alfworld"), tmp_path, env={})
        after = compute_fingerprint(
            _cell("alfworld", model_id="Qwen/Qwen2.5-3B"), tmp_path, env={}
        )
        assert before != after

    def test_baseline_change_flips(self, tmp_path):
        _seed_helpers(tmp_path)
        before = compute_fingerprint(_cell("alfworld", baseline="sdar"), tmp_path, env={})
        after = compute_fingerprint(_cell("alfworld", baseline="grpo"), tmp_path, env={})
        assert before != after

    def test_flag_change_flips(self, tmp_path):
        _seed_helpers(tmp_path)
        cell = _cell("alfworld")
        flag = FLAG_ALLOWLIST[0]
        before = compute_fingerprint(cell, tmp_path, env={flag: "0"})
        after = compute_fingerprint(cell, tmp_path, env={flag: "1"})
        assert before != after

    def test_trainer_version_flag_flips(self, tmp_path):
        """REPROLAB_TRAINER_VERSION stands in for the un-hashed train_cell.py."""
        _seed_helpers(tmp_path)
        cell = _cell("alfworld")
        before = compute_fingerprint(cell, tmp_path, env={"REPROLAB_TRAINER_VERSION": "1"})
        after = compute_fingerprint(cell, tmp_path, env={"REPROLAB_TRAINER_VERSION": "2"})
        assert before != after

    def test_non_allowlisted_env_var_ignored(self, tmp_path):
        _seed_helpers(tmp_path)
        cell = _cell("alfworld")
        before = compute_fingerprint(cell, tmp_path, env={})
        after = compute_fingerprint(cell, tmp_path, env={"SOME_RANDOM_VAR": "xyz"})
        assert before == after

    def test_env_none_reads_os_environ(self, tmp_path, monkeypatch):
        _seed_helpers(tmp_path)
        cell = _cell("alfworld")
        flag = FLAG_ALLOWLIST[0]
        monkeypatch.delenv(flag, raising=False)
        baseline = compute_fingerprint(cell, tmp_path, env={})
        absent = compute_fingerprint(cell, tmp_path)  # env=None → os.environ
        assert baseline == absent  # both see "" for the unset flag
        monkeypatch.setenv(flag, "shaped")
        present = compute_fingerprint(cell, tmp_path)
        assert present != baseline


# ---------------------------------------------------------------------------
# compute_fingerprint — env-scoped flags (the resume "re-run only X" guarantee)
# ---------------------------------------------------------------------------

class TestFlagEnvScoping:
    """Env-scoped flags invalidate ONLY their env's cells (the resume value prop)."""

    def test_alfworld_flag_flips_alfworld_not_search_qa(self, tmp_path):
        _seed_helpers(tmp_path)
        off = {"REPROLAB_ALFWORLD_SHAPED_REWARD": "0"}
        on = {"REPROLAB_ALFWORLD_SHAPED_REWARD": "1"}
        assert (
            compute_fingerprint(_cell("alfworld"), tmp_path, env=off)
            != compute_fingerprint(_cell("alfworld"), tmp_path, env=on)
        )  # alfworld cells re-run
        assert (
            compute_fingerprint(_cell("search_qa"), tmp_path, env=off)
            == compute_fingerprint(_cell("search_qa"), tmp_path, env=on)
        )  # search_qa cells are NOT disturbed by an alfworld flag

    def test_search_qa_flag_flips_search_qa_not_alfworld(self, tmp_path):
        _seed_helpers(tmp_path)
        off = {"REPROLAB_SEARCH_QA_DENSE": "0"}
        on = {"REPROLAB_SEARCH_QA_DENSE": "1"}
        assert (
            compute_fingerprint(_cell("search_qa"), tmp_path, env=off)
            != compute_fingerprint(_cell("search_qa"), tmp_path, env=on)
        )
        assert (
            compute_fingerprint(_cell("alfworld"), tmp_path, env=off)
            == compute_fingerprint(_cell("alfworld"), tmp_path, env=on)
        )

    def test_global_trainer_version_flips_every_env(self, tmp_path):
        _seed_helpers(tmp_path)
        v1 = {"REPROLAB_TRAINER_VERSION": "1"}
        v2 = {"REPROLAB_TRAINER_VERSION": "2"}
        for env_name in ("alfworld", "search_qa", "webshop"):
            assert (
                compute_fingerprint(_cell(env_name), tmp_path, env=v1)
                != compute_fingerprint(_cell(env_name), tmp_path, env=v2)
            ), f"global flag must flip {env_name}"


# ---------------------------------------------------------------------------
# compute_fingerprint — fail-soft on missing helpers
# ---------------------------------------------------------------------------

class TestFailSoft:
    def test_missing_helper_does_not_raise(self, tmp_path):
        # Empty code dir — no helper files at all.
        h = compute_fingerprint(_cell("alfworld"), tmp_path, env={})
        assert len(h) == 64  # produced a hash, did not raise

    def test_present_then_absent_flips(self, tmp_path):
        _seed_helpers(tmp_path)
        cell = _cell("alfworld")
        with_file = compute_fingerprint(cell, tmp_path, env={})
        (tmp_path / "alfworld_env.py").unlink()
        without_file = compute_fingerprint(cell, tmp_path, env={})
        assert with_file != without_file

    def test_two_missing_envs_differ_by_params_not_crash(self, tmp_path):
        # No helpers present; two different envs still hash distinctly because
        # the env param differs (and never raise).
        alf = compute_fingerprint(_cell("alfworld"), tmp_path, env={})
        sqa = compute_fingerprint(_cell("search_qa"), tmp_path, env={})
        assert alf != sqa

    def test_unknown_env_is_failsoft(self, tmp_path):
        _seed_helpers(tmp_path)
        h = compute_fingerprint(_cell("totally_unknown_env"), tmp_path, env={})
        assert len(h) == 64

    def test_non_dict_cell_is_failsoft(self, tmp_path):
        _seed_helpers(tmp_path)
        # Defensive: a malformed cell must degrade, not crash.
        h = compute_fingerprint(None, tmp_path, env={})  # type: ignore[arg-type]
        assert len(h) == 64
