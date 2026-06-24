"""Pin the compute-constraint guidance in implement_baseline — and verify the
prompt is identical under both API-key and OAuth auth surfaces (parity).

The 2026-05-23 mandate: 'support API only mode as well as OAuth, elegantly
dynamically support both modes'. The compute-constraint prompt addition is a
pure text change at the prompt-build layer — provider-independent by
construction.

ComputeConstraintGuidance: guidance fires based on BOTH sandbox_mode AND
gpu_mode — not a static sandbox-name heuristic — so docker+gpu_mode=max
correctly skips the constraint, and runpod always skips it.

AuthSurfaceParity: the prompt text is identical for `claude` (API) and
`claude-oauth` (subscription) — proving zero auth-surface fork.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.agents.baseline_implementation import run_with_sdk
from backend.agents.schemas import (
    EnvironmentSpec,
    PaperClaimMap,
)


def _capture_prompt(monkeypatch):
    """Patch collect_agent_text to capture the prompt argument; return the
    captured-list so the test can inspect what would have been sent."""
    captured: list[dict] = []

    async def _fake_collect(agent_name, prompt, **kwargs):
        captured.append({
            "agent": agent_name, "prompt": prompt,
            "model": kwargs.get("model"), "provider": kwargs.get("provider"),
        })
        return ""

    # collect_agent_text is lazy-imported inside run_with_sdk, so patch at its
    # source module (backend.agents.runtime.invoke), not at the consumer.
    monkeypatch.setattr(
        "backend.agents.runtime.invoke.collect_agent_text",
        _fake_collect,
    )
    return captured


def _minimal_inputs(tmp_path: Path):
    """Build the minimum object set run_with_sdk needs."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "prj_test").mkdir(parents=True, exist_ok=True)
    (runs_root / "prj_test" / "code").mkdir(parents=True, exist_ok=True)
    pcm = PaperClaimMap(core_contribution="test paper")
    env = EnvironmentSpec(dockerfile="FROM python:3.11", framework="pytorch")
    contract = None
    return runs_root, pcm, env, contract


class TestDynamicComputeGuidance:
    """Runtime detection block is ALWAYS-ON; policy overlay is gpu_mode-driven.

    Per the 2026-05-23 night refactor: the agent writes ONE script that
    detects torch.cuda.is_available() at runtime and adapts. Hard-coding
    either mode at build time is wrong — same artifact runs on CPU docker
    AND GPU runpod.

    Always-on (every call): RUNTIME COMPUTE DETECTION block.
    Policy overlays:
    - gpu_mode='off' → adds CPU-only overlay (entrypoint targets CPU path)
    - gpu_mode='max' → adds GPU overlay (entrypoint targets GPU path)
    - gpu_mode in {auto, prefer, None}: no overlay (runtime detection wins)
    """

    @pytest.mark.parametrize("sandbox,gpu_mode", [
        ("docker",        None),
        ("docker",        "auto"),
        ("docker",        "prefer"),
        ("local",         None),
        ("runpod",        None),
        ("runpod",        "auto"),
        (None,            None),
        ("unknown_value", None),
    ])
    def test_runtime_detection_block_is_always_present(
        self, sandbox, gpu_mode, tmp_path, monkeypatch
    ):
        """The runtime-detection block fires for EVERY sandbox+gpu_mode combo
        (except the policy-overlay cases tested separately). This is the
        invariant 'we should always support gpu or cpu dynamically'."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode=sandbox,
            gpu_mode=gpu_mode,
        ))
        prompt = captured[0]["prompt"]
        assert "RUNTIME COMPUTE DETECTION" in prompt
        assert "torch.cuda.is_available" in prompt
        assert "HAS_GPU" in prompt
        # Adaptive guidance must mention BOTH scale-down (CPU) and scale-up (GPU)
        assert "Scale-down on CPU" in prompt
        assert "Scale-up on GPU" in prompt

    def test_engineering_standards_block_is_always_present(self, tmp_path, monkeypatch):
        """The elite-ML-engineer standards block is always-on (every sandbox/gpu_mode):
        faithful-algorithm + measured-metrics rails, plus the self-verify-real-compute
        habit (the structural anti-stub rail that the 2026-06-19 gpt-chat-latest run
        would have tripped: 0-GPU 'success' emitting total_length/chunk_count)."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker", gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        assert "ENGINEERING STANDARDS" in prompt
        assert "FAITHFUL ALGORITHM" in prompt
        assert "MEASURED METRICS ONLY" in prompt
        # the anti-stub self-verification rail (the elite habit)
        assert "SELF-VERIFY BEFORE SCALING" in prompt
        assert "max_memory_allocated" in prompt

    def test_gpu_mode_off_adds_cpu_only_policy_overlay(self, tmp_path, monkeypatch):
        """--gpu-mode=off → user explicitly forbids GPU; overlay says 'commands.json
        targets CPU path'. The runtime detection block is STILL present (the GPU
        branch in the code is dead-but-present for portability)."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",
            gpu_mode="off",
        ))
        prompt = captured[0]["prompt"]
        assert "RUNTIME COMPUTE DETECTION" in prompt  # always-on
        assert "POLICY OVERLAY — --gpu-mode=off" in prompt
        assert "CPU/smoke path" in prompt
        assert "POLICY OVERLAY — --gpu-mode=max" not in prompt

    def test_gpu_mode_max_adds_gpu_target_policy_overlay(self, tmp_path, monkeypatch):
        """--gpu-mode=max → user explicitly demands GPU; overlay says 'commands.json
        targets GPU path'. CPU branch stays in code as safety net."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",
            gpu_mode="max",
        ))
        prompt = captured[0]["prompt"]
        assert "RUNTIME COMPUTE DETECTION" in prompt  # always-on
        assert "POLICY OVERLAY — --gpu-mode=max" in prompt
        assert "full-scale (GPU) path" in prompt
        assert "POLICY OVERLAY — --gpu-mode=off" not in prompt

    def test_default_no_policy_overlay(self, tmp_path, monkeypatch):
        """gpu_mode=auto / None / prefer → no policy overlay; runtime detection
        decides at execution time. This is the most common case."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",
            gpu_mode="auto",
        ))
        prompt = captured[0]["prompt"]
        assert "RUNTIME COMPUTE DETECTION" in prompt
        assert "POLICY OVERLAY" not in prompt


class TestAuthSurfaceParity:
    """Verify the sandbox-aware prompt is IDENTICAL under both API-key and
    OAuth auth surfaces — proves zero auth fork in the new code path."""

    def _render_prompt_with_provider(self, tmp_path, monkeypatch, provider: str | None, model: str | None):
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",  # use the CPU path so guidance fires
            provider=provider,
            model=model,
        ))
        return captured[0]["prompt"]

    def test_api_and_oauth_produce_identical_prompts(self, tmp_path, monkeypatch):
        # API-key path: provider="anthropic", model="claude-sonnet-4-6"
        api_prompt = self._render_prompt_with_provider(
            tmp_path, monkeypatch, provider="anthropic", model="claude-sonnet-4-6"
        )
        # OAuth path: provider="anthropic", model="claude-oauth" (the OAuth alias)
        oauth_prompt = self._render_prompt_with_provider(
            tmp_path, monkeypatch, provider="anthropic", model="claude-oauth"
        )
        # Identical text — the prompt is provider-agnostic. The only auth-
        # specific code is BELOW this layer (in collect_agent_text → SDK).
        assert api_prompt == oauth_prompt, (
            "implement_baseline prompt diverged between API and OAuth modes — "
            "the prompt-build layer must be auth-agnostic."
        )

    def test_openai_provider_also_identical_prompt(self, tmp_path, monkeypatch):
        # If a future deployment uses OpenAI for the baseline agent, the
        # prompt MUST still carry the runtime-detection guidance.
        prompt = self._render_prompt_with_provider(
            tmp_path, monkeypatch, provider="openai", model="gpt-5"
        )
        assert "RUNTIME COMPUTE DETECTION" in prompt
        assert "torch.cuda.is_available" in prompt


class TestGpuModePlumbedThroughRunContext:
    """Pin the 2026-05-23 evening fix: RunContext.gpu_mode is threaded from
    ExecutionProfile so _compute_constraint_guidance gets the right signal.
    Without this, ctx.gpu_mode is always None and the dynamic decision
    collapses to "sandbox_mode alone" — same bug as before the helper."""

    def test_runcontext_has_gpu_mode_field(self):
        """RunContext dataclass MUST expose a gpu_mode attribute (default None
        for back-compat). Removing this field reverts dynamic detection to
        "sandbox name alone" — same bug as before."""
        from backend.agents.rlm.context import RunContext
        import dataclasses
        names = {f.name for f in dataclasses.fields(RunContext)}
        assert "gpu_mode" in names, (
            "RunContext.gpu_mode field is required for sandbox-aware baseline "
            "guidance. Without it, ctx.gpu_mode is always None and runpod runs "
            "incorrectly trigger CPU smoke-test guidance."
        )

    def test_gpu_mode_default_is_None_for_backward_compat(self):
        """Existing call sites that don't pass gpu_mode must still work."""
        from backend.agents.rlm.context import RunContext
        from pathlib import Path
        ctx = RunContext(
            project_id="prj_test",
            project_dir=Path("/tmp/test"),
            runs_root=Path("/tmp"),
            dashboard=None,
            cost_ledger=None,
            llm_client=None,
            provider="anthropic",
            model="claude-sonnet-4-6",
        )
        assert ctx.gpu_mode is None, (
            "gpu_mode must default to None for back-compat with call sites "
            "that don't pass it (e.g. existing tests + scripts)."
        )


# ---------------------------------------------------------------------------
# New helpers: rubric checklist, dataset setup, paper override
# ---------------------------------------------------------------------------

class TestRubricChecklistBlock:
    """_rubric_checklist_block reads generated_rubric.json and formats top-20
    leaves sorted by weight descending."""

    def test_missing_file_returns_empty_string(self, tmp_path):
        """No generated_rubric.json → empty string, no crash."""
        from backend.agents.baseline_implementation import _rubric_checklist_block
        result = _rubric_checklist_block(tmp_path)
        assert result == ""

    def test_flat_rubric_leaves_appear_in_output(self, tmp_path):
        """Leaf nodes (no sub_tasks) are listed with weight and requirements."""
        from backend.agents.baseline_implementation import _rubric_checklist_block
        rubric = {
            "sub_tasks": [
                {"requirements": "Implement sigmoid gate g_t = sigmoid(beta * Delta_t)", "weight": 0.5},
                {"requirements": "Use stop_gradient on the gate", "weight": 0.3},
                {"requirements": "Set lambda=0.1 in the combined loss", "weight": 0.2},
            ]
        }
        (tmp_path / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        result = _rubric_checklist_block(tmp_path)
        assert "RUBRIC CHECKLIST" in result
        # Leaves appear sorted by weight descending
        assert "[w=0.50]" in result
        assert "sigmoid gate" in result
        assert "[w=0.30]" in result
        assert "[w=0.20]" in result

    def test_nested_sub_tasks_walks_to_leaves(self, tmp_path):
        """Nested sub_tasks are recursed; only leaf nodes (no children) appear."""
        from backend.agents.baseline_implementation import _rubric_checklist_block
        rubric = {
            "sub_tasks": [
                {
                    "requirements": "Algorithm fidelity",
                    "weight": 1.0,
                    "sub_tasks": [
                        {"requirements": "Gate formula correct", "weight": 0.6},
                        {"requirements": "Loss components correct", "weight": 0.4},
                    ],
                }
            ]
        }
        (tmp_path / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        result = _rubric_checklist_block(tmp_path)
        assert "RUBRIC CHECKLIST" in result
        assert "Gate formula correct" in result
        assert "Loss components correct" in result
        # Non-leaf "Algorithm fidelity" should NOT appear as its own entry
        # (it has children, so it's not a leaf)
        assert "[w=1.00]" not in result

    def test_top_20_cap(self, tmp_path):
        """Only the top 20 leaves by weight are included."""
        from backend.agents.baseline_implementation import _rubric_checklist_block
        leaves = [
            {"requirements": f"Leaf {i}", "weight": float(i) / 100}
            for i in range(30)
        ]
        rubric = {"sub_tasks": leaves}
        (tmp_path / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        result = _rubric_checklist_block(tmp_path)
        # Should have exactly 20 entries (header line + 20 leaf lines)
        leaf_lines = [ln for ln in result.splitlines() if ln.strip().startswith("[w=")]
        assert len(leaf_lines) == 20

    def test_requirements_truncated_to_250_chars(self, tmp_path):
        """Long requirements text is truncated at 250 chars with ellipsis."""
        from backend.agents.baseline_implementation import _rubric_checklist_block
        long_req = "x" * 300
        rubric = {"sub_tasks": [{"requirements": long_req, "weight": 0.5}]}
        (tmp_path / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        result = _rubric_checklist_block(tmp_path)
        # The rendered requirement must be ≤ 253 chars (250 + "...")
        leaf_line = next(ln for ln in result.splitlines() if "[w=" in ln)
        req_part = leaf_line.split("] ", 1)[1]
        assert len(req_part) <= 253
        assert req_part.endswith("...")

    def test_weight_sorted_descending(self, tmp_path):
        """Highest-weight leaf appears before lower-weight leaf in output."""
        from backend.agents.baseline_implementation import _rubric_checklist_block
        rubric = {
            "sub_tasks": [
                {"requirements": "Low", "weight": 0.1},
                {"requirements": "High", "weight": 0.9},
                {"requirements": "Mid", "weight": 0.5},
            ]
        }
        (tmp_path / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        result = _rubric_checklist_block(tmp_path)
        pos_high = result.index("High")
        pos_mid = result.index("Mid")
        pos_low = result.index("Low")
        assert pos_high < pos_mid < pos_low


class TestLoadPaperOverride:
    """_load_paper_override loads docs/papers/<arxiv_id>.yaml from repo root."""

    def test_missing_arxiv_id_returns_empty(self):
        from backend.agents.baseline_implementation import _load_paper_override
        assert _load_paper_override(None) == ""
        assert _load_paper_override("") == ""

    def test_missing_yaml_file_returns_empty(self):
        from backend.agents.baseline_implementation import _load_paper_override
        # An arxiv_id for which no yaml file exists
        result = _load_paper_override("9999.99999")
        assert result == ""

    def test_sdar_yaml_returns_non_empty_string(self):
        """The real 2605.15155.yaml must exist and produce a non-empty block."""
        from backend.agents.baseline_implementation import _load_paper_override
        result = _load_paper_override("2605.15155")
        assert result != "", "docs/papers/2605.15155.yaml must exist and be non-empty"
        assert "PAPER-SPECIFIC GUIDANCE" in result
        assert "2605.15155" in result

    def test_sdar_yaml_surfaces_algorithm_invariants(self):
        """Key SDAR algorithm fields must appear in the override block."""
        from backend.agents.baseline_implementation import _load_paper_override
        result = _load_paper_override("2605.15155")
        assert "gate_formula" in result
        assert "stop_gradient_on_gate" in result
        assert "lambda" in result

    def test_override_with_temp_yaml(self, tmp_path, monkeypatch):
        """Override loader works with an arbitrary yaml placed at the expected path."""
        from backend.agents import baseline_implementation as bi
        # Redirect _REPO_ROOT to tmp_path so we can place a yaml there
        monkeypatch.setattr(bi, "_REPO_ROOT", tmp_path)
        yaml_dir = tmp_path / "docs" / "papers"
        yaml_dir.mkdir(parents=True)
        (yaml_dir / "1234.56789.yaml").write_text(
            "key: value\nnested:\n  a: 1\n", encoding="utf-8"
        )
        result = bi._load_paper_override("1234.56789")
        assert "PAPER-SPECIFIC GUIDANCE" in result
        assert "key" in result
        assert "value" in result


class TestDatasetSetupBlock:
    """_DATASET_SETUP_BLOCK is always included in the guidance."""

    def test_dataset_setup_always_present(self, tmp_path, monkeypatch):
        """DATASET SETUP block fires for every sandbox+gpu_mode combo."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",
            gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        assert "DATASET SETUP" in prompt
        assert "alfworld-download" in prompt
        assert "load_dataset" in prompt

    def test_dataset_setup_present_for_runpod(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", raising=False)
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="runpod",
            gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        assert "DATASET SETUP" in prompt
        assert "/workspace/data" in prompt

    def test_dataset_setup_uses_writable_root_for_local(self, tmp_path, monkeypatch):
        """Local sandbox: run.py points the volume-mount root at a writable dir; the
        DATASET-SETUP guidance must use THAT root and never leak the RunPod-only
        /workspace path (the 2026-05-29 SDAR env_load_failed root cause)."""
        data_root = str(tmp_path / "data_cache")
        monkeypatch.setenv("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", data_root)
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="local",
            gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        assert "DATASET SETUP" in prompt
        assert f"{data_root}/data/alfworld" in prompt
        assert "/workspace/data" not in prompt


class TestPromptAssemblyOrder:
    """Verify the canonical prompt-assembly order is respected."""

    def test_assembly_order_no_rubric_no_override(self, tmp_path, monkeypatch):
        """Without rubric or override: NO_STUB < RUNTIME_DETECTION < DATASET_SETUP."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",
            gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        pos_stub = prompt.index("NO STUB")
        pos_runtime = prompt.index("RUNTIME COMPUTE DETECTION")
        pos_dataset = prompt.index("DATASET SETUP")
        assert pos_stub < pos_runtime < pos_dataset

    def test_rubric_checklist_appears_after_dataset_setup(self, tmp_path, monkeypatch):
        """When generated_rubric.json exists, RUBRIC CHECKLIST comes after DATASET SETUP."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        # Write a minimal rubric into the project dir
        project_dir = runs_root / "prj_test"
        rubric = {"sub_tasks": [{"requirements": "Test leaf", "weight": 0.5}]}
        (project_dir / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",
            gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        pos_dataset = prompt.index("DATASET SETUP")
        pos_checklist = prompt.index("RUBRIC CHECKLIST")
        assert pos_dataset < pos_checklist

    def test_paper_override_appears_after_rubric_checklist(self, tmp_path, monkeypatch):
        """PAPER-SPECIFIC GUIDANCE comes after RUBRIC CHECKLIST."""
        from backend.agents import baseline_implementation as bi
        monkeypatch.setattr(bi, "_REPO_ROOT", tmp_path)
        yaml_dir = tmp_path / "docs" / "papers"
        yaml_dir.mkdir(parents=True)
        (yaml_dir / "2605.15155.yaml").write_text(
            "algorithm_invariants:\n  gate_formula: test\n", encoding="utf-8"
        )
        # Also write a rubric so we can verify order relative to checklist
        captured = _capture_prompt(monkeypatch)
        runs_root = tmp_path / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        project_dir = runs_root / "2605.15155"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "code").mkdir(parents=True, exist_ok=True)
        rubric = {"sub_tasks": [{"requirements": "Gate leaf", "weight": 0.5}]}
        (project_dir / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        pcm = PaperClaimMap(core_contribution="SDAR")
        env = EnvironmentSpec(dockerfile="FROM python:3.11", framework="pytorch")
        asyncio.run(run_with_sdk(
            "2605.15155", runs_root, pcm, env, None,
            sandbox_mode="docker",
            gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        pos_checklist = prompt.index("RUBRIC CHECKLIST")
        pos_override = prompt.index("PAPER-SPECIFIC GUIDANCE")
        assert pos_checklist < pos_override

    def test_both_rubric_and_override_present_in_prompt(self, tmp_path, monkeypatch):
        """When both inputs exist, both RUBRIC CHECKLIST and PAPER-SPECIFIC GUIDANCE appear."""
        from backend.agents import baseline_implementation as bi
        monkeypatch.setattr(bi, "_REPO_ROOT", tmp_path)
        yaml_dir = tmp_path / "docs" / "papers"
        yaml_dir.mkdir(parents=True)
        (yaml_dir / "2605.15155.yaml").write_text(
            "algorithm_invariants:\n  beta: 10\n", encoding="utf-8"
        )
        captured = _capture_prompt(monkeypatch)
        runs_root = tmp_path / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        project_dir = runs_root / "2605.15155"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "code").mkdir(parents=True, exist_ok=True)
        rubric = {"sub_tasks": [{"requirements": "Leaf A", "weight": 0.8}]}
        (project_dir / "generated_rubric.json").write_text(
            json.dumps(rubric), encoding="utf-8"
        )
        pcm = PaperClaimMap(core_contribution="SDAR")
        env = EnvironmentSpec(dockerfile="FROM python:3.11", framework="pytorch")
        asyncio.run(run_with_sdk(
            "2605.15155", runs_root, pcm, env, None,
            sandbox_mode="docker",
            gpu_mode=None,
        ))
        prompt = captured[0]["prompt"]
        assert "RUBRIC CHECKLIST" in prompt
        assert "PAPER-SPECIFIC GUIDANCE" in prompt


class TestExtractArxivId:
    """_extract_arxiv_id parses bare arXiv IDs from project_id strings."""

    def test_bare_arxiv_id(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("2605.15155") == "2605.15155"

    def test_prefixed_arxiv_id(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("arXiv_2605.15155_abc123") == "2605.15155"

    def test_no_arxiv_id_returns_none(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("prj_test") is None
        assert _extract_arxiv_id("") is None

    def test_five_digit_suffix(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("1706.03762") == "1706.03762"
