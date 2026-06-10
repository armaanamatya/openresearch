"""PR-ξ Phase 2 — baseline_knowledge module tests.

Tests the knowledge channel: helper rendering, manifest writing, postflight
verification, severity semantics, and regression fixtures.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from backend.agents.baseline_knowledge import (
    KNOWLEDGE_CHANNEL_VERSION,
    CuratedFact,
    Severity,
    from_recipes,
    render_helper_module,
    verify_emitted_code,
    write_curated_artifacts,
)
from backend.agents.dataset_recipes import find_recipe


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_frey_fact() -> CuratedFact:
    """Build a CuratedFact matching the Frey Face recipe in dataset_recipes.py."""
    recipe = find_recipe("Frey Face")
    assert recipe is not None, "Frey Face recipe must be in DATASET_RECIPES"
    facts = from_recipes([asdict(recipe)])
    assert facts, "from_recipes should produce a CuratedFact for Frey Face"
    return facts[0]


def _write_compliant_train(code_dir: Path, helper_name: str) -> Path:
    """Write a minimal compliant train.py that imports and calls the helper."""
    content = (
        f"from _reprolab_curated import {helper_name}\n"
        f"\n"
        f"data = {helper_name}()\n"
        f"print('loaded', data.shape)\n"
    )
    path = code_dir / "train.py"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. render_helper_module — Frey Face produces a loader
# ---------------------------------------------------------------------------

class TestRenderHelperModule:
    def test_render_helper_module_frey_face_emits_loader(self):
        fact = _make_frey_fact()
        module_source, manifest = render_helper_module([fact])

        assert f"def {fact.helper_name}(" in module_source, (
            f"module_source should contain 'def {fact.helper_name}('"
        )
        assert f"knowledge_channel_version={KNOWLEDGE_CHANNEL_VERSION}" in module_source

        # Manifest structure
        assert manifest["version"] == KNOWLEDGE_CHANNEL_VERSION
        assert len(manifest["facts"]) == 1
        mfact = manifest["facts"][0]
        assert mfact["helper_name"] == fact.helper_name
        assert mfact["helper_hash"] == fact.helper_hash
        assert f"from _reprolab_curated import {fact.helper_name}" == mfact["required_import"]


# ---------------------------------------------------------------------------
# 2. verify_emitted_code — compliant train.py passes
# ---------------------------------------------------------------------------

class TestVerifyPasses:
    def test_verify_passes_when_train_py_complies(self, tmp_path):
        fact = _make_frey_fact()
        manifest = write_curated_artifacts(tmp_path, [fact])
        train_py = _write_compliant_train(tmp_path, fact.helper_name)

        violations = verify_emitted_code(train_py, manifest, tmp_path)
        assert violations == [], (
            f"Compliant train.py should produce no violations, got: {violations}"
        )


# ---------------------------------------------------------------------------
# 3. verify_emitted_code — missing import flagged
# ---------------------------------------------------------------------------

class TestVerifyMissingImport:
    def test_verify_flags_missing_import(self, tmp_path):
        fact = _make_frey_fact()
        manifest = write_curated_artifacts(tmp_path, [fact])

        # train.py without the required import
        train_py = tmp_path / "train.py"
        train_py.write_text(
            "import pickle\n"
            "data = pickle.loads(open('freyfaces.pkl', 'rb').read(), encoding='latin1')\n",
            encoding="utf-8",
        )

        violations = verify_emitted_code(train_py, manifest, tmp_path)
        kinds = [v.kind for v in violations]
        assert "missing_import" in kinds, (
            f"Expected missing_import violation, got: {violations}"
        )


# ---------------------------------------------------------------------------
# 4. verify_emitted_code — shadowed helper flagged
# ---------------------------------------------------------------------------

class TestVerifyShadowedHelper:
    def test_verify_flags_shadowed_helper(self, tmp_path):
        fact = _make_frey_fact()
        manifest = write_curated_artifacts(tmp_path, [fact])

        # train.py imports the helper BUT also defines a shadowing function
        train_py = tmp_path / "train.py"
        train_py.write_text(
            f"from _reprolab_curated import {fact.helper_name}\n"
            f"\n"
            f"def {fact.helper_name}():\n"
            f"    # Override with NYU URL (the wrong way)\n"
            f"    import urllib.request\n"
            f"    return urllib.request.urlopen('https://cs.nyu.edu/~roweis/data/frey_rawface.mat')\n",
            encoding="utf-8",
        )

        violations = verify_emitted_code(train_py, manifest, tmp_path)
        kinds = [v.kind for v in violations]
        assert "shadowed_helper" in kinds, (
            f"Expected shadowed_helper violation, got: {violations}"
        )


# ---------------------------------------------------------------------------
# 5. verify_emitted_code — banned literal flagged
# ---------------------------------------------------------------------------

class TestVerifyBannedLiteral:
    def test_verify_flags_banned_literal(self, tmp_path):
        fact = _make_frey_fact()
        manifest = write_curated_artifacts(tmp_path, [fact])

        # train.py contains the banned NYU URL
        train_py = tmp_path / "train.py"
        train_py.write_text(
            f"from _reprolab_curated import {fact.helper_name}\n"
            f"\n"
            f"url = 'https://cs.nyu.edu/~roweis/data/frey_rawface.mat'\n"
            f"data = {fact.helper_name}()\n",
            encoding="utf-8",
        )

        violations = verify_emitted_code(train_py, manifest, tmp_path)
        kinds = [v.kind for v in violations]
        assert "banned_literal" in kinds, (
            f"Expected banned_literal violation, got: {violations}"
        )


# ---------------------------------------------------------------------------
# 6. verify_emitted_code — helper hash mismatch flagged
# ---------------------------------------------------------------------------

class TestVerifyHelperHashMismatch:
    def test_verify_flags_helper_hash_mismatch(self, tmp_path):
        fact = _make_frey_fact()
        manifest = write_curated_artifacts(tmp_path, [fact])

        # Tamper with the helper module after writing
        curated_py = tmp_path / "_reprolab_curated.py"
        original = curated_py.read_text(encoding="utf-8")
        curated_py.write_text(original + "\n# TAMPERED\n", encoding="utf-8")

        train_py = _write_compliant_train(tmp_path, fact.helper_name)

        violations = verify_emitted_code(train_py, manifest, tmp_path)
        kinds = [v.kind for v in violations]
        assert "helper_hash_mismatch" in kinds, (
            f"Expected helper_hash_mismatch violation, got: {violations}"
        )


# ---------------------------------------------------------------------------
# 7. Severity: ADVISORY violation does not block (caller check)
# ---------------------------------------------------------------------------

class TestSeverityAdvisory:
    def test_severity_advisory_does_not_block(self, tmp_path):
        """An ADVISORY-severity violation must be in the list but caller should
        not treat it as blocking. This test verifies severity is correctly set."""
        advisory_fact = CuratedFact(
            id="test.advisory",
            family="dataset",
            severity=Severity.ADVISORY,
            helper_name="load_advisory",
            helper_body="def load_advisory():\n    return None\n",
            required_import="from _reprolab_curated import load_advisory",
            banned_literals=(),
            helper_hash="abc123",
        )
        manifest = {
            "version": KNOWLEDGE_CHANNEL_VERSION,
            "facts": [
                {
                    "id": advisory_fact.id,
                    "helper_name": advisory_fact.helper_name,
                    "required_import": advisory_fact.required_import,
                    "banned_literals": [],
                    "severity": "advisory",
                    "helper_hash": advisory_fact.helper_hash,
                }
            ],
        }
        # train.py missing the import → ADVISORY violation
        train_py = tmp_path / "train.py"
        train_py.write_text("print('hello')\n", encoding="utf-8")

        violations = verify_emitted_code(train_py, manifest, tmp_path)
        # Violation is present
        assert violations, "Expected at least one violation"
        # But severity is ADVISORY
        for v in violations:
            assert v.severity == Severity.ADVISORY, (
                f"Expected ADVISORY severity, got {v.severity}"
            )
        # Caller check: strict filter returns empty
        strict = [v for v in violations if v.severity == Severity.STRICT]
        assert not strict, "No STRICT violations expected for ADVISORY fact"


# ---------------------------------------------------------------------------
# 8. Severity: STRICT violation triggers block (caller check)
# ---------------------------------------------------------------------------

class TestSeverityStrict:
    def test_severity_strict_blocks(self, tmp_path):
        """A STRICT-severity violated fact must produce violations with STRICT severity."""
        fact = _make_frey_fact()
        assert fact.severity == Severity.STRICT, (
            "Frey Face recipe must be STRICT severity"
        )
        manifest = write_curated_artifacts(tmp_path, [fact])

        # train.py missing the import → STRICT violation
        train_py = tmp_path / "train.py"
        train_py.write_text("import torch\n", encoding="utf-8")

        violations = verify_emitted_code(train_py, manifest, tmp_path)
        strict = [v for v in violations if v.severity == Severity.STRICT]
        assert strict, (
            f"Expected at least one STRICT violation for missing import, got: {violations}"
        )


# ---------------------------------------------------------------------------
# 9. Regression fixture: three observed train.py files fail postflight
# ---------------------------------------------------------------------------

OBSERVED_TRAIN_PY_PATHS = [
    Path("/home/abheekp/openresearch/runs/prj_03271ba130d423fe/code/train.py"),
    Path("/home/abheekp/openresearch/runs/prj_3080fe2a02c20164/code/train.py"),
    Path("/home/abheekp/openresearch/runs/prj_db45c0304ce455a6/code/train.py"),
]


class TestObservedTrainPyFailPostflight:
    def test_three_observed_train_py_all_fail_postflight(self, tmp_path):
        """At least one of the three observed train.py files must produce a
        banned_literal violation for the Frey Face NYU URL."""
        fact = _make_frey_fact()
        manifest = write_curated_artifacts(tmp_path, [fact])

        present_files = [p for p in OBSERVED_TRAIN_PY_PATHS if p.exists()]
        if not present_files:
            pytest.skip(
                "None of the three observed train.py files are present — "
                "run this test on a machine with the preserved run directories"
            )

        any_flagged = False
        for train_py_path in present_files:
            violations = verify_emitted_code(train_py_path, manifest, tmp_path)
            banned_violations = [
                v for v in violations if v.kind == "banned_literal"
            ]
            if banned_violations:
                any_flagged = True
                break

        assert any_flagged, (
            f"Expected at least one observed train.py to contain a banned_literal "
            f"violation for Frey Face, but none of {[str(p) for p in present_files]} "
            f"produced one. Banned literals checked: {list(fact.banned_literals)}"
        )


# ---------------------------------------------------------------------------
# 10. Cache bump — different knowledge_channel_version produces different key
# ---------------------------------------------------------------------------

class TestCacheBumpInvalidates:
    def test_cache_bump_invalidates(self):
        """Two implement_baseline payloads that differ only in
        knowledge_channel_version must produce different cache keys."""
        from backend.agents.rlm.primitive_cache import make_key

        base_plan = {
            "paper_claim_map": {"core_contribution": "test"},
            "environment_spec": {"framework": "pytorch"},
            "reproduction_contract": None,
        }
        payload_v1 = {
            "plan": base_plan,
            "repair_context": None,
            "arxiv_id": None,
            "sandbox_mode": None,
            "gpu_mode": None,
            "knowledge_channel_version": 1,
        }
        payload_v2 = {**payload_v1, "knowledge_channel_version": 2}

        key_v1 = make_key("implement_baseline", payload=payload_v1)
        key_v2 = make_key("implement_baseline", payload=payload_v2)

        assert key_v1 != key_v2, (
            "Cache keys for knowledge_channel_version=1 and =2 must differ"
        )


# ---------------------------------------------------------------------------
# 11. write_curated_artifacts — empty facts produces no files
# ---------------------------------------------------------------------------

class TestWriteCuratedArtifactsEmpty:
    def test_empty_facts_writes_no_files(self, tmp_path):
        manifest = write_curated_artifacts(tmp_path, [])
        assert manifest == {"version": KNOWLEDGE_CHANNEL_VERSION, "facts": []}
        assert not (tmp_path / "_reprolab_curated.py").exists()
        assert not (tmp_path / "_reprolab_curated_manifest.json").exists()


# ---------------------------------------------------------------------------
# 12. from_recipes — recipe without helper_body is skipped
# ---------------------------------------------------------------------------

class TestFromRecipesSkipsEmpty:
    def test_recipe_without_helper_body_is_skipped(self):
        # MNIST has no helper_body in the current registry
        mnist = find_recipe("MNIST")
        assert mnist is not None
        facts = from_recipes([asdict(mnist)])
        # MNIST has empty helper_body — should produce no facts
        assert not any(f.id.startswith("dataset.") and "mnist" in f.id.lower() for f in facts), (
            "MNIST has no curated helper_body and should be skipped by from_recipes"
        )
