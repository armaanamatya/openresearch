"""Unit tests for ScopeSpec and DatasetSlice schemas (PR A foundation)."""

from __future__ import annotations


from backend.agents.schemas import DatasetSlice, ScopeSpec


class TestDatasetSlice:
    def test_minimal(self):
        d = DatasetSlice(name="ALFWorld")
        assert d.name == "ALFWorld"
        assert d.episodes is None
        assert d.split is None
        assert d.normalized_id() == "ALFWorld"

    def test_full(self):
        d = DatasetSlice(name="WebShop", episodes=32, split="eval")
        assert d.episodes == 32
        assert d.split == "eval"


class TestScopeSpecConstruction:
    def test_empty_defaults(self):
        s = ScopeSpec()
        assert s.models == []
        assert s.skip_models == []
        assert s.datasets == []
        assert s.seeds == []
        assert s.eval_slice == {}
        assert s.budget_per_model == {}
        assert s.force_clean_cache is False
        assert s.free_text == ""

    def test_datasets_coerce_from_strings(self):
        s = ScopeSpec(datasets=["ALFWorld", "WebShop"])
        assert len(s.datasets) == 2
        assert all(isinstance(d, DatasetSlice) for d in s.datasets)
        assert [d.name for d in s.datasets] == ["ALFWorld", "WebShop"]

    def test_datasets_coerce_mixed(self):
        s = ScopeSpec(datasets=["ALFWorld", {"name": "WebShop", "episodes": 32}])
        assert s.datasets[0].episodes is None
        assert s.datasets[1].episodes == 32

    def test_datasets_passthrough_dict(self):
        s = ScopeSpec(datasets=[{"name": "X", "split": "eval"}])
        assert s.datasets[0].split == "eval"


class TestScopeSpecProperties:
    def test_is_multi_model(self):
        assert ScopeSpec(models=[]).is_multi_model is False
        assert ScopeSpec(models=["a"]).is_multi_model is False
        assert ScopeSpec(models=["a", "b"]).is_multi_model is True

    def test_is_multi_dataset(self):
        assert ScopeSpec(datasets=[]).is_multi_dataset is False
        assert ScopeSpec(datasets=["x"]).is_multi_dataset is False
        assert ScopeSpec(datasets=["x", "y"]).is_multi_dataset is True

    def test_dataset_ids(self):
        s = ScopeSpec(datasets=["A", "B"])
        assert s.dataset_ids() == ["A", "B"]


class TestRequestedEvidenceIds:
    def test_empty(self):
        assert ScopeSpec().requested_evidence_ids() == set()

    def test_models_only(self):
        s = ScopeSpec(models=["m1", "m2"])
        assert s.requested_evidence_ids() == {"m1", "m2"}

    def test_datasets_only(self):
        s = ScopeSpec(datasets=["d1", "d2"])
        assert s.requested_evidence_ids() == {"d1", "d2"}

    def test_cross_product(self):
        s = ScopeSpec(models=["m1", "m2"], datasets=["d1", "d2"])
        assert s.requested_evidence_ids() == {"m1/d1", "m1/d2", "m2/d1", "m2/d2"}


class TestScopeSpecMerge:
    def test_merge_with_none_paper_default(self):
        s = ScopeSpec(models=["m1"])
        merged = s.merge_with_paper_default(None)
        assert merged.models == ["m1"]
        # model_copy returns a new instance
        assert merged is not s

    def test_operator_wins_per_field(self):
        paper = ScopeSpec(models=["m1", "m2", "m3"], seeds=[42, 43, 44])
        op = ScopeSpec(models=["m1"])  # operator narrows to one model; no seeds set
        merged = op.merge_with_paper_default(paper)
        assert merged.models == ["m1"]               # operator wins
        assert merged.seeds == [42, 43, 44]          # falls back to paper

    def test_skip_models_removes_from_models(self):
        paper = ScopeSpec(models=["m1", "m2", "m3"])
        op = ScopeSpec(skip_models=["m2"])
        merged = op.merge_with_paper_default(paper)
        assert merged.models == ["m1", "m3"]
        assert "m2" in merged.skip_models

    def test_skip_models_union(self):
        paper = ScopeSpec(models=["m1", "m2"], skip_models=["legacy_a"])
        op = ScopeSpec(skip_models=["legacy_b"])
        merged = op.merge_with_paper_default(paper)
        assert set(merged.skip_models) == {"legacy_a", "legacy_b"}

    def test_free_text_concatenated(self):
        paper = ScopeSpec(free_text="paper guidance")
        op = ScopeSpec(free_text="operator note")
        merged = op.merge_with_paper_default(paper)
        assert "operator note" in merged.free_text
        assert "paper guidance" in merged.free_text
        # operator first (we write our own constraint first), paper second
        assert merged.free_text.index("operator note") < merged.free_text.index("paper guidance")

    def test_force_clean_cache_is_or(self):
        assert ScopeSpec(force_clean_cache=True).merge_with_paper_default(
            ScopeSpec(force_clean_cache=False)
        ).force_clean_cache is True
        assert ScopeSpec(force_clean_cache=False).merge_with_paper_default(
            ScopeSpec(force_clean_cache=True)
        ).force_clean_cache is True
        assert ScopeSpec(force_clean_cache=False).merge_with_paper_default(
            ScopeSpec(force_clean_cache=False)
        ).force_clean_cache is False
