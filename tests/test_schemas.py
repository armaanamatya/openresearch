"""Tests for PaperClaimMap / MetricSpec coercion and validation in schemas.py."""

from __future__ import annotations

import pytest

from backend.agents.schemas import PaperClaimMap, MetricSpec, DatasetRequirement


# ---------------------------------------------------------------------------
# MetricSpec — definition defaults to ""
# ---------------------------------------------------------------------------

def test_metric_spec_definition_defaults_to_empty():
    m = MetricSpec(name="C2ST")
    assert m.definition == ""
    assert m.name == "C2ST"


def test_metric_spec_with_definition():
    m = MetricSpec(name="Accuracy", definition="fraction of correct predictions")
    assert m.definition == "fraction of correct predictions"


# ---------------------------------------------------------------------------
# PaperClaimMap — claims coercion
# ---------------------------------------------------------------------------

def test_claims_plain_string_coerced_to_dict():
    pcm = PaperClaimMap(core_contribution="x", claims=["some claim text"])
    assert len(pcm.claims) == 1
    item = pcm.claims[0]
    assert isinstance(item, dict)
    assert "some claim text" in item.values()


def test_claims_dict_passes_through():
    pcm = PaperClaimMap(
        core_contribution="x",
        claims=[{"method": "PPO", "dataset": "MuJoCo", "metric": "reward", "expected_result": "300"}],
    )
    assert pcm.claims[0]["method"] == "PPO"


def test_claims_mixed_strings_and_dicts():
    pcm = PaperClaimMap(
        core_contribution="x",
        claims=["plain text", {"method": "DQN", "dataset": "Atari"}],
    )
    assert len(pcm.claims) == 2
    assert isinstance(pcm.claims[0], dict)
    assert isinstance(pcm.claims[1], dict)
    assert pcm.claims[1]["method"] == "DQN"


# ---------------------------------------------------------------------------
# PaperClaimMap — metrics coercion
# ---------------------------------------------------------------------------

def test_metrics_dict_missing_definition_gets_default():
    pcm = PaperClaimMap(core_contribution="x", metrics=[{"name": "C2ST"}])
    assert len(pcm.metrics) == 1
    assert pcm.metrics[0].name == "C2ST"
    assert pcm.metrics[0].definition == ""


def test_metrics_plain_string_coerced_to_metric_spec():
    pcm = PaperClaimMap(core_contribution="x", metrics=["accuracy"])
    assert len(pcm.metrics) == 1
    assert pcm.metrics[0].name == "accuracy"
    assert pcm.metrics[0].definition == ""


def test_metrics_complete_dict_passes_through():
    pcm = PaperClaimMap(
        core_contribution="x",
        metrics=[{"name": "FID", "definition": "Frechet Inception Distance"}],
    )
    assert pcm.metrics[0].name == "FID"
    assert pcm.metrics[0].definition == "Frechet Inception Distance"


# ---------------------------------------------------------------------------
# PaperClaimMap — datasets coercion (Bug A's sibling: Qwen passes bare strings)
# ---------------------------------------------------------------------------

def test_datasets_plain_string_coerced_to_dict():
    pcm = PaperClaimMap(core_contribution="x", datasets=["Gaussian Linear"])
    assert len(pcm.datasets) == 1
    assert pcm.datasets[0].name == "Gaussian Linear"


def test_datasets_mixed_strings_and_dicts():
    pcm = PaperClaimMap(
        core_contribution="x",
        datasets=["Two Moons", {"name": "SLCP", "source": "sbibm"}],
    )
    assert len(pcm.datasets) == 2
    assert pcm.datasets[0].name == "Two Moons"
    assert pcm.datasets[1].name == "SLCP"
    assert pcm.datasets[1].source == "sbibm"


def test_datasets_accepts_prebuilt_model_instances():
    """The offline paper-understanding agent passes DatasetRequirement
    instances directly — the coercion must pass them through, not drop them.
    This guards the exact regression the coercion-passthrough fix addressed."""
    pcm = PaperClaimMap(
        core_contribution="x",
        datasets=[DatasetRequirement(name="MuJoCo"), DatasetRequirement(name="Atari")],
    )
    assert [d.name for d in pcm.datasets] == ["MuJoCo", "Atari"]


def test_metrics_accepts_prebuilt_model_instances():
    """MetricSpec instances pass through the coercion untouched."""
    pcm = PaperClaimMap(
        core_contribution="x",
        metrics=[MetricSpec(name="reward", definition="cumulative episode reward")],
    )
    assert pcm.metrics[0].name == "reward"
    assert pcm.metrics[0].definition == "cumulative episode reward"


# ---------------------------------------------------------------------------
# Regression — fully-formed PaperClaimMap still validates
# ---------------------------------------------------------------------------

def test_fully_formed_paper_claim_map():
    pcm = PaperClaimMap(
        core_contribution="Proximal Policy Optimization improves sample efficiency",
        claims=[
            {
                "method": "PPO",
                "dataset": "MuJoCo HalfCheetah",
                "metric": "mean episode reward",
                "expected_result": "3000+",
            }
        ],
        metrics=[{"name": "Mean Episode Reward", "definition": "average cumulative reward per episode"}],
        model_architecture="actor-critic MLP",
    )
    assert pcm.core_contribution.startswith("Proximal")
    assert pcm.claims[0]["method"] == "PPO"
    assert pcm.metrics[0].name == "Mean Episode Reward"
    assert pcm.metrics[0].definition == "average cumulative reward per episode"


# ---------------------------------------------------------------------------
# training_recipe coercion — pins the 2026-05-25 Adam regression
# (agent built method_spec with `str(s2.get("training_recipe", ""))[:300]`,
# turning a dict into a string, which crashed detect_environment's PaperClaimMap
# construction with `ValidationError: training_recipe: Input should be a valid
# dictionary or instance of TrainingRecipe`)
# ---------------------------------------------------------------------------


def test_training_recipe_dict_passes_through():
    pcm = PaperClaimMap(core_contribution="", training_recipe={"optimizer": "SGD", "learning_rate": "0.01"})
    assert pcm.training_recipe.optimizer == "SGD"
    assert pcm.training_recipe.learning_rate == "0.01"


def test_training_recipe_str_dict_repr_parsed():
    """The exact 2026-05-25 Adam pattern: str() of a dict."""
    pcm = PaperClaimMap(
        core_contribution="",
        training_recipe="{'optimizer': 'Adam', 'learning_rate': '0.001', 'batch_size': '128'}",
    )
    assert pcm.training_recipe.optimizer == "Adam"
    assert pcm.training_recipe.learning_rate == "0.001"
    assert pcm.training_recipe.batch_size == "128"


def test_training_recipe_str_prose_wraps_into_other_hparams():
    """Non-dict prose strings are preserved in other_hparams.raw, not lost."""
    pcm = PaperClaimMap(
        core_contribution="",
        training_recipe="Adam optimizer with lr=0.001 batch=128 dropout=0.5",
    )
    assert pcm.training_recipe.other_hparams.get("raw") == \
        "Adam optimizer with lr=0.001 batch=128 dropout=0.5"


def test_training_recipe_list_wraps_into_other_hparams():
    pcm = PaperClaimMap(core_contribution="", training_recipe=["Adam", "lr=0.001"])
    assert pcm.training_recipe.other_hparams.get("items") == ["Adam", "lr=0.001"]


def test_training_recipe_none_defaults_to_empty():
    pcm = PaperClaimMap(core_contribution="", training_recipe=None)
    assert pcm.training_recipe.optimizer == ""
    assert pcm.training_recipe.learning_rate == ""


def test_training_recipe_long_prose_truncated_to_1000():
    long = "x " * 1000  # 2000 chars
    pcm = PaperClaimMap(core_contribution="", training_recipe=long)
    assert len(pcm.training_recipe.other_hparams["raw"]) <= 1000
