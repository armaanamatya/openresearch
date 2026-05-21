"""Tests for PaperClaimMap / MetricSpec coercion and validation in schemas.py."""

from __future__ import annotations

import pytest

from backend.agents.schemas import PaperClaimMap, MetricSpec


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
