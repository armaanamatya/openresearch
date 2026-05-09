"""Paper Understanding Agent — LLM-powered claim extraction.

This module provides two modes:
  1. ``run_with_sdk()`` — invokes the configured agent runtime for LLM analysis
  2. ``run_offline()`` — deterministic extraction without LLM (for tests/CI)

Both produce a PaperClaimMap and write it to the project's runs directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.agents.runtime.base import AgentRuntime, ProviderName
from backend.agents.schemas import (
    Ambiguity,
    DatasetRequirement,
    MetricSpec,
    PaperClaimMap,
    RiskLevel,
    TrainingRecipe,
)

logger = logging.getLogger(__name__)


def run_offline(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
) -> PaperClaimMap:
    """Deterministic paper understanding without LLM.

    Extracts structure from the workspace claim_map (section titles + excerpts)
    and applies heuristic extraction. Good for tests and offline demos.
    """
    entries = workspace_claim_map.get("entries", [])

    # Extract sections
    sections: dict[str, str] = {}
    for entry in entries:
        title = entry.get("title", "").lower()
        excerpt = entry.get("excerpt", "")
        sections[title] = excerpt

    # Heuristic extraction
    core_contribution = _extract_contribution(sections)
    claims = _extract_claims(sections)
    datasets = _extract_datasets(sections)
    metrics = _extract_metrics(sections)
    architecture = _extract_architecture(sections)
    recipe = _extract_training_recipe(sections)
    eval_protocol = _extract_eval_protocol(sections)
    hardware = _extract_hardware(sections)
    ambiguities = _extract_ambiguities(sections)

    claim_map = PaperClaimMap(
        core_contribution=core_contribution,
        claims=claims,
        datasets=datasets,
        metrics=metrics,
        model_architecture=architecture,
        training_recipe=recipe,
        evaluation_protocol=eval_protocol,
        hardware_clues=hardware,
        ambiguities=ambiguities,
    )

    # Write to disk
    out_dir = Path(runs_root) / project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paper_claim_map.json"
    out_path.write_text(claim_map.model_dump_json(indent=2))
    logger.info("Paper claim map written to %s", out_path)

    return claim_map


async def run_with_sdk(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
    *,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
) -> PaperClaimMap:
    """Full LLM-powered paper understanding via the configured agent runtime."""
    from backend.agents.runtime.invoke import collect_agent_text

    project_dir = Path(runs_root) / project_id
    claim_map_context = json.dumps(workspace_claim_map, indent=2)

    prompt = (
        f"Analyze the paper for project {project_id}.\n"
        f"The parsed workspace claim_map is:\n```json\n{claim_map_context}\n```\n\n"
        f"Read any additional files in {project_dir} for more context.\n"
        f"Write the complete PaperClaimMap JSON to {project_dir}/paper_claim_map.json"
    )

    full_text = await collect_agent_text(
        "paper-understanding",
        prompt,
        project_dir=project_dir,
        model=model,
        provider=provider,
        runtime=runtime,
        max_turns=20,
    )

    # Try to read the written file first
    out_path = project_dir / "paper_claim_map.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        return PaperClaimMap(**data)

    # Fall back to parsing from agent output
    data = _extract_json(full_text)
    claim_map = PaperClaimMap(**data)
    out_path.write_text(claim_map.model_dump_json(indent=2))
    return claim_map


# ---------------------------------------------------------------------------
# Heuristic extractors (offline mode)
# ---------------------------------------------------------------------------

def _extract_contribution(sections: dict[str, str]) -> str:
    """Extract core contribution from abstract."""
    abstract = sections.get("abstract", "")
    if abstract:
        # Take first 2 sentences as contribution summary
        sentences = abstract.replace("\n", " ").split(". ")
        return ". ".join(sentences[:2]).strip() + "."
    return "Core contribution not found in paper sections."


def _extract_claims(sections: dict[str, str]) -> list[dict[str, str]]:
    """Extract testable claims from experiments/results sections."""
    claims: list[dict[str, str]] = []
    for title, text in sections.items():
        if any(kw in title for kw in ("experiment", "result", "evaluation", "abstract")):
            claims.append({
                "method": _extract_method_name(sections),
                "dataset": _find_dataset_mentions(text),
                "metric": _find_metric_mentions(text),
                "expected_result": _find_result_numbers(text),
            })
    if not claims:
        claims.append({
            "method": _extract_method_name(sections),
            "dataset": "See paper",
            "metric": "See paper",
            "expected_result": "See paper",
        })
    return claims


def _extract_datasets(sections: dict[str, str]) -> list[DatasetRequirement]:
    """Find dataset mentions across all sections."""
    all_text = " ".join(sections.values()).lower()
    datasets: list[DatasetRequirement] = []

    known_datasets = {
        "cartpole": ("CartPole-v1", "Gymnasium", "bundled"),
        "cifar": ("CIFAR-10", "torchvision", "torchvision.datasets.CIFAR10"),
        "mnist": ("MNIST", "torchvision", "torchvision.datasets.MNIST"),
        "imagenet": ("ImageNet", "image-net.org", "manual download"),
        "mujoco": ("MuJoCo", "Gymnasium", "pip install gymnasium[mujoco]"),
        "atari": ("Atari", "Gymnasium", "pip install gymnasium[atari]"),
    }

    for keyword, (name, source, download) in known_datasets.items():
        if keyword in all_text:
            datasets.append(DatasetRequirement(
                name=name, source=source, download_method=download,
            ))

    return datasets


def _extract_metrics(sections: dict[str, str]) -> list[MetricSpec]:
    """Find metric definitions."""
    metrics: list[MetricSpec] = []
    all_text = " ".join(sections.values()).lower()

    known_metrics = [
        ("reward", "Mean episode reward"),
        ("accuracy", "Classification accuracy"),
        ("loss", "Training loss"),
        ("return", "Cumulative return"),
        ("error rate", "Error rate"),
    ]

    for keyword, definition in known_metrics:
        if keyword in all_text:
            metrics.append(MetricSpec(name=keyword, definition=definition))

    return metrics


def _extract_architecture(sections: dict[str, str]) -> str:
    """Extract model architecture description."""
    for title, text in sections.items():
        if any(kw in title for kw in ("method", "model", "architecture", "approach")):
            return text[:500].replace("\n", " ").strip()
    return "Architecture details not found."


def _extract_training_recipe(sections: dict[str, str]) -> TrainingRecipe:
    """Extract training hyperparameters."""
    all_text = " ".join(sections.values())
    recipe = TrainingRecipe()

    import re
    lr_match = re.search(r"learning.?rate[:\s]+([0-9.e-]+)", all_text, re.IGNORECASE)
    if lr_match:
        recipe.learning_rate = lr_match.group(1)

    opt_match = re.search(r"(Adam|SGD|RMSprop|AdamW)", all_text)
    if opt_match:
        recipe.optimizer = opt_match.group(1)

    batch_match = re.search(r"batch.?size[:\s]+(\d+)", all_text, re.IGNORECASE)
    if batch_match:
        recipe.batch_size = batch_match.group(1)

    epoch_match = re.search(r"(\d+)\s*(?:epochs?|iterations?|timesteps?|steps)", all_text, re.IGNORECASE)
    if epoch_match:
        recipe.epochs_or_steps = epoch_match.group(0)

    return recipe


def _extract_eval_protocol(sections: dict[str, str]) -> str:
    """Extract evaluation protocol."""
    for title, text in sections.items():
        if any(kw in title for kw in ("experiment", "evaluation", "result")):
            return text[:300].replace("\n", " ").strip()
    return "Evaluation protocol not found."


def _extract_hardware(sections: dict[str, str]) -> list[str]:
    """Find hardware mentions."""
    import re
    all_text = " ".join(sections.values())
    hardware: list[str] = []

    for pattern in [
        r"(?:GPU|NVIDIA|RTX|GTX|V100|A100|P100|T4|H100)[^\n.]*",
        r"(?:CPU|cores?)[^\n.]*\d+",
        r"\d+\s*GB\s*(?:memory|RAM|VRAM)",
    ]:
        for match in re.findall(pattern, all_text, re.IGNORECASE):
            hardware.append(match.strip())

    return hardware


def _extract_ambiguities(sections: dict[str, str]) -> list[Ambiguity]:
    """Detect common ambiguities in ML papers."""
    ambiguities: list[Ambiguity] = []
    all_text = " ".join(sections.values()).lower()
    idx = 1

    # Common missing details in ML papers
    checks = [
        ("adam epsilon", "Adam optimizer epsilon value", "1e-5", RiskLevel.high),
        ("weight init", "Weight initialization scheme", "orthogonal", RiskLevel.high),
        ("learning rate schedule", "Learning rate schedule/decay", "linear decay", RiskLevel.medium),
        ("advantage normalization", "Advantage normalization method", "per-minibatch", RiskLevel.medium),
        ("value loss clip", "Value function loss clipping", "clipped", RiskLevel.medium),
        ("gradient clip", "Gradient clipping method/threshold", "max_grad_norm=0.5", RiskLevel.medium),
        ("gae lambda", "GAE lambda value", "0.95", RiskLevel.low),
        ("entropy", "Entropy bonus coefficient", "0.01", RiskLevel.low),
        ("random seed", "Random seed for reproducibility", "not specified", RiskLevel.medium),
        ("number of workers", "Number of parallel workers/environments", "not specified", RiskLevel.medium),
    ]

    for keyword, detail, default, risk in checks:
        # Mark as ambiguity if the keyword is NOT mentioned (common pattern)
        if keyword not in all_text:
            ambiguities.append(Ambiguity(
                assumption_id=f"A{idx:03d}",
                detail=f"{detail} not specified in paper",
                chosen_value=default,
                evidence=[],
                risk=risk,
            ))
            idx += 1

    return ambiguities


def _extract_method_name(sections: dict[str, str]) -> str:
    """Try to find the method name from title/abstract."""
    abstract = sections.get("abstract", "")
    if "proximal policy optimization" in abstract.lower() or "ppo" in abstract.lower():
        return "PPO"
    if "mixmatch" in abstract.lower():
        return "MixMatch"
    # Generic fallback
    return abstract.split(".")[0][:100] if abstract else "Unknown method"


def _find_dataset_mentions(text: str) -> str:
    """Find dataset names in text."""
    text_lower = text.lower()
    for name in ["CartPole-v1", "CIFAR-10", "ImageNet", "MNIST", "MuJoCo"]:
        if name.lower() in text_lower:
            return name
    return "See paper"


def _find_metric_mentions(text: str) -> str:
    """Find metric names in text."""
    text_lower = text.lower()
    for metric in ["reward", "accuracy", "loss", "return", "error rate", "f1"]:
        if metric in text_lower:
            return metric
    return "See paper"


def _find_result_numbers(text: str) -> str:
    """Find reported result numbers."""
    import re
    numbers = re.findall(r"\d+\.?\d*", text[:500])
    if numbers:
        return f"See paper (values: {', '.join(numbers[:5])})"
    return "See paper"


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from text, handling markdown fences."""
    import re
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[brace_start : i + 1])
    raise ValueError(f"No JSON found in output: {text[:200]}")
