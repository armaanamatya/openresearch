"""Budget estimator for paper reproduction runs.

Single public entry point: `estimate_paper_budget(source, *, recipe_mode, ...)`.
Makes one Sonnet LLM call to estimate training workload, then computes GPU cost
and API cost across all supported providers.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §estimator.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Codex C5 fix: 1.5 was the original spec value but doesn't absorb RunPod
# cold-start (5-10 min flat on a 30-min run = 20-33% overhead alone). 2.0
# covers cold-start (~10 min) + eval (~10%) + checkpoint I/O (~5%) + safety.
# Override via REPROLAB_ESTIMATE_OVERHEAD_MULTIPLIER for operators willing to
# run tighter.
_OVERHEAD_MULTIPLIER: float = float(
    os.environ.get("REPROLAB_ESTIMATE_OVERHEAD_MULTIPLIER", "2.0")
)
_COMPRESSED_RATIO: float = 0.15
_P90_MULTIPLIER: float = 1.4  # p50 → p90 for GPU hours
_MAX_PAPER_CHARS: int = 120_000  # ~30k tokens at 4 chars/token

# Maps ROOT_MODELS keys to MODEL_PRICING keys for the cost table.
# The estimator surfaces all provider options; sub-agent always uses
# anthropic.claude-sonnet-4-6 (or oauth) regardless of root.
_ROOT_MODEL_TO_PRICING_KEY: dict[str, str] = {
    "gpt-5": "openai.gpt-5",
    "qwen3-coder": "featherless.qwen3-coder-480b",
    "kimi-k2.5": "moonshot.kimi-k2-5",
    "claude": "anthropic.claude-opus-4-7",
    "claude-oauth": "anthropic.claude-oauth",
    "qwen3-coder-featherless": "featherless.qwen3-coder-480b",
    "azure-gpt-4o": "azure.gpt-4o",
}

# Paper categories for calibration lookup.
_PAPER_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "rl_policy": ["reinforcement", "policy gradient", "reward", "ppo", "grpo", "actor-critic"],
    "nlp_seq": ["transformer", "language model", "bert", "gpt", "attention", "token"],
    "vision_cls": ["image", "convolutional", "resnet", "vit", "vision", "segmentation"],
    "generative": ["diffusion", "vae", "gan", "generative", "latent"],
}


def _classify_paper(text: str) -> str:
    lower = text[:5000].lower()
    for category, patterns in _PAPER_CATEGORY_PATTERNS.items():
        if any(p in lower for p in patterns):
            return category
    return "nlp_seq"  # safe default


async def _fetch_pdf_bytes(source_kind: str, source: str) -> tuple[bytes, str]:
    """Return (pdf_bytes, paper_id).

    For arxiv_id / arxiv_url: fetch from arXiv.  For pdf_path: read from disk.
    Returns the raw bytes and a stable paper_id string.
    """
    import httpx

    if source_kind == "pdf_path":
        path = Path(source)
        return path.read_bytes(), path.stem

    if source_kind == "arxiv_id":
        arxiv_id = source.strip()
        url = f"https://arxiv.org/pdf/{arxiv_id}"
    elif source_kind == "arxiv_url":
        url = re.sub(r"^(https?://arxiv\.org)/abs/", r"\1/pdf/", source, flags=re.IGNORECASE)
        arxiv_id = url.rstrip("/").rsplit("/", 1)[-1]
    else:
        raise ValueError(f"Unknown source_kind: {source_kind!r}")

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"user-agent": "ReproLab/estimator"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return resp.content, arxiv_id


# Codex C3 fix: PDFs occasionally carry embedded text-layer content like
# author email signatures, leaked credentials in appendices, or quoted .env
# fragments. Anthropic logs every prompt input — sending raw bytes to the
# estimator's Sonnet call is a leakage vector. We strip lines that match
# common secret patterns *before* the text reaches the LLM. The patterns are
# conservative (false positives just redact a paragraph; false negatives
# leak); add new ones cautiously.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-(?:ant-|proj-|svcacct-|[A-Za-z0-9]{20,})[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|API_KEY)\s*[=:]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"gho_[A-Za-z0-9]{30,}"),
)


def _redact_secrets(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


# Codex C2 helper: best-effort regex extraction of paper hardware clues so
# the GPU resolver picks a realistic SKU instead of a hardcoded RTX 4090.
# Returns (gpu_string_for_resolver, vram_gb_estimate). Heuristics are
# conservative: a paper that explicitly says "H100 80GB" should resolve to
# h100; a paper with no GPU mention falls back to the safe 24GB default.
_GPU_MENTION_PATTERNS: tuple[tuple[re.Pattern[str], str, int], ...] = (
    (re.compile(r"\bH200\b", re.IGNORECASE), "H200", 141),
    (re.compile(r"\bH100[\s-]*SXM\b", re.IGNORECASE), "H100 SXM", 80),
    (re.compile(r"\bH100\b", re.IGNORECASE), "H100", 80),
    (re.compile(r"\bA100[\s-]*(?:SXM|80\s*GB)\b", re.IGNORECASE), "A100 80GB", 80),
    (re.compile(r"\bA100\b", re.IGNORECASE), "A100", 40),
    (re.compile(r"\bA6000\b", re.IGNORECASE), "A6000", 48),
    (re.compile(r"\bA5000\b", re.IGNORECASE), "A5000", 24),
    (re.compile(r"\bRTX[\s-]*4090\b", re.IGNORECASE), "RTX 4090", 24),
    (re.compile(r"\bV100[\s-]*32\s*GB\b", re.IGNORECASE), "V100 32GB", 32),
    (re.compile(r"\bV100\b", re.IGNORECASE), "V100", 16),
    (re.compile(r"\bP100\b", re.IGNORECASE), "P100", 16),
    (re.compile(r"\bTPU\b", re.IGNORECASE), "TPU", 16),  # treat as 16GB fallback
)


def _extract_gpu_clues(paper_text: str) -> tuple[str, int]:
    """Best-effort paper-text → (GPU label, VRAM estimate in GB).

    Picks the *highest-VRAM* explicit mention so an ablation paper that
    cites both RTX 4090 and H100 resolves to H100 (worst-case for cost).
    Falls back to (RTX 4090, 24GB) when no recognizable GPU is mentioned.
    """
    best_vram = 0
    best_label = ""
    for pat, label, vram_gb in _GPU_MENTION_PATTERNS:
        if pat.search(paper_text) and vram_gb > best_vram:
            best_vram = vram_gb
            best_label = label
    if best_label:
        return best_label, best_vram
    return "RTX 4090", 24


def _extract_text_from_pdf(pdf_bytes: bytes, max_chars: int = _MAX_PAPER_CHARS) -> str:
    """Extract text from PDF bytes using pymupdf (fitz), truncated to max_chars.

    Output is run through `_redact_secrets` so any embedded API keys / tokens
    / private keys are masked before downstream consumers (LLM, logs) see it.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("estimator: fitz not available; using empty paper text")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        doc = fitz.open(tmp.name)
        pages = [doc[i].get_text() for i in range(min(len(doc), 30))]
        doc.close()
    return _redact_secrets("\n".join(pages))[:max_chars]


async def _llm_estimate_workload(
    paper_text: str,
    sku_id: str,
    *,
    anthropic_api_key: str | None = None,
) -> dict:
    """Call Sonnet to estimate training workload.

    Returns dict with keys:
      experiment_count, total_epochs_across_all_experiments,
      avg_epoch_seconds_on_target_gpu, confidence
    Falls back to conservative defaults on any failure.
    """
    system = (
        "You are a research cost estimator. Analyze the paper and return JSON only.\n"
        "Return exactly: "
        '{"experiment_count": <int>, '
        '"total_epochs_across_all_experiments": <int>, '
        '"avg_epoch_seconds_on_target_gpu": <float>, '
        '"confidence": "high"|"medium"|"low"}\n'
        "Use the paper\'s reported training cost when stated; otherwise extrapolate "
        "from architecture, dataset size, and batch size. "
        f"Assume target GPU: {sku_id}."
    )
    prompt = (
        f"Paper text (first ~30k tokens):\n\n{paper_text[:_MAX_PAPER_CHARS]}\n\n"
        "Estimate training cost. Return JSON only."
    )

    try:
        import anthropic as _anthropic

        api_key = anthropic_api_key
        if not api_key:
            from backend.config import get_settings
            settings = get_settings()
            api_key = getattr(settings, "anthropic_api_key", None) or ""

        if not api_key:
            raise ValueError("No Anthropic API key available for LLM workload estimate")

        client = _anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip() if msg.content else "{}"
        parsed = json.loads(raw)
        return {
            "experiment_count": int(parsed.get("experiment_count", 1)),
            "total_epochs_across_all_experiments": int(parsed.get("total_epochs_across_all_experiments", 100)),
            "avg_epoch_seconds_on_target_gpu": float(parsed.get("avg_epoch_seconds_on_target_gpu", 30.0)),
            "confidence": parsed.get("confidence", "low"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("estimator: LLM workload estimate failed (%s), using defaults", exc)
        return {
            "experiment_count": 1,
            "total_epochs_across_all_experiments": 100,
            "avg_epoch_seconds_on_target_gpu": 30.0,
            "confidence": "low",
        }


def _compute_wall_clock_seconds(
    experiment_count: int,
    total_epochs: int,
    avg_epoch_seconds: float,
    recipe_mode: str,
) -> float:
    base = experiment_count * total_epochs * avg_epoch_seconds * _OVERHEAD_MULTIPLIER
    if recipe_mode == "compressed":
        return base * _COMPRESSED_RATIO
    return base


def _compute_api_cost_table(
    priors: dict[str, dict[str, float]],
    recipe_mode: str,
) -> list[dict]:
    """Compute per-model API cost for the cross-provider table."""
    from backend.services.pricing.calibration import _DEFAULT_PRIMITIVE_CALL_COUNTS
    from backend.services.pricing.catalog import MODEL_PRICING

    call_counts = _DEFAULT_PRIMITIVE_CALL_COUNTS.get(recipe_mode, _DEFAULT_PRIMITIVE_CALL_COUNTS["strict"])

    total_input = sum(
        priors.get(prim, {}).get("avg_input_tokens", 0) * count
        for prim, count in call_counts.items()
    )
    total_output = sum(
        priors.get(prim, {}).get("avg_output_tokens", 0) * count
        for prim, count in call_counts.items()
    )

    rows = []
    for model_key, entry in MODEL_PRICING.items():
        provider, model_id = model_key.split(".", 1)
        usd = (
            total_input * entry.usd_per_1m_input / 1_000_000
            + total_output * entry.usd_per_1m_output / 1_000_000
        )
        if entry.subscription_usd_per_month is not None and entry.assumed_runs_per_month:
            usd += entry.subscription_usd_per_month / entry.assumed_runs_per_month
        rows.append({
            "provider": provider,
            "model_id": model_id,
            "input_tokens": int(total_input),
            "output_tokens": int(total_output),
            "usd": round(usd, 4),
            "is_subscription": bool(entry.subscription_note or entry.subscription_usd_per_month),
            "subscription_note": entry.subscription_note or None,
        })
    return rows


async def estimate_paper_budget(
    source: str,
    *,
    source_kind: str = "arxiv_id",
    recipe_mode: Literal["strict", "compressed", "both"] = "both",
    target_root_model: str | None = None,
    runs_root: Path | None = None,
    anthropic_api_key: str | None = None,
) -> dict:
    """Estimate run cost for a paper.

    Args:
        source: arXiv ID, URL, or local PDF path.
        source_kind: "arxiv_id" | "arxiv_url" | "pdf_path".
        recipe_mode: "strict", "compressed", or "both".
        target_root_model: If set, surface only this model in the API table.
        runs_root: Override for the runs directory (for cache).
        anthropic_api_key: Override API key for the LLM workload call.

    Returns:
        A dict conforming to PaperBudgetEstimate (not yet validated; the route
        validates on the way out).

    Invariant 7: this function never spawns a subprocess.
    """
    from backend.agents.schemas import GpuRequirements
    from backend.config import get_settings
    from backend.services.pricing.cache import (
        CALIBRATION_SCHEMA_VERSION,
        get_cached,
        set_cached,
    )
    from backend.services.pricing.calibration import get_primitive_priors
    from backend.services.pricing.catalog import CATALOG_SCHEMA_VERSION, GPU_PRICING
    from backend.services.runtime.gpu_resolver import resolve as _resolve_gpu

    settings = get_settings()
    if runs_root is None:
        runs_root = Path(settings.runs_root) if settings.runs_root else Path("runs")

    # --- 1. Resolve paper identity
    pdf_bytes, paper_id = await _fetch_pdf_bytes(source_kind, source)
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    # Normalise recipe_mode: "both" → compute strict then compressed; cache separately.
    modes_to_compute: list[str] = (
        ["strict", "compressed"] if recipe_mode == "both" else [recipe_mode]
    )

    # --- Cache lookup — return immediately if both requested modes are cached
    cached_recipes: dict[str, dict] = {}
    for mode in modes_to_compute:
        hit = get_cached(runs_root, sha256, mode)
        if hit is not None:
            cached_recipes[mode] = hit

    if len(cached_recipes) == len(modes_to_compute):
        primary = cached_recipes[modes_to_compute[0]]
        return primary

    # --- 2. Extract paper text
    paper_text = _extract_text_from_pdf(pdf_bytes)
    paper_category = _classify_paper(paper_text)

    # --- 3. GPU resolution (lightweight — no SSE, no disk cache, no context needed)
    # Codex C2 fix: extract paper-mentioned GPU + a rough VRAM estimate from
    # the paper text *before* calling the resolver. The previous hardcoded
    # 24GB RTX 4090 was wrong for any H100-class paper. We use a regex pass
    # rather than a second LLM call to keep the estimator one round-trip.
    paper_gpu_string, estimated_vram_gb = _extract_gpu_clues(paper_text)
    default_req = GpuRequirements(
        estimated_vram_gb=estimated_vram_gb,
        paper_gpu_string=paper_gpu_string,
        paper_gpu_count=1,
        reasoning=(
            f"estimator regex extraction: gpu={paper_gpu_string!r}, "
            f"vram={estimated_vram_gb}GB"
        ),
        confidence=0.5,
    )
    try:
        from backend.config import get_settings as _gs
        _settings = _gs()
        _cloud_types: tuple[str, ...] = (
            ("COMMUNITY", "SECURE")
            if getattr(_settings, "runpod_cloud_type", "COMMUNITY") == "SECURE"
            else ("COMMUNITY",)
        )
        gpu_plan = _resolve_gpu(
            default_req,
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=None,
            headroom_multiplier=1.25,
            fallback_vram_gb=24,
            cloud_types=_cloud_types,
        )
        sku_id = gpu_plan.short_name
        usd_per_hour = gpu_plan.sku_usd_per_hr
    except Exception as exc:  # noqa: BLE001
        logger.warning("estimator: GPU resolution failed (%s), using rtx4090 fallback", exc)
        sku_id = "rtx4090"
        usd_per_hour = GPU_PRICING["rtx4090"].usd_per_hour

    # Codex I6 fix: derive label from the resolved SKU's cloud_type instead
    # of hardcoded "COMMUNITY".
    _entry = GPU_PRICING.get(sku_id)
    _cloud_label = _entry.cloud_type if _entry is not None else "COMMUNITY"
    sku_label = f"{sku_id.upper()} (RunPod {_cloud_label})"

    # --- 4. LLM workload estimate (one call) — fail-soft: defaults if LLM fails
    try:
        workload = await _llm_estimate_workload(
            paper_text,
            sku_id,
            anthropic_api_key=anthropic_api_key,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("estimator: workload estimate raised (%s), using defaults", exc)
        workload = {
            "experiment_count": 1,
            "total_epochs_across_all_experiments": 100,
            "avg_epoch_seconds_on_target_gpu": 30.0,
            "confidence": "low",
        }
    experiment_count = workload["experiment_count"]
    total_epochs = workload["total_epochs_across_all_experiments"]
    avg_epoch_seconds = workload["avg_epoch_seconds_on_target_gpu"]
    llm_confidence = workload["confidence"]

    # --- 5 + 6 + 7 + 8. Compute for each recipe mode
    calibration_n = 0
    try:
        from backend.services.pricing.calibration import _calibration_path, recompute_calibration
        _cal_path = _calibration_path()
        if _cal_path.exists():
            import json as _json
            cal_raw = _json.loads(_cal_path.read_text(encoding="utf-8"))
            calibration_n = cal_raw.get("based_on_n_preserved_runs", 0)
    except Exception:  # noqa: BLE001
        pass

    # Codex C1 fix: was max(100, ...) which always returned 100, defeating the
    # whole calibration feedback loop. The intent is "start at 100, shrink as
    # runs accumulate, floor at 10". Each preserved run shaves 5 pts off the
    # window; capped at the 10-100 range.
    precision_window = max(10, min(100, 100 - calibration_n * 5))

    now_utc = datetime.now(timezone.utc).isoformat()
    results: dict[str, dict] = {}

    for mode in modes_to_compute:
        if mode in cached_recipes:
            results[mode] = cached_recipes[mode]
            continue

        priors = get_primitive_priors(paper_category, mode)
        wall_clock_seconds = _compute_wall_clock_seconds(
            experiment_count, total_epochs, avg_epoch_seconds, mode
        )
        wall_clock_hours_p50 = wall_clock_seconds / 3600.0
        wall_clock_hours_p90 = wall_clock_hours_p50 * _P90_MULTIPLIER

        gpu_usd_p50 = wall_clock_hours_p50 * usd_per_hour
        gpu_usd_p90 = wall_clock_hours_p90 * usd_per_hour

        api_rows = _compute_api_cost_table(priors, mode)
        api_usds = [r["usd"] for r in api_rows if not r["is_subscription"] or r["usd"] > 0]
        api_usd_best = min(api_usds, default=0.0)
        api_usd_worst = max(api_usds, default=0.0)

        if mode == "strict":
            recipe_label = "Strict reproduction"
            recipe_description = "Paper's training recipe verbatim."
            fidelity_label = "high"
            declared_reductions: list[str] = []
        else:
            recipe_label = "Claim-match (minimize-compute)"
            recipe_description = (
                f"Modern fast equivalent (~{int(_COMPRESSED_RATIO * 100)}% compute "
                "of paper recipe). Validates claims, not full reproducibility."
            )
            fidelity_label = "claim-match"
            declared_reductions = [
                "Replaced paper training schedule with a compressed equivalent.",
                f"~{int((1 - _COMPRESSED_RATIO) * 100)}% fewer epochs.",
            ]

        estimate_id = (
            f"{sha256[:8]}_{mode}_{CATALOG_SCHEMA_VERSION}_{CALIBRATION_SCHEMA_VERSION}"
        )
        estimate = {
            "paper": {"id": paper_id, "title": paper_id, "sha256": sha256},
            "gpu": {
                "sku_id": sku_id,
                "label": sku_label,
                "usd_per_hour": usd_per_hour,
                "estimated_hours": {"p50": round(wall_clock_hours_p50, 2), "p90": round(wall_clock_hours_p90, 2)},
                "usd_total": {"p50": round(gpu_usd_p50, 2), "p90": round(gpu_usd_p90, 2)},
            },
            "api": api_rows,
            "recipes": {
                mode: {
                    "label": recipe_label,
                    "description": recipe_description,
                    "gpu_usd": round(gpu_usd_p50, 2),
                    "api_usd_best": round(api_usd_best, 4),
                    "api_usd_worst": round(api_usd_worst, 4),
                    "wall_clock_hours_p50": round(wall_clock_hours_p50, 2),
                    "fidelity_label": fidelity_label,
                    "declared_reductions": declared_reductions,
                }
            },
            "calibration_metadata": {
                "based_on_n_preserved_runs": calibration_n,
                "precision_window_pct": precision_window,
                "catalog_schema_version": CATALOG_SCHEMA_VERSION,
                "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
                "estimated_at_utc": now_utc,
            },
            "estimate_id": estimate_id,
        }
        set_cached(runs_root, sha256, mode, estimate)
        results[mode] = estimate

    # --- 9. Merge if both recipes computed
    if recipe_mode == "both" and "strict" in results and "compressed" in results:
        strict = results["strict"]
        compressed = results["compressed"]
        merged_recipes = {
            "strict": strict["recipes"]["strict"],
            "compressed": compressed["recipes"]["compressed"],
        }
        base = dict(strict)
        base["recipes"] = merged_recipes
        base["estimate_id"] = (
            f"{sha256[:8]}_both_{CATALOG_SCHEMA_VERSION}_{CALIBRATION_SCHEMA_VERSION}"
        )
        return base

    primary_mode = modes_to_compute[0]
    return results[primary_mode]
