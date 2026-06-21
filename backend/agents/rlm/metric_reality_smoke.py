"""
P2 — metric-reality smoke (§4.2 of 2026-06-20-pre-gpu-code-review-and-report-validation-design.md).

Runs ~2 training steps on a tiny slice of each representative cell BEFORE the
full GPU grid to assert POSITIVE proof that the training loop backprops a real
loss and produces varied, non-zero metrics.

Design split:
  - evaluate_smoke_trace(trace, peak_vram_gb)  — PURE; testable without I/O.
  - run_metric_reality_smoke(*, ctx, code_dir, cells)  — I/O; mocks in tests.

Default-OFF: OPENRESEARCH_METRIC_REALITY_SMOKE.
Fail-OPEN only when the smoke CANNOT LAUNCH (no GPU, subprocess won't spawn).
Any executor-side outcome — including "ran but wrote no trace" — is JUDGED.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def metric_reality_smoke_enabled() -> bool:
    """True iff OPENRESEARCH_METRIC_REALITY_SMOKE opts the smoke ON."""
    return (
        os.environ.get("OPENRESEARCH_METRIC_REALITY_SMOKE", "").strip().lower()
        in _ENABLED_VALUES
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default: smoke at most this many cells.
_DEFAULT_SMOKE_MAX_CELLS = 4
# Default per-cell timeout for the smoke. 900s (was 300s): loading a real model +
# env + running one tiny step exceeds 300s for ML cells, so a 300s smoke timed out
# during model-load and got wrongly judged. A timeout is now treated as INCONCLUSIVE
# (fail-open), not a rejection — but a generous budget still lets fast cells complete
# and be verified empirically.
_DEFAULT_SMOKE_TIMEOUT_S = 900


def _smoke_max_cells() -> int:
    try:
        return max(1, int(os.environ.get("OPENRESEARCH_SMOKE_MAX_CELLS", "") or _DEFAULT_SMOKE_MAX_CELLS))
    except (ValueError, TypeError):
        return _DEFAULT_SMOKE_MAX_CELLS


def _smoke_timeout_s() -> float:
    try:
        v = float(os.environ.get("OPENRESEARCH_SMOKE_TIMEOUT_S", "") or _DEFAULT_SMOKE_TIMEOUT_S)
        return max(30.0, v)
    except (ValueError, TypeError):
        return _DEFAULT_SMOKE_TIMEOUT_S


# ---------------------------------------------------------------------------
# PURE: evaluate_smoke_trace
# ---------------------------------------------------------------------------

# Minimum VRAM (GiB) that counts as evidence of real GPU training.
_VRAM_FLOOR_GIB = 0.5

# PRIMARY training-loss keys, in priority order.  These are the optimised loss
# (the thing backprop minimises) — NOT diagnostics like kl/nll.  The smoke checks
# the PRIMARY loss's per-step series varies, so a constant training loss masked by
# a varying kl cannot false-pass (fix codex review #2).
_PRIMARY_LOSS_KEYS = ("loss", "total_loss", "l_grpo", "pg_loss", "policy_loss", "grpo_loss")

# Lenient fallback loss-key pattern (used only when no PRIMARY key is present).
_LOSS_KEY_RE = re.compile(r"\b(?:loss|l_grpo|pg_loss|kl|nll)\b", re.IGNORECASE)


def _is_loss_key(k: str) -> bool:
    return bool(_LOSS_KEY_RE.search(k))


# Tokens that disqualify a "*loss" key from counting as the OPTIMISED training
# objective: eval-split losses (val/test/dev) prove nothing about backprop, and
# loss_scale is an AMP gradient-scaling factor, not a loss.
_NON_TRAINING_LOSS_TOKENS = frozenset(
    {"val", "valid", "validation", "eval", "test", "dev", "holdout", "scale"}
)


def _is_primary_loss_key(k: str) -> bool:
    """True iff ``k`` names the optimised training loss (what backprop minimises).

    Token-aware: ``train_loss``/``ce_loss``/``mse_loss``/``reconstruction_loss`` are
    recognised (the exact-match ``_PRIMARY_LOSS_KEYS`` missed every ``*_loss`` name,
    and the word-boundary fallback regex missed them too because ``_`` is a word
    char), while eval-split losses (``val_loss``/``test_loss``) and the AMP
    ``loss_scale`` factor are excluded — a varying val_loss on a frozen model is not
    proof of training.
    """
    lk = k.lower()
    if lk in _PRIMARY_LOSS_KEYS:
        return True
    toks = re.split(r"[^a-z0-9]+", lk)
    if "loss" not in toks:
        return False
    return not any(t in _NON_TRAINING_LOSS_TOKENS for t in toks)


def _grad_param_values(step_records: list[dict]) -> tuple[list[float], list[float]]:
    """Collect grad_norm and param_delta_* values across step records (floats only)."""
    grads: list[float] = []
    deltas: list[float] = []
    for rec in step_records:
        for k, v in rec.items():
            if not isinstance(k, str):
                continue
            lk = k.lower()
            if "grad_norm" in lk:
                try:
                    grads.append(float(v))
                except (TypeError, ValueError):
                    pass
            if "param_delta" in lk:
                try:
                    deltas.append(float(v))
                except (TypeError, ValueError):
                    pass
    return grads, deltas


def _get_float(d: dict, key_pred) -> list[float]:
    """Return list of floats for keys in d that satisfy key_pred (excluding config keys)."""
    from backend.agents.rlm.zero_metrics_detection import _is_excluded_key  # noqa: PLC0415
    result = []
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        if _is_excluded_key(k):
            continue
        if key_pred(k):
            try:
                result.append(float(v))
            except (TypeError, ValueError):
                pass
    return result


def _primary_loss_series(step_records: list[dict]) -> list[float]:
    """Per-step series of the PRIMARY loss (first present primary key per record).

    Canonical ``_PRIMARY_LOSS_KEYS`` win by priority; otherwise the first token-loss
    key (sorted, deterministic) is used so ``train_loss``-style traces are covered.
    """
    series: list[float] = []
    for rec in step_records:
        str_keys = [k for k in rec.keys() if isinstance(k, str)]
        lowered = {k.lower(): k for k in str_keys}
        chosen: str | None = None
        for pk in _PRIMARY_LOSS_KEYS:
            if pk in lowered:
                chosen = lowered[pk]
                break
        if chosen is None:
            cands = sorted(k for k in str_keys if _is_primary_loss_key(k))
            chosen = cands[0] if cands else None
        if chosen is not None:
            try:
                series.append(float(rec[chosen]))
            except (TypeError, ValueError):
                pass
    return series


def evaluate_smoke_trace(
    trace: list[dict] | dict | Any,
    peak_vram_gb: float | None,
) -> dict:
    """Pure check of a smoke trace dict or list-of-dicts.

    Returns {ok: bool, failure_class: str|None, detail: str}.

    Requirements for ok=True:
      1. trace is a list with >=2 step records.
      2. The PRIMARY training loss varies across steps and is not all-zero.
         (A varying diagnostic like kl cannot satisfy this — fix #2.)  When no
         primary key is present, fall back to "any loss-like key varies".
      3. If peak_vram_gb is not None, it must be > _VRAM_FLOOR_GIB.
      4. grad_norm / param_delta_norm — when present — must be finite and > 0.

    Reward is informational only: a ~2-step sparse-reward smoke (ALFWorld)
    legitimately has reward 0.0, so an all-zero reward is NOT a failure here
    (fix #3); the post-run zero-metrics guard covers reward on the full run.
    """
    # Normalise to list-of-dicts.
    if isinstance(trace, dict):
        trace = [trace]
    if not isinstance(trace, list):
        return {
            "ok": False,
            "failure_class": "smoke_metrics_unreal",
            "detail": "smoke trace is not a list or dict",
        }

    step_records = [r for r in trace if isinstance(r, dict)]

    # Requirement 1: >=1 step record. Relaxed from >=2 — for slow-rollout RL envs
    # (ALFWorld ~20-30 min/step) a 2-step smoke is infeasible; one step with positive
    # backprop evidence still catches the disconnected/fake-loss failure. Step-variation
    # is checked below only when >=2 records are present.
    if len(step_records) < 1:
        return {
            "ok": False,
            "failure_class": "smoke_metrics_unreal",
            "detail": "smoke trace has 0 step records — trainer produced no per-step metrics",
        }

    # Grad/param evidence is collected once: it both gates the no-loss-key path
    # (FIX 2) and drives the Requirement-4 sanity check below.
    grad_norms, param_deltas = _grad_param_values(step_records)
    has_grad_evidence = bool(grad_norms or param_deltas)

    # Requirement 2: PRIMARY loss present and >0; varied across steps when >=2 records.
    # With a single record, loss>0 alone catches the v6 disconnected-loss (0.0) failure;
    # the constant-across-steps check needs >=2 records and is skipped for one record.
    primary = _primary_loss_series(step_records)
    if primary:
        if all(v == 0.0 for v in primary):
            return {
                "ok": False,
                "failure_class": "smoke_metrics_unreal",
                "detail": (
                    f"smoke trace: the primary loss is 0.0 across all {len(primary)} step(s) — "
                    f"loss is not connected to the model graph"
                ),
            }
        if len(primary) >= 2 and len(set(primary)) == 1:
            return {
                "ok": False,
                "failure_class": "smoke_metrics_unreal",
                "detail": (
                    f"smoke trace: the primary loss is constant across {len(primary)} steps "
                    f"(value={primary[0]}) — training is not updating the model"
                ),
            }
    else:
        # Fallback: no PRIMARY loss key present — use the lenient pooled loss check.
        all_loss: list[float] = []
        for rec in step_records:
            all_loss.extend(_get_float(rec, _is_loss_key))
        if not all_loss:
            # No training-loss key at all. Accept iff there is POSITIVE backprop
            # evidence (grad_norm/param_delta, validated finite & >0 by Requirement 4
            # below); otherwise reject with an actionable, paper-agnostic hint. The
            # post-run zero-metrics + evidence guards remain the backstop.
            if not has_grad_evidence:
                return {
                    "ok": False,
                    "failure_class": "smoke_metrics_unreal",
                    "detail": (
                        "smoke trace has step records but no training-loss key and no "
                        "grad_norm/param_delta — log a per-step loss (or grad_norm) so "
                        "the smoke can verify backprop"
                    ),
                }
            # grad evidence present: fall through; Requirement 4 is the gate.
        elif all(v == 0.0 for v in all_loss):
            return {
                "ok": False,
                "failure_class": "smoke_metrics_unreal",
                "detail": f"smoke trace: all {len(all_loss)} loss values are 0.0 — loss is disconnected",
            }
        elif len(all_loss) >= 2 and len(set(all_loss)) == 1:
            return {
                "ok": False,
                "failure_class": "smoke_metrics_unreal",
                "detail": (
                    f"smoke trace: loss is constant across {len(all_loss)} values "
                    f"(value={all_loss[0]}) — training is not updating the model"
                ),
            }

    # Requirement 3: VRAM floor (only when peak_vram_gb is provided).
    if peak_vram_gb is not None and peak_vram_gb <= _VRAM_FLOOR_GIB:
        return {
            "ok": False,
            "failure_class": "smoke_metrics_unreal",
            "detail": (
                f"smoke trace: peak VRAM {peak_vram_gb:.3f} GiB <= "
                f"{_VRAM_FLOOR_GIB} GiB floor — GPU training claim not supported"
            ),
        }

    # Requirement 4: grad_norm / param_delta_norm finite and >0 when present.
    all_grad_norms = grad_norms
    all_param_deltas = param_deltas
    if all_grad_norms:
        bad = [v for v in all_grad_norms if not math.isfinite(v) or v <= 0.0]
        if bad:
            return {
                "ok": False,
                "failure_class": "smoke_metrics_unreal",
                "detail": (
                    f"smoke trace: grad_norm values include non-finite or <=0 entries "
                    f"({bad[:3]}) — gradients are not flowing"
                ),
            }
    if all_param_deltas:
        bad = [v for v in all_param_deltas if not math.isfinite(v) or v <= 0.0]
        if bad:
            return {
                "ok": False,
                "failure_class": "smoke_metrics_unreal",
                "detail": (
                    f"smoke trace: param_delta_norm values include non-finite or <=0 entries "
                    f"({bad[:3]}) — parameters are not updating"
                ),
            }

    return {"ok": True, "failure_class": None, "detail": ""}


# ---------------------------------------------------------------------------
# Cell grouping
# ---------------------------------------------------------------------------

def _group_key(cell: dict) -> tuple[str, str, str, int]:
    """Return a grouping key: (model_id, eval_env, script_family, gpus)."""
    model = str(
        cell.get("model_key") or cell.get("model_id") or cell.get("model") or "default"
    )
    env = str(cell.get("env") or cell.get("eval_env") or "default")
    script = str(cell.get("script") or cell.get("train_script") or "train_cell.py")
    script_family = Path(script).stem
    gpus = max(1, int(cell.get("gpus", 1) if cell.get("gpus") else 1))
    return (model, env, script_family, gpus)


def _select_smoke_representatives(cells: list[dict], max_cells: int) -> list[dict]:
    """Pick one cell per (model_id, eval_env, script_family, gpus) group, capped."""
    seen: dict[tuple, dict] = {}
    for cell in cells:
        key = _group_key(cell)
        if key not in seen:
            seen[key] = cell
    representatives = [seen[k] for k in sorted(seen)]
    return representatives[:max_cells]


# ---------------------------------------------------------------------------
# Single-cell smoke runner
# ---------------------------------------------------------------------------

class _SmokeCellResult(NamedTuple):
    trace: dict | list | None
    peak_vram_gb: float | None
    launched: bool
    timed_out: bool
    returncode: int | None
    output_tail: str


def _run_one_smoke_cell(
    cell: dict,
    cell_script: Path,
    gpu_id: str,
    output_dir: Path,
    timeout_s: float,
) -> _SmokeCellResult:
    """Run one cell for ~2 steps; return (trace, peak_vram_gb, launched, timed_out, returncode, output_tail).

    `launched` is True whenever the subprocess actually started; False ONLY when it
    could not be spawned (OSError) — a harness fail-open signal.
    `timed_out` is True when the subprocess hit `timeout_s` and was killed — for a
    slow-rollout env that is INCONCLUSIVE (the caller fails open), not broken code; a
    NATURAL exit (timed_out False) with no/bad trace is still judged (codex Area-4).
    `returncode` is the subprocess exit code (None if not yet set or timed-out kill).
    `output_tail` is the last 4000 characters of combined stdout+stderr — the traceback
    end is what matters for crash diagnosis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ}
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["OPENRESEARCH_CELL_OUTPUT_DIR"] = str(output_dir)
    env["OPENRESEARCH_CELL_PARAMS"] = json.dumps(cell)
    env["OPENRESEARCH_CELL_MAX_STEPS"] = "2"
    env["OPENRESEARCH_CELL_TINY_SLICE"] = "1"

    cmd = [sys.executable, str(cell_script), f"--cell-id={cell.get('id', '')}", f"--output-dir={output_dir}"]

    # VRAM sampling (best-effort, fail-soft).
    peak_vram_gib: float | None = None
    baseline_mib: float | None = None
    _stop = None
    _vram_readings: list[float] = []
    try:
        import threading as _threading  # noqa: PLC0415
        from backend.agents.rlm.gpu_cell_runner import _sample_vram_mib, _poll_peak_vram_daemon  # noqa: PLC0415
        baseline_mib = _sample_vram_mib(gpu_id)
        _stop = _threading.Event()
        _t = _threading.Thread(
            target=_poll_peak_vram_daemon,
            args=(gpu_id, 2.0, _stop, _vram_readings),
            daemon=True,
        )
        _t.start()
    except Exception:  # noqa: BLE001
        baseline_mib = None
        _stop = None

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        logger.debug("metric_reality_smoke: subprocess won't spawn: %s", exc)
        if _stop is not None:
            _stop.set()
        return _SmokeCellResult(None, None, False, False, None, "")  # launched=False — spawn-fail fail-open signal

    timed_out = False
    captured_out = ""
    try:
        captured_out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            captured_out, _ = proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            captured_out = ""
    finally:
        if _stop is not None:
            _stop.set()

    # Compute net peak VRAM.
    if _vram_readings and baseline_mib is not None:
        net_peak_mib = max(_vram_readings) - baseline_mib
        peak_vram_gib = net_peak_mib / 1024.0

    # Read the trace: prefer smoke_trace.json, fall back to metrics.json.
    trace: dict | list | None = None
    for fname in ("smoke_trace.json", "metrics.json"):
        p = output_dir / fname
        if p.exists():
            try:
                trace = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:  # noqa: BLE001
                continue

    output_tail = (captured_out or "")[-4000:]
    return _SmokeCellResult(trace, peak_vram_gib, True, timed_out, proc.returncode, output_tail)  # launched=True even when trace is None


# ---------------------------------------------------------------------------
# GPU discovery helper
# ---------------------------------------------------------------------------

def _get_available_gpu_ids() -> list[str]:
    """Return a list of visible GPU ids (str) — same logic as gpu_cell_runner."""
    try:
        from backend.agents.rlm.gpu_cell_runner import discover_visible_gpus  # noqa: PLC0415
        return [str(g) for g in discover_visible_gpus()]
    except Exception:  # noqa: BLE001
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and cvd not in ("", "NoDevFiles"):
            return [g.strip() for g in cvd.split(",") if g.strip()]
        return []


# ---------------------------------------------------------------------------
# Public I/O entrypoint
# ---------------------------------------------------------------------------

def run_metric_reality_smoke(
    *,
    ctx: Any,
    code_dir: Path,
    cells: list[dict] | None,
) -> dict:
    """Run smoke cells; return a SmokeVerdict dict {ok, failure_class, detail}.

    Fail-OPEN (return {ok: True}) only when no GPU is visible or NO cell could be
    launched at all — a harness/infra problem.  A launched cell that wrote no trace
    or a bad trace is JUDGED as smoke_metrics_unreal (fix codex review #1).
    """
    if not metric_reality_smoke_enabled():
        return {"ok": True, "failure_class": None, "detail": ""}

    gpus = _get_available_gpu_ids()
    if not gpus:
        logger.info("metric_reality_smoke: no visible GPUs — fail-open, skipping smoke")
        return {"ok": True, "failure_class": None, "detail": ""}

    # Resolve cells list: try to read from cells.json if not supplied.
    if cells is None:
        cells_path = code_dir / "cells.json"
        if cells_path.is_file():
            try:
                manifest = json.loads(cells_path.read_text(encoding="utf-8"))
                cells = [c for c in (manifest.get("cells") or []) if isinstance(c, dict) and c.get("id")]
            except Exception:  # noqa: BLE001
                cells = []
        else:
            cells = []

    # Resolve cell script.
    cell_script: Path | None = None
    for name in ("train_cell.py", "train.py"):
        p = code_dir / name
        if p.is_file():
            cell_script = p
            break

    if not cells or cell_script is None:
        # No cell-route code present — nothing to smoke, not a failure.
        logger.debug("metric_reality_smoke: no cells.json / cell script — skip smoke")
        return {"ok": True, "failure_class": None, "detail": ""}

    max_cells = _smoke_max_cells()
    timeout = _smoke_timeout_s()
    representatives = _select_smoke_representatives(cells, max_cells)

    any_launched = False
    failures: list[str] = []
    inconclusive: list[str] = []

    with tempfile.TemporaryDirectory(prefix="or_smoke_") as tmpdir:
        tmp = Path(tmpdir)
        for i, cell in enumerate(representatives):
            gpu_id = gpus[i % len(gpus)]
            cell_out = tmp / f"cell_{i}"
            cell_id = cell.get("id", f"rep_{i}")

            trace, peak_vram, launched, timed_out, returncode, output_tail = _run_one_smoke_cell(
                cell, cell_script, gpu_id, cell_out, timeout
            )

            if not launched:
                logger.warning(
                    "metric_reality_smoke: cell %s could not launch (subprocess error) — skip",
                    cell_id,
                )
                continue

            any_launched = True
            records = (
                [r for r in trace if isinstance(r, dict)] if isinstance(trace, list)
                else ([trace] if isinstance(trace, dict) else [])
            )
            if not records:
                if timed_out:
                    # Timed out before producing any step record — INCONCLUSIVE for a
                    # slow-rollout env (e.g. ALFWorld), not broken code. Fail-open for
                    # this cell: the full-grid timeout cap + the in-run zero-metrics guard
                    # remain the backstop. Only a NATURAL exit (below) is judged.
                    inconclusive.append(cell_id)
                    logger.info(
                        "metric_reality_smoke: cell %s timed out before any step record "
                        "— inconclusive (fail-open)", cell_id,
                    )
                else:
                    # Natural exit with no per-step trace — a trainer that completes
                    # without logging is JUDGED (codex Area-4 intent preserved).
                    if returncode not in (0, None):
                        failures.append(
                            f"[{cell_id}] crashed (exit {returncode}) before writing a per-step trace. "
                            f"Fix the error shown below, then write metrics.json/smoke_trace.json incrementally per step.\n"
                            f"--- last output ---\n{output_tail}"
                        )
                    else:
                        failures.append(
                            f"[{cell_id}] exited (rc={returncode}) without writing any per-step trace — "
                            f"write metrics.json/smoke_trace.json incrementally per step"
                        )
                continue

            # >=1 record (natural exit, or a timeout that still produced a partial
            # trace) — judge what we have.
            verdict = evaluate_smoke_trace(trace, peak_vram)
            if not verdict["ok"]:
                failures.append(f"[{cell_id}] {verdict['detail']}")

    # Fail-OPEN only when NO cell launched at all (harness/infra problem).
    if not any_launched:
        logger.info(
            "metric_reality_smoke: no representative cell could launch (%d tried) — fail-open",
            len(representatives),
        )
        return {"ok": True, "failure_class": None, "detail": ""}

    if failures:
        detail = "smoke_metrics_unreal: " + "; ".join(failures[:3])
        logger.warning("metric_reality_smoke: %s", detail)
        return {"ok": False, "failure_class": "smoke_metrics_unreal", "detail": detail}

    logger.info(
        "metric_reality_smoke: smoke passed (%d inconclusive/fail-open) — proceeding",
        len(inconclusive),
    )
    return {"ok": True, "failure_class": None, "detail": ""}
