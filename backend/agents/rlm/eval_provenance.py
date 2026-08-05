"""
Eval-metric provenance guard — verifiable held-out measurement check.

Pure stdlib module — no third-party imports.  Detects run_experiment results
whose result-claiming rate metric (accuracy/success_rate/f1/em/…) cannot be
recomputed from persisted per-example eval records, or whose reported value
does not match the mean of those records.

The motivating case is the SDAR pipeline where ``accuracy = mean_success × 100``
and ``success ≡ reward``, so a true success of ~0.0083 is reported as
``accuracy ≈ 0.83``.  Every existing guard misses this: the value is in [0,1],
non-zero, and uses a real key.

Design:
  - Producer API (``record_eval``): agent-facing, copied into ``code/`` by the
    harness.  Computes mean(outcomes), persists an ``eval_provenance.json``
    sidecar, and RETURNS the metric value for the agent to write into
    ``metrics.json``.  Fail-soft — never raises from the training script.
  - Guard API (``eval_provenance_should_veto``): harness-facing, disk-based.
    Scans cell-level ``metrics.json`` files under the ``outputs/`` subtree,
    checks that each result-claiming rate metric is backed by a verifiable
    ``eval_provenance.json``, and vetoes if the recomputed mean disagrees with
    the reported value (tolerance: 1e-3).  The records-free AGGREGATE schema
    (``verl_metrics_adapter``) is accepted only when the run's resolved
    ``rlm_state/repo_spec.json`` mode is ``"execute"`` — fail-closed for
    adapt-mode / mode-unknown runs, which must persist per-example records.
  - Flag default-OFF: ``OPENRESEARCH_EVAL_PROVENANCE_GUARD``.
  - Fail-soft everywhere: any exception → safe fallback.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATE_METRIC_KEYS: frozenset[str] = frozenset({
    "accuracy",
    "success_rate",
    "success",
    "f1",
    "em",
    "exact_match",
    "precision",
    "recall",
    "pass_rate",
})

SCHEMA_VERSION: int = 1

_RECORDS_SAMPLE_CAP: int = 64   # store at most this many per-example records in the sidecar
_RATE_EPS: float = 1e-4          # tolerance above 1.0 (a value of 1.0 is fine)
_RECOMPUTE_TOL: float = 1e-3     # |reported − recomputed| tolerance


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def eval_provenance_guard_enabled() -> bool:
    """True iff ``OPENRESEARCH_EVAL_PROVENANCE_GUARD`` is in {'1','true','yes','on'}."""
    return os.environ.get(
        "OPENRESEARCH_EVAL_PROVENANCE_GUARD", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def _min_eval_n() -> int:
    """Eval-coverage floor from ``OPENRESEARCH_MIN_EVAL_N`` (0 / unset / invalid = off).

    A sub-knob of the eval-provenance guard: only consulted when that guard is
    enabled (coverage needs the ``eval_provenance.json`` sidecar the guard
    enforces). When > 0, a success cell whose ``n_eval`` is below the floor is
    vetoed — a mean that is correct but computed over too few held-out examples
    is not a substantiated benchmark result.
    """
    try:
        v = int(os.environ.get("OPENRESEARCH_MIN_EVAL_N", "0").strip() or "0")
    except (TypeError, ValueError):
        return 0
    return v if v > 0 else 0


# ---------------------------------------------------------------------------
# Producer API  (agent-facing; copied into code/ by the harness)
# ---------------------------------------------------------------------------

def record_eval(
    output_dir: str | Path,
    *,
    model_key: str,
    env: str,
    baseline: str,
    metric_name: str,
    records: list[dict[str, Any]],
    held_out: bool = True,
    seed: int | None = None,
    train_ids: list | None = None,
) -> float:
    """Compute the eval metric as mean of per-example outcomes, persist a
    verifiable ``<output_dir>/eval_provenance.json`` sidecar, and RETURN the
    metric value for the agent to write into ``metrics.json``.

    Fail-soft — never raises from the training script.

    Args:
        output_dir:   Directory to write ``eval_provenance.json`` into
                      (created if absent).
        model_key:    Model identifier (e.g. ``"qwen3-1.7b"``).
        env:          Environment/dataset name (e.g. ``"alfworld"``).
        baseline:     Baseline name (e.g. ``"grpo"``).
        metric_name:  Metric key (e.g. ``"success_rate"``).
        records:      List of per-example dicts, each with a numeric
                      ``"outcome"`` in [0, 1].  Example::

                          [{"id": "ep_0", "outcome": 1.0,
                            "prediction": "...", "gold": "..."},
                           {"id": "ep_1", "outcome": 0.0, ...}]

        held_out:     Whether evaluation was on a held-out slice (default True).
        seed:         Optional RNG seed stamped into the sidecar.
        train_ids:    Optional list of task ids used during training (e.g. game-file
                      paths for ALFWorld, "source:index" strings for Search-QA).
                      Persisted so the guard can check eval/train disjointness.
                      Omitted from the sidecar when None.

    Returns:
        ``mean(outcome)`` across all valid records (0.0 if records are
        empty or all invalid).
    """
    # --- Validate and coerce records ---
    valid_outcomes: list[float] = []
    try:
        if isinstance(records, (list, tuple)):
            for rec in records:
                try:
                    v = float(rec["outcome"]) if isinstance(rec, dict) else float(rec)
                    if math.isfinite(v):
                        valid_outcomes.append(v)
                except (TypeError, KeyError, ValueError):
                    pass
    except Exception:  # noqa: BLE001 — fail-soft
        valid_outcomes = []

    value: float = sum(valid_outcomes) / len(valid_outcomes) if valid_outcomes else 0.0

    # --- Build sidecar payload ---
    try:
        capped_records: list[dict[str, Any]] = []
        records_truncated = False
        try:
            all_recs = list(records) if isinstance(records, (list, tuple)) else []
            # Keep the OUTCOME of EVERY record so the guard can recompute the exact
            # mean at any N (floats are tiny); cap only the heavy prediction/gold text
            # to the first _RECORDS_SAMPLE_CAP records to bound the sidecar size.
            for i, rec in enumerate(all_recs):
                if isinstance(rec, dict):
                    slim: dict[str, Any] = {}
                    if "id" in rec:
                        slim["id"] = rec["id"]
                    if "outcome" in rec:
                        slim["outcome"] = rec["outcome"]
                    if i < _RECORDS_SAMPLE_CAP:
                        for field in ("prediction", "gold"):
                            if field in rec:
                                slim[field] = rec[field]
                    else:
                        records_truncated = True  # heavy text dropped for this record
                    capped_records.append(slim)
                else:
                    # non-dict record — store the raw value (it IS the outcome)
                    capped_records.append(rec)
        except Exception:  # noqa: BLE001
            capped_records = []
            records_truncated = False

        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_key": model_key,
            "env": env,
            "baseline": baseline,
            "metric_name": metric_name,
            "metric_value": value,
            "n_eval": len(valid_outcomes),
            "held_out": held_out,
            "seed": seed,
            "records": capped_records,
            "records_truncated": records_truncated,
        }

        # Persist train_ids for the disjointness check (omit key when not supplied).
        try:
            if train_ids is not None:
                all_train_ids = list(train_ids)
                payload["train_ids"] = [str(tid) for tid in all_train_ids[:_RECORDS_SAMPLE_CAP]]
                payload["n_train"] = len(all_train_ids)
        except Exception:  # noqa: BLE001 — fail-soft
            pass

        out_dir = Path(output_dir)
        target = out_dir / "eval_provenance.json"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001 — fail-soft: return the computed value regardless
            pass
    except Exception:  # noqa: BLE001 — fail-soft
        pass

    return value


# ---------------------------------------------------------------------------
# Guard API  (harness-facing; disk-based)
# ---------------------------------------------------------------------------

def _resolved_repro_mode(code_dir: Path) -> str | None:
    """Resolved reproduction mode from ``rlm_state/repo_spec.json`` (the
    sibling of ``code_dir`` under ``runs/<project_id>/``), or None when the
    file is absent/unreadable/mode-less.

    Consumers treat None as NOT-execute (fail-closed): the aggregate
    provenance schema below is only legal for execute-mode runs (the
    verl_metrics_adapter only exists to bridge the authors' own pipeline),
    and an unknown mode must never widen acceptance — an adapt-mode cell
    must persist per-example records.
    """
    try:
        spec_path = Path(code_dir).parent / "rlm_state" / "repo_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        mode = spec.get("mode") if isinstance(spec, dict) else None
        return mode if isinstance(mode, str) and mode.strip() else None
    except Exception:  # noqa: BLE001 — fail-soft to "unknown" (treated as not-execute)
        return None


def _find_rate_metric(m: dict[str, Any]) -> tuple[str, float] | None:
    """Return (key, value) for the first top-level rate-metric key with a
    finite positive numeric value, or None if none found."""
    for k, v in m.items():
        if k.lower() not in RATE_METRIC_KEYS:
            continue
        # A boolean (e.g. "success": true used as a per-cell status flag) is NOT a
        # rate metric — float(True) == 1.0 would otherwise demand a sidecar. Skip it.
        if isinstance(v, bool):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv) and fv > 0.0:
            return (k, fv)
    return None


def eval_provenance_should_veto(code_dir: str | Path) -> tuple[bool, str | None]:
    """Disk-based veto.  Returns ``(True, message)`` iff the guard is ENABLED
    and at least one result-claiming cell fails verification.

    Pure / fail-soft: any error → ``(False, None)``.

    The guard processes only ``metrics.json`` files that are:
      * inside an ``outputs/`` subtree (i.e. ``"outputs"`` in their path parts), AND
      * contain a ``"status"`` key (marking them as cell-level results).

    A cell is checked only when its status is a success-equivalent string.  A
    cell with no rate-metric key (e.g. RL-reward-only) is conservatively exempt.
    """
    if not eval_provenance_guard_enabled():
        return (False, None)

    try:
        code_dir = Path(code_dir)
        if not code_dir.is_dir():
            return (False, None)
    except Exception:  # noqa: BLE001
        return (False, None)

    _SUCCESS_STATUSES = frozenset({"ok", "success", "succeeded", "completed"})
    violations: list[str] = []

    # Resolved once per call: the aggregate-provenance branch below is gated on
    # the run's resolved reproduction mode (execute-only; None = fail-closed).
    repro_mode = _resolved_repro_mode(code_dir)

    try:
        for metrics_path in code_dir.rglob("metrics.json"):
            try:
                # Only cell-level results (inside an outputs/ subtree with a status key)
                if "outputs" not in metrics_path.parts:
                    continue

                # Load metrics
                try:
                    m = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue

                if not isinstance(m, dict):
                    continue

                # Only process success-status cells
                status = str(m.get("status", "")).lower()
                if status not in _SUCCESS_STATUSES:
                    continue

                # Find the first result-claiming rate metric with a positive value
                pair = _find_rate_metric(m)
                if pair is None:
                    # No rate metric key — conservatively exempt (RL-reward-only cells)
                    continue

                rate_key, reported = pair
                cell_label = str(metrics_path.parent)

                # Provenance may live in a standalone sidecar (local/docker) OR
                # INLINE inside metrics.json under "eval_provenance" (the GKE
                # path — the pod entrypoint uploads only metrics.json, so a
                # standalone sidecar never round-trips back to the host).
                sc: Any = None
                prov_path = metrics_path.parent / "eval_provenance.json"
                if prov_path.exists():
                    try:
                        sc = json.loads(prov_path.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        # Sidecar unreadable — treat as absent
                        violations.append(
                            f"{cell_label}: eval_provenance.json is unreadable"
                        )
                        if len(violations) >= 3:
                            break
                        continue
                elif isinstance(m.get("eval_provenance"), dict):
                    sc = m["eval_provenance"]

                if sc is None:
                    violations.append(
                        f"{cell_label}: claims {rate_key}={reported:.4f} but has no"
                        f" eval_provenance (neither an eval_provenance.json sidecar nor"
                        f" an inline metrics.json['eval_provenance'] — the metric is not"
                        f" verifiable as a real held-out measurement)"
                    )
                    if len(violations) >= 3:
                        break
                    continue

                if not isinstance(sc, dict):
                    violations.append(
                        f"{cell_label}: eval_provenance has unexpected format"
                    )
                    if len(violations) >= 3:
                        break
                    continue

                # Eval-coverage floor (OPENRESEARCH_MIN_EVAL_N, default 0=off): the
                # checks below verify the reported metric IS the mean of real
                # outcomes; this verifies ENOUGH held-out examples backed it (a
                # correct mean over 3 examples is not a benchmark result).
                # Conservative: skips when the sidecar carries no numeric n_eval.
                floor = _min_eval_n()
                if floor > 0:
                    sc_n = sc.get("n_eval")
                    if (
                        isinstance(sc_n, (int, float))
                        and not isinstance(sc_n, bool)
                        and sc_n < floor
                    ):
                        violations.append(
                            f"{cell_label}: n_eval={int(sc_n)} <"
                            f" OPENRESEARCH_MIN_EVAL_N={floor} — too few held-out examples"
                            f" to substantiate {rate_key}={reported:.4f}"
                        )
                        if len(violations) >= 3:
                            break
                        continue

                # Aggregate-provenance schema (execute-mode verl adapter): a
                # value-preservingly copied aggregate metric with NO per-example
                # records (verl exposes only an aggregate — synthesizing fake
                # per-example records would violate the no-fabrication red
                # line). Verified by a value-preserving cross-check against
                # metrics.json instead of a records-recompute.
                #
                # EXECUTE-MODE-ONLY (fail-closed): the aggregate schema is
                # accepted ONLY when the run's resolved
                # ``rlm_state/repo_spec.json`` mode is "execute" — the one mode
                # where the authors' own pipeline is the metric producer and
                # per-example records genuinely do not exist. An adapt-mode (or
                # mode-unknown) cell carrying an aggregate marker is vetoed:
                # adapt-mode trainers are harness-guided and MUST persist
                # per-example records via record_eval.
                if (
                    sc.get("provenance_kind") == "aggregate"
                    or sc.get("adapter") == "verl_metrics_adapter"
                ):
                    if repro_mode != "execute":
                        violations.append(
                            f"{cell_label}: aggregate eval_provenance is only"
                            f" accepted for execute-mode runs"
                            f" (rlm_state/repo_spec.json mode="
                            f"{repro_mode or 'unknown'}) — adapt-mode cells must"
                            f" persist per-example records via record_eval"
                        )
                        if len(violations) >= 3:
                            break
                        continue

                    try:
                        mv = float(sc.get("metric_value"))
                        if not math.isfinite(mv):
                            raise ValueError("non-finite metric_value")
                    except (TypeError, ValueError):
                        violations.append(
                            f"{cell_label}: aggregate eval_provenance.json has no"
                            f" numeric metric_value"
                        )
                        if len(violations) >= 3:
                            break
                        continue

                    if mv < 0.0 or mv > 1.0 + _RATE_EPS:
                        violations.append(
                            f"{cell_label}: aggregate metric_value out of [0,1]"
                            f" (likely ×100-scaled)"
                        )
                        if len(violations) >= 3:
                            break
                        continue

                    if abs(reported - mv) > _RECOMPUTE_TOL:
                        violations.append(
                            f"{cell_label}: metrics.json disagrees with aggregate"
                            f" eval_provenance.metric_value"
                            f" ({rate_key}={reported:.4f} vs sidecar={mv:.4f})"
                        )
                        if len(violations) >= 3:
                            break
                        continue

                    # Honesty binding (closes the self-attestation loophole): the
                    # value cross-checks above only prove INTERNAL consistency
                    # (reported == metric_value, in [0,1]). To be credited as a real
                    # held-out measurement the aggregate must ALSO be bound to
                    # something the cell did not simply assert — EITHER:
                    #   (a) a ``source`` file that resolves on disk (the local/docker
                    #       path, where the value was read from a real artifact), OR
                    #   (b) the trusted harness producer marker
                    #       ``adapter == "verl_metrics_adapter"`` — the value-preserving
                    #       shim (harness-owned code copied into the cell) whose GKE
                    #       source path is a POD-side path that legitimately does not
                    #       round-trip to the host (the ONLY artifact the pod uploads
                    #       is metrics.json).
                    # A bare aggregate with NEITHER is unverifiable self-attestation
                    # → veto. (Source resolution is tried both absolute and relative
                    # to the metrics dir, to survive a relocated run tree.)
                    src = sc.get("source")
                    src_resolves = False
                    if isinstance(src, str) and src.strip():
                        try:
                            src_resolves = (
                                Path(src).exists()
                                or (metrics_path.parent / Path(src).name).exists()
                            )
                        except Exception:  # noqa: BLE001 — fail-soft to "does not resolve"
                            src_resolves = False
                    produced_by_trusted_adapter = sc.get("adapter") == "verl_metrics_adapter"
                    if not (src_resolves or produced_by_trusted_adapter):
                        violations.append(
                            f"{cell_label}: aggregate {rate_key}={reported:.4f} is"
                            f" unverifiable — no resolvable source file and not stamped"
                            f" by the trusted verl_metrics_adapter (self-attestation"
                            f" alone is not a held-out measurement)"
                        )
                        if len(violations) >= 3:
                            break
                        continue

                    # All checks pass — verifiably the authors' own reported number.
                    continue

                records = sc.get("records")
                if not isinstance(records, list) or len(records) == 0:
                    violations.append(
                        f"{cell_label}: empty eval records in eval_provenance.json"
                    )
                    if len(violations) >= 3:
                        break
                    continue

                # Validate per-example outcomes
                outcomes: list[float] = []
                out_of_range = False
                for rec in records:
                    try:
                        if isinstance(rec, dict):
                            ov = float(rec["outcome"])
                        else:
                            ov = float(rec)
                        if not math.isfinite(ov):
                            out_of_range = True
                            break
                        if ov < 0.0 or ov > 1.0 + _RATE_EPS:
                            out_of_range = True
                            break
                        outcomes.append(ov)
                    except (TypeError, KeyError, ValueError):
                        pass  # skip uncoercible records

                if out_of_range:
                    violations.append(
                        f"{cell_label}: per-example outcome out of [0,1] in"
                        f" eval_provenance.json (outcomes must be fractions, not"
                        f" counts or ×100-scaled values)"
                    )
                    if len(violations) >= 3:
                        break
                    continue

                if not outcomes:
                    violations.append(
                        f"{cell_label}: no valid numeric outcomes in eval_provenance.json"
                    )
                    if len(violations) >= 3:
                        break
                    continue

                recomputed = sum(outcomes) / len(outcomes)

                # Core check: reported value must match mean(outcomes)
                if abs(reported - recomputed) > _RECOMPUTE_TOL:
                    violations.append(
                        f"{cell_label}: reported {rate_key}={reported:.4f} !="
                        f" mean(eval records)={recomputed:.4f}"
                        f" (difference {abs(reported - recomputed):.4f} > tol {_RECOMPUTE_TOL};"
                        f" likely reward×100 or other non-held-out scaling)"
                    )
                    if len(violations) >= 3:
                        break
                    continue

                # Cross-check with sidecar's own metric_value field (if present)
                sc_metric_value = sc.get("metric_value")
                if sc_metric_value is not None:
                    try:
                        sc_mv = float(sc_metric_value)
                        if abs(reported - sc_mv) > _RECOMPUTE_TOL:
                            violations.append(
                                f"{cell_label}: metrics.json disagrees with"
                                f" eval_provenance.metric_value"
                                f" ({rate_key}={reported:.4f} vs sidecar={sc_mv:.4f})"
                            )
                            if len(violations) >= 3:
                                break
                    except (TypeError, ValueError):
                        pass  # non-numeric metric_value — skip cross-check

                # Disjointness check: eval set must not overlap the training set.
                # Conservative: only fires when BOTH train_ids (non-empty) AND at
                # least one record carries an "id" field are present — missing
                # either side is not a violation (old runs without ids are safe).
                try:
                    sc_train_ids = sc.get("train_ids")
                    if isinstance(sc_train_ids, list) and sc_train_ids:
                        eval_ids = {
                            str(rec["id"])
                            for rec in records
                            if isinstance(rec, dict) and "id" in rec
                        }
                        if eval_ids:
                            train_id_set = {str(tid) for tid in sc_train_ids}
                            overlap = eval_ids & train_id_set
                            if overlap:
                                violations.append(
                                    f"{cell_label}: eval set overlaps the training set"
                                    f" ({len(overlap)} shared task id(s)) — the metric"
                                    f" is in-sample, not held-out"
                                )
                                if len(violations) >= 3:
                                    break
                except Exception:  # noqa: BLE001 — fail-soft
                    pass

            except Exception:  # noqa: BLE001 — per-cell fail-soft
                continue

    except Exception:  # noqa: BLE001 — outer scan fail-soft
        return (False, None)

    if violations:
        return (True, "; ".join(violations))
    return (False, None)


# ---------------------------------------------------------------------------
# Repair message
# ---------------------------------------------------------------------------

def eval_provenance_repair_message(detail: str) -> str:
    """Actionable repair directive naming ``detail`` and instructing the agent
    to use ``record_eval`` from ``eval_provenance`` to produce a verifiable
    held-out metric.  Mirrors the tone and shape of ``zero_metrics_repair_message``.
    """
    try:
        detail_str = str(detail).strip() if detail else "(no detail)"
        return (
            f"fabrication_suspected: eval-metric provenance check failed — {detail_str}. "
            f"Re-implement: compute each cell's eval metric on a HELD-OUT slice vs gold "
            f"using real model outputs. Call "
            f"`record_eval(output_dir, model_key=..., env=..., baseline=..., "
            f"metric_name='success_rate', records=["
            f"{{'id':..., 'outcome':0/1 or f1, 'prediction':..., 'gold':...}}, ...])` "
            f"from `eval_provenance`, and write its returned fraction (in [0,1]) to "
            f"metrics.json. "
            f"Do NOT scale by 100/128. Do NOT set success ≡ reward. "
            f"Each per-example outcome must be a real measurement in [0,1] "
            f"(e.g. 1.0 if the model's answer matches gold, 0.0 otherwise; "
            f"token-F1 for extractive QA; exact-match for MATH)."
        )
    except Exception:  # noqa: BLE001
        return (
            "fabrication_suspected: eval-metric provenance check failed. "
            "Re-implement: compute eval metrics on a held-out slice, call record_eval "
            "from eval_provenance, and write the returned fraction (in [0,1]) to metrics.json."
        )
