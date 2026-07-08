"""verl → metrics adapter — value-preserving execute-mode metrics seam.

Copyable helper — mirror of the ``rubric_guard.py``/``eval_provenance.py``
pattern (see ``_HARNESS_CODE_HELPERS`` in ``baseline_implementation.py``).
Zero non-stdlib dependencies so the copy-and-paste route into ``code/``
always works inside an agent sandbox.

Motivating case: execute mode (``OPENRESEARCH_REPRODUCTION_MODE=execute``)
seeds the authors' own repository into ``code/`` and runs their pipeline
VERBATIM behind the command/launcher seam (``gpu_cell_runner.py``'s
``cell["command"]``). The authors' trainer (e.g. verl) writes its OWN log
format — not the harness's flat ``metrics.json`` contract — so the cell
finishes with no ``metrics.json`` for ``aggregate_cell_metrics``/the leaf
scorer to read. This module bridges that gap WITHOUT ever scaling,
recomputing, or otherwise altering the measured value: it locates the
authors' own reported number and copies it verbatim.

Two source strategies, tried in order:

  1. A machine-readable val/summary JSON file directly under ``output_dir``
     (glob ``*val*.json`` / ``*summary*.json``) — flat or nested, looked up
     by ``success_rate_key`` (a ``/``-joined path, e.g. ``"val/success_rate"``
     matches either a literal top-level key of that name OR the nested
     ``{"val": {"success_rate": ...}}`` shape).
  2. A regex scan across the files matched by ``log_glob`` (``$OUTPUT_DIR`` is
     expanded to ``output_dir``), matching the key as a WHOLE TOKEN not
     followed by ``/`` — so a per-dataset sub-key
     (``val/success_rate/nq:0.418``) is never confused with the aggregate key
     (``val/success_rate:0.456``) — and taking the LAST occurrence across all
     matched files (verl logs the running val curve; the last line is the
     final/most-converged value).

Fail-honest: when no value is found, ``metrics.json`` is written as
``{"status": "failed"}`` with NO ``success_rate`` key — never a fabricated
0.0 (mirrors the zero-metrics-guard red line: an honest "couldn't measure it"
beats a silent fake number).
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JSON_GLOBS: tuple[str, ...] = ("*val*.json", "*summary*.json")
_NUMBER_RE = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


# ---------------------------------------------------------------------------
# JSON strategy
# ---------------------------------------------------------------------------

def _flatten_json(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a nested dict into ``{"a/b/c": value}`` numeric leaves.

    A top-level key that ALREADY contains literal ``/`` characters (e.g. a
    verl summary dumped as ``{"val/success_rate": 0.456}``) is preserved
    as-is (``prefix`` is empty so the key passes through unchanged) — so both
    the flat-literal-key shape and the genuinely-nested shape resolve through
    the same dotted-path lookup.  Booleans are excluded (``bool`` is an
    ``int`` subclass in Python and would otherwise silently coerce to 0.0/1.0).
    """
    flat: dict[str, float] = {}
    if not isinstance(obj, dict):
        return flat
    for key, val in obj.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(val, dict):
            flat.update(_flatten_json(val, path))
        elif isinstance(val, bool):
            continue
        elif isinstance(val, (int, float)):
            flat[path] = float(val)
    return flat


def _find_json_value(output_dir: Path, key: str) -> tuple[float, str, str] | None:
    """Return ``(value, source_path, source_line)`` from a val/summary JSON, or None."""
    seen: set[str] = set()
    for pattern in _JSON_GLOBS:
        for path in sorted(output_dir.glob(pattern)):
            key_str = str(path)
            if key_str in seen:
                continue
            seen.add(key_str)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — unreadable/non-JSON file, try the next
                continue
            flat = _flatten_json(data)
            if key in flat:
                value = flat[key]
                return value, key_str, json.dumps({key: value})
    return None


# ---------------------------------------------------------------------------
# Log-regex strategy
# ---------------------------------------------------------------------------

def _expand_log_glob(output_dir: Path, log_glob: str) -> str:
    return log_glob.replace("$OUTPUT_DIR", str(output_dir))


def _build_prefix_pattern(prefix: str) -> re.Pattern:
    """Match a per-data_source family key ``<prefix><suffix>`` and capture both
    the full key and its value.

    Used when an eval key ends in ``/`` (e.g. ``val/test_score/``): verl's
    reward-manager validation logs one key PER data_source
    (``val/test_score/math``, ``val/test_score/openai/gsm8k``) and emits NO
    bare aggregate, so the exact-key match finds nothing. The suffix is
    everything up to the first whitespace / quote / ``=`` (a data_source may
    itself contain ``/``), and the trailing value tolerates the same
    numpy/torch scalar wrappers as the exact-key pattern. Callers require the
    matches to collapse to exactly ONE distinct full key before trusting a
    value (never averaging across sources — that would not be value-preserving).
    """
    prefix_esc = re.escape(prefix)
    return re.compile(
        rf"(?<![\w/])({prefix_esc}[^\s'\"=]+?)['\"]?\s*[:=]\s*"
        rf"(?:np\.float64\(|np\.float32\(|torch\.tensor\(|tensor\()?"
        rf"({_NUMBER_RE})"
    )


def _build_key_pattern(key: str) -> re.Pattern:
    """Whole-token key match: not preceded/followed by a word char or ``/``.

    This is what keeps ``val/success_rate/nq:0.418`` (a per-dataset sub-key)
    from matching the aggregate key ``val/success_rate`` — the char right
    after the literal key text is ``/``, which the trailing negative
    lookahead rejects. Symmetrically, a preceding ``/`` or word char (e.g. an
    outer-nested key ``outer/val/success_rate`` or the per-dataset
    ``val/musique_success_rate``) is also rejected, since that means the true
    key is longer than the one requested.

    verl logs its val metrics as a Python dict repr on the console, e.g.
    ``'val/success_rate': np.float64(0.4562844669117647)`` — so after the key
    the pattern tolerates an optional closing quote, a ``:``/``=`` separator,
    and a numpy/torch scalar wrapper (``np.float64(`` / ``tensor(`` …) before
    the number. The bare JSON/plain forms (``"val/success_rate": 0.456``,
    ``val/success_rate=0.456``) still match (the wrapper group is optional).
    """
    key_esc = re.escape(key)
    return re.compile(
        rf"(?<![\w/]){key_esc}(?![\w/])['\"]?\s*[:=]\s*"
        rf"(?:np\.float64\(|np\.float32\(|torch\.tensor\(|tensor\()?"
        rf"({_NUMBER_RE})"
    )


def _find_in_logs(output_dir: Path, log_glob: str, key: str) -> tuple[float, str, str] | None:
    """Return ``(value, source_path, source_line)`` for the LAST matching
    occurrence of ``key`` across every file matched by ``log_glob``, or None."""
    expanded = _expand_log_glob(output_dir, log_glob)
    paths = sorted(glob.glob(expanded))
    pattern = _build_key_pattern(key)

    last: tuple[float, str, str] | None = None
    for path_str in paths:
        try:
            text = Path(path_str).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()
            last = (value, path_str, line)  # keep overwriting → last wins
    return last


def _find_prefix_in_json(output_dir: Path, prefix: str) -> tuple[float, str, str] | None:
    """Single-distinct-key prefix lookup across ALL val/summary JSONs. Returns a
    value only when exactly ONE full key starts with ``prefix`` across every file
    AND that key carries a single consistent value everywhere it appears.

    Unlike a log curve, JSON files have no temporal order, so "last wins" is
    undefined: if the same key appears in two files with DIFFERENT values (or two
    distinct keys match), the result is ambiguous → None (never guess which is
    the final/converged value — that would not be value-preserving)."""
    found: dict[str, tuple[float, str, str]] = {}
    conflicting: set[str] = set()
    for pattern in _JSON_GLOBS:
        for path in sorted(output_dir.glob(pattern)):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            flat = _flatten_json(data)
            for k, v in flat.items():
                if not (k.startswith(prefix) and k != prefix):
                    continue
                if k in found and found[k][0] != v:
                    conflicting.add(k)
                found.setdefault(k, (v, str(path), json.dumps({k: v})))
    distinct = [k for k in found if k not in conflicting]
    if len(distinct) == 1 and not conflicting:
        return found[distinct[0]]
    return None


def _find_prefix_in_logs(output_dir: Path, log_glob: str, prefix: str) -> tuple[float, str, str] | None:
    """Single-distinct-key prefix lookup across logs. Scans every ``prefix``
    family key, keeps each key's LAST value (verl logs a running val curve),
    and returns it only when the family collapses to exactly ONE distinct full
    key — never averaging across data_sources (not value-preserving)."""
    expanded = _expand_log_glob(output_dir, log_glob)
    pattern = _build_prefix_pattern(prefix)
    last_by_key: dict[str, tuple[float, str, str]] = {}
    for path_str in sorted(glob.glob(expanded)):
        try:
            text = Path(path_str).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            full_key = match.group(1)
            try:
                value = float(match.group(2))
            except (TypeError, ValueError):
                continue
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()
            last_by_key[full_key] = (value, path_str, line)  # keep overwriting → last wins
    if len(last_by_key) == 1:
        return next(iter(last_by_key.values()))
    return None


def _find_value(output_dir: Path, log_glob: str, key: str) -> tuple[float, str, str] | None:
    """JSON-first, log-regex-fallback lookup for a single key.

    A ``key`` ending in ``/`` is a per-data_source PREFIX (e.g.
    ``val/test_score/``) resolved to the single distinct concrete sub-key when
    one exists; a normal key uses the exact whole-token match.
    """
    if key.endswith("/"):
        found = _find_prefix_in_json(output_dir, key)
        if found is not None:
            return found
        return _find_prefix_in_logs(output_dir, log_glob, key)
    found = _find_json_value(output_dir, key)
    if found is not None:
        return found
    return _find_in_logs(output_dir, log_glob, key)


# ---------------------------------------------------------------------------
# Public producer
# ---------------------------------------------------------------------------

def write_cell_metrics_from_verl(
    output_dir: str | Path,
    *,
    model_key: str | None = None,
    env: str | None = None,
    baseline: str | None = None,
    log_glob: str = "$OUTPUT_DIR/*.log",
    success_rate_key: str = "val/success_rate",
    metric_name: str = "success_rate",
    extra_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Adapt a verl (or verl-shaped) cell's own output into the canonical
    ``metrics.json`` leaf shape, WITHOUT scaling/recomputing the measured
    value, recording exactly where the value came from.

    The measured value is written under ``metric_name`` — ``"success_rate"``
    (default) for a genuinely HELD-OUT validation key (a result-claiming rate
    metric the eval-provenance guard checks), or e.g. ``"reward_mean"`` for a
    TRAINING reward (NOT a held-out measurement — the caller labels it honestly
    so it is never mistaken for a success rate).

    Provenance is written BOTH inline in ``metrics.json`` (under
    ``"eval_provenance"`` — the only artifact guaranteed to survive the GKE
    pod→GCS→host round-trip, since the pod entrypoint uploads only
    ``metrics.json``) AND as a standalone ``eval_provenance.json`` sidecar (for
    the local/docker path). The guard reads whichever is present.

    Returns the written ``metrics.json`` dict (also useful for a caller that
    wants the value without re-reading the file).

    Fail-honest: if no value is found, returns/writes ``{"status": "failed"}``
    with NO metric key — never a fabricated 0.0.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = _find_value(out_dir, log_glob, success_rate_key)
    if found is None:
        metrics: dict[str, Any] = {"status": "failed"}
        _write_json(out_dir / "metrics.json", metrics)
        return metrics

    value, source_path, source_line = found

    metrics = {metric_name: value, "status": "success"}

    extra_provenance: dict[str, dict[str, str]] = {}
    for extra_key in extra_keys or ():
        extra_found = _find_value(out_dir, log_glob, extra_key)
        if extra_found is not None:
            extra_value, extra_source, extra_line = extra_found
            metrics[extra_key] = extra_value
            extra_provenance[extra_key] = {"source": extra_source, "source_line": extra_line}

    provenance = {
        "schema_version": 1,
        "adapter": "verl_metrics_adapter",
        "provenance_kind": "aggregate",
        "model_key": model_key,
        "env": env,
        "baseline": baseline,
        "metric_name": metric_name,
        "success_rate_key": success_rate_key,
        "metric_value": value,
        "source": source_path,
        "source_line": source_line,
        "extra_keys": extra_provenance,
    }
    # Inline copy travels inside metrics.json (survives the pod→host round-trip);
    # the standalone sidecar covers the local/docker path where it survives too.
    metrics["eval_provenance"] = provenance
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "eval_provenance.json", provenance)

    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Best-effort JSON write — never raises (mirrors eval_provenance.py's fail-soft I/O)."""
    try:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass
