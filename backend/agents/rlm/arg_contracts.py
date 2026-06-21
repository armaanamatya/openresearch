"""Generic argument pre-validation guard for RLM primitives.

Blocks a primitive call when an argument carries a placeholder/sentinel value
(e.g. 'unknown', 'tbd', 'n/a') in a field that must be a real token, returning
a crisp, repairable error dict instead of letting the placeholder propagate.

Flag-gated: set OPENRESEARCH_ARG_CONTRACTS=1 to enable. Default OFF — with the
flag unset this module is a no-op (all public functions return None).

Pure stdlib-only module (inspect + os). No external dependencies.
"""

from __future__ import annotations

import inspect
import os


# ---------------------------------------------------------------------------
# Flag reader
# ---------------------------------------------------------------------------

def arg_contracts_enabled() -> bool:
    return os.environ.get("OPENRESEARCH_ARG_CONTRACTS", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


# ---------------------------------------------------------------------------
# Sentinel set
# Case-insensitive; compared after .strip().lower().
# Empty/whitespace string also counts as a violation.
# Deliberately EXCLUDES "none"/"null" — ambiguous, could be legit.
# ---------------------------------------------------------------------------

_SENTINELS: frozenset[str] = frozenset({
    "unknown", "tbd", "n/a", "na", "placeholder",
    "todo", "fixme", "xxx", "tba",
})


# ---------------------------------------------------------------------------
# Declarative table: primitive name -> parameter names whose string-leaf
# values must be scanned for sentinels.
# Extension point: add entries here to cover new primitives.
# ---------------------------------------------------------------------------

PRIMITIVE_ARG_CONTRACTS: dict[str, tuple[str, ...]] = {
    "plan_reproduction": ("method_spec", "paper_claim_map"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_sentinel(value: str) -> bool:
    """Return True if *value* is empty/whitespace or a known sentinel token."""
    stripped = value.strip()
    if not stripped:
        return True
    return stripped.lower() in _SENTINELS


def _scan_value(value: object, path: str, violations: list[tuple[str, str]]) -> None:
    """Recursively scan *value* for string leaves that are sentinels.

    Records (path, leaf_value) pairs into *violations* (up to 5 total).
    Handles: str, dict, list, tuple. Non-str scalars are ignored.
    """
    if len(violations) >= 5:
        return
    if isinstance(value, str):
        if _is_sentinel(value):
            violations.append((path, value))
    elif isinstance(value, dict):
        for k, v in value.items():
            if len(violations) >= 5:
                break
            _scan_value(v, f"{path}.{k}", violations)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            if len(violations) >= 5:
                break
            _scan_value(item, f"{path}[{i}]", violations)
    # Non-str scalars (int, float, bool, None, etc.) are intentionally ignored.


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------

def validate_primitive_args(
    name: str,
    fn: object,
    args: tuple,
    kwargs: dict,
) -> dict | None:
    """Return a repairable guard dict if a declared arg contains a placeholder/sentinel
    value, else None.

    Fail-soft: any internal error returns None.
    No-op unless the flag is on and *name* is in PRIMITIVE_ARG_CONTRACTS.
    """
    try:
        if not arg_contracts_enabled():
            return None
        if name not in PRIMITIVE_ARG_CONTRACTS:
            return None

        declared_params = PRIMITIVE_ARG_CONTRACTS[name]

        # Bind positional + keyword args to parameter names.
        # Use bind_partial so missing/extra params don't raise.
        try:
            sig = inspect.signature(fn)  # type: ignore[arg-type]
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            bound_args: dict[str, object] = bound.arguments
        except Exception:  # noqa: BLE001
            return None

        violations: list[tuple[str, str]] = []
        for param_name in declared_params:
            if param_name not in bound_args:
                continue
            _scan_value(bound_args[param_name], param_name, violations)
            if len(violations) >= 5:
                break

        if not violations:
            return None

        # Build the paths description for the top-level error string.
        paths_desc = ", ".join(
            f"{path}={val!r}" for path, val in violations
        )
        n = len(violations)

        contract_violations = [
            {
                "area": "Argument grounding",
                "detail": f"{path} was the placeholder {val!r}",
                "hint": (
                    "Extract the REAL value from the paper (understand_section/"
                    "extract_hyperparameters) and pass it verbatim; never "
                    "'unknown', 'tbd', or an empty string as a placeholder."
                ),
            }
            for path, val in violations
        ]

        return {
            "success": False,
            "failure_class": "arg_contract",
            "source": "arg_guard",
            "error": (
                f"{name}: {n} argument(s) contain placeholder values that must be "
                f"real tokens from the paper: [{paths_desc}]"
            ),
            "contract_violations": contract_violations,
        }

    except Exception:  # noqa: BLE001
        # Fail-soft: guard must never raise.
        return None
