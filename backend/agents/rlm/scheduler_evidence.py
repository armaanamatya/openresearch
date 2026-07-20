"""Fail-closed authority evidence for branch/rung scheduler transitions.

This module is deliberately independent of the grade/report path. It turns a
*harness-ledger-attested* cell receipt into an authority input only when the
receipt is bound to an immutable paper-step ladder, a coherent canonical
evidence bundle, a measured metric artifact, and a resumable checkpoint state.
It must never fall back to ``final_report.score`` or campaign ``scope_rung``.

The cell runners own writing receipts; this reader is pure except for verifying
the named local artifacts.  Any missing, malformed, or mismatched value makes
the result ``None`` so callers preserve the base campaign decision.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


SCHEMA_VERSION = 1
_DIRECTIONS = frozenset({"maximize", "minimize"})
_SHA256_LENGTH = 64
_REQUIRED_CHECKPOINT_STATE = frozenset(
    {"model_sha256", "optimizer_sha256", "lr_scheduler_sha256", "rng_sha256", "data_order_sha256"}
)


@dataclass(frozen=True)
class PaperStepLadder:
    """Immutable, operator/paper-pinned fidelity schedule for one campaign."""

    paper_ref: str
    metric_id: str
    direction: Literal["maximize", "minimize"]
    r_max_steps: int
    rung_steps: tuple[int, ...]
    schedule_source_sha256: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported scheduler evidence schema")
        if not self.paper_ref or not self.metric_id or self.direction not in _DIRECTIONS:
            raise ValueError("ladder needs paper_ref, metric_id, and known direction")
        if type(self.r_max_steps) is not int or self.r_max_steps <= 0:
            raise ValueError("r_max_steps must be a positive integer")
        if not _is_sha256(self.schedule_source_sha256):
            raise ValueError("schedule_source_sha256 must be a SHA-256 digest")
        if not self.rung_steps or self.rung_steps[-1] != self.r_max_steps:
            raise ValueError("rung_steps must end at r_max_steps")
        previous = 0
        for step in self.rung_steps:
            if type(step) is not int or step <= previous or step > self.r_max_steps:
                raise ValueError("rung_steps must be strictly increasing positive integers")
            previous = step

    @property
    def sha256(self) -> str:
        return _sha256_json(asdict(self))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PaperStepLadder":
        return cls(
            paper_ref=_nonempty_string(raw.get("paper_ref"), "paper_ref"),
            metric_id=_nonempty_string(raw.get("metric_id"), "metric_id"),
            direction=_direction(raw.get("direction")),
            r_max_steps=_positive_int(raw.get("r_max_steps"), "r_max_steps"),
            rung_steps=tuple(_step_list(raw.get("rung_steps"))),
            schedule_source_sha256=_sha(raw.get("schedule_source_sha256"), "schedule_source_sha256"),
            schema_version=_positive_int(raw.get("schema_version", SCHEMA_VERSION), "schema_version"),
        )


@dataclass(frozen=True)
class BranchRungReceipt:
    """One verified branch continuation boundary, suitable for authority only."""

    campaign_id: str
    branch_id: str
    parent_branch_id: str | None
    attempt_n: int
    cell_id: str
    paper_ref: str
    ladder_sha256: str
    from_step: int
    to_step: int
    metric_id: str
    direction: Literal["maximize", "minimize"]
    metric_value: float
    metric_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    evidence_bundle_sha256: str
    code_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    run_spec_sha256: str
    seed: int
    termination_cause: str | None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported scheduler evidence schema")
        for field in ("campaign_id", "branch_id", "cell_id", "paper_ref", "metric_id", "checkpoint_path"):
            if not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        if type(self.attempt_n) is not int or self.attempt_n < 1:
            raise ValueError("attempt_n must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.direction not in _DIRECTIONS or not math.isfinite(self.metric_value):
            raise ValueError("receipt metric must have known direction and finite value")
        if type(self.from_step) is not int or type(self.to_step) is not int or self.from_step < 0 or self.to_step <= self.from_step:
            raise ValueError("receipt step range is invalid")
        for field in (
            "ladder_sha256", "metric_sha256", "checkpoint_sha256", "evidence_bundle_sha256",
            "code_fingerprint_sha256", "dataset_fingerprint_sha256",
            "run_spec_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a SHA-256 digest")
        if self.termination_cause is not None and not isinstance(self.termination_cause, str):
            raise ValueError("termination_cause must be a string or null")


def load_verified_receipt(
    path: Path | str, *, ladder: PaperStepLadder, run_dir: Path | str
) -> BranchRungReceipt | None:
    """Read one receipt and return it only when every local binding verifies."""
    try:
        run_dir = Path(run_dir).resolve()
        receipt_path = _receipt_path(path, run_dir)
        if receipt_path is None:
            return None
        receipt_bytes = receipt_path.read_bytes()
        raw = json.loads(receipt_bytes)
        if not isinstance(raw, Mapping):
            return None
        receipt = _receipt_from_mapping(raw)
        if not _ledger_attests_receipt(receipt, receipt_bytes, run_dir):
            return None
        if not _matches_ladder(receipt, ladder):
            return None
        if not _verify_metric(raw.get("metric"), receipt, run_dir):
            return None
        if not _verify_checkpoint(raw.get("checkpoint"), receipt, run_dir):
            return None
        if not _verify_evidence_bundle(raw.get("evidence_bundle"), receipt, run_dir):
            return None
        if not _verify_fingerprints(raw, receipt, run_dir):
            return None
        return receipt
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _receipt_from_mapping(raw: Mapping[str, Any]) -> BranchRungReceipt:
    metric = _mapping(raw.get("metric"), "metric")
    checkpoint = _mapping(raw.get("checkpoint"), "checkpoint")
    evidence_bundle = _mapping(raw.get("evidence_bundle"), "evidence_bundle")
    fingerprints = _mapping(raw.get("fingerprints"), "fingerprints")
    return BranchRungReceipt(
        campaign_id=_nonempty_string(raw.get("campaign_id"), "campaign_id"),
        branch_id=_nonempty_string(raw.get("branch_id"), "branch_id"),
        parent_branch_id=_optional_string(raw.get("parent_branch_id"), "parent_branch_id"),
        attempt_n=_positive_int(raw.get("attempt_n"), "attempt_n"),
        cell_id=_nonempty_string(raw.get("cell_id"), "cell_id"),
        paper_ref=_nonempty_string(raw.get("paper_ref"), "paper_ref"),
        ladder_sha256=_sha(raw.get("ladder_sha256"), "ladder_sha256"),
        from_step=_nonnegative_int(raw.get("from_step"), "from_step"),
        to_step=_positive_int(raw.get("to_step"), "to_step"),
        metric_id=_nonempty_string(metric.get("id"), "metric.id"),
        direction=_direction(metric.get("direction")),
        metric_value=_finite_number(metric.get("value"), "metric.value"),
        metric_sha256=_sha(metric.get("sha256"), "metric.sha256"),
        checkpoint_path=_nonempty_string(checkpoint.get("path"), "checkpoint.path"),
        checkpoint_sha256=_sha(checkpoint.get("sha256"), "checkpoint.sha256"),
        evidence_bundle_sha256=_sha(evidence_bundle.get("sha256"), "evidence_bundle.sha256"),
        code_fingerprint_sha256=_sha(fingerprints.get("code_sha256"), "fingerprints.code_sha256"),
        dataset_fingerprint_sha256=_sha(fingerprints.get("dataset_sha256"), "fingerprints.dataset_sha256"),
        run_spec_sha256=_sha(fingerprints.get("run_spec_sha256"), "fingerprints.run_spec_sha256"),
        seed=_integer(raw.get("seed"), "seed"),
        termination_cause=_optional_string(raw.get("termination_cause"), "termination_cause"),
        schema_version=_positive_int(raw.get("schema_version", SCHEMA_VERSION), "schema_version"),
    )


def _matches_ladder(receipt: BranchRungReceipt, ladder: PaperStepLadder) -> bool:
    try:
        rung_index = ladder.rung_steps.index(receipt.to_step)
    except ValueError:
        return False
    expected_from = 0 if rung_index == 0 else ladder.rung_steps[rung_index - 1]
    return (
        receipt.paper_ref == ladder.paper_ref
        and receipt.ladder_sha256 == ladder.sha256
        and receipt.metric_id == ladder.metric_id
        and receipt.direction == ladder.direction
        and receipt.from_step == expected_from
        and receipt.to_step <= ladder.r_max_steps
    )


def _verify_metric(raw: Any, receipt: BranchRungReceipt, run_dir: Path) -> bool:
    if not isinstance(raw, Mapping):
        return False
    artifact = _safe_relative_file(raw.get("artifact_path"), run_dir)
    artifact_bytes = _read_bytes(artifact)
    if artifact_bytes is None or _sha256_bytes(artifact_bytes) != receipt.metric_sha256:
        return False
    try:
        payload = json.loads(artifact_bytes)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, Mapping):
        return False
    value = payload.get(receipt.metric_id)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == receipt.metric_value
    )


def _verify_checkpoint(raw: Any, receipt: BranchRungReceipt, run_dir: Path) -> bool:
    if not isinstance(raw, Mapping):
        return False
    path = _safe_relative_file(raw.get("path"), run_dir)
    checkpoint_bytes = _read_bytes(path)
    if path is None or str(raw.get("path")) != receipt.checkpoint_path or checkpoint_bytes is None or _sha256_bytes(checkpoint_bytes) != receipt.checkpoint_sha256:
        return False
    state = raw.get("state")
    if not isinstance(state, Mapping) or set(state) != _REQUIRED_CHECKPOINT_STATE:
        return False
    state_path = _safe_relative_file(raw.get("state_path"), run_dir)
    state_bytes = _read_bytes(state_path)
    if state_bytes is None or _sha256_bytes(state_bytes) != raw.get("state_sha256"):
        return False
    try:
        state_manifest = json.loads(state_bytes)
    except json.JSONDecodeError:
        return False
    return isinstance(state_manifest, Mapping) and state_manifest == state and all(_is_sha256(value) for value in state.values())


def _verify_evidence_bundle(raw: Any, receipt: BranchRungReceipt, run_dir: Path) -> bool:
    if not isinstance(raw, Mapping):
        return False
    path = _safe_relative_file(raw.get("path"), run_dir)
    bundle_bytes = _read_bytes(path)
    if bundle_bytes is None or _sha256_bytes(bundle_bytes) != receipt.evidence_bundle_sha256:
        return False
    try:
        bundle = json.loads(bundle_bytes)
    except json.JSONDecodeError:
        return False
    base = (
        isinstance(bundle, Mapping)
        and bundle.get("schema") == 1
        and bundle.get("coherent") is True
        and bundle.get("metrics_sha256") == receipt.metric_sha256
        and bundle.get("code_tree_digest") == receipt.code_fingerprint_sha256
    )
    return base


def _verify_fingerprints(raw: Mapping[str, Any], receipt: BranchRungReceipt, run_dir: Path) -> bool:
    fingerprints = raw.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        return False
    dataset = _safe_relative_file(fingerprints.get("dataset_manifest_path"), run_dir)
    run_spec = _safe_relative_file(fingerprints.get("run_spec_path"), run_dir)
    dataset_bytes = _read_bytes(dataset)
    run_spec_bytes = _read_bytes(run_spec)
    return (
        dataset_bytes is not None
        and run_spec_bytes is not None
        and _sha256_bytes(dataset_bytes) == receipt.dataset_fingerprint_sha256
        and _sha256_bytes(run_spec_bytes) == receipt.run_spec_sha256
    )


def _ledger_attests_receipt(receipt: BranchRungReceipt, receipt_bytes: bytes, run_dir: Path) -> bool:
    """Only the controller's append-only campaign ledger may attest a receipt."""
    ledger = run_dir / "campaign" / "attempts.jsonl"
    try:
        rows = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    receipt_sha256 = _sha256_bytes(receipt_bytes)
    for line in rows:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(row, Mapping) or row.get("status") != "scheduler_receipt":
            continue
        if (
            row.get("receipt_sha256") == receipt_sha256
            and row.get("branch_id") == receipt.branch_id
            and row.get("attempt_n") == receipt.attempt_n
            and row.get("paper_ref") == receipt.paper_ref
            and row.get("run_spec_sha256") == receipt.run_spec_sha256
        ):
            return True
    return False


def _receipt_path(value: Path | str, root: Path) -> Path | None:
    raw = Path(value)
    unresolved = raw if raw.is_absolute() else root / raw
    if _has_symlink_component(unresolved, root):
        return None
    candidate = unresolved.resolve()
    allowed_root = (root / "campaign" / "scheduler_receipts").resolve()
    try:
        candidate.relative_to(allowed_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _safe_relative_file(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    if raw.is_absolute():
        return None
    unresolved = root / raw
    if _has_symlink_component(unresolved, root):
        return None
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _has_symlink_component(path: Path, root: Path) -> bool:
    """Reject links before resolution so a run cannot redirect evidence reads."""
    root = root.resolve()
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        parent = current.parent
        try:
            parent.relative_to(root)
        except ValueError:
            return True
        if parent == current:
            return True
        current = parent


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _read_bytes(path: Path | None) -> bytes | None:
    if path is None:
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_LENGTH and all(char in "0123456789abcdef" for char in value)


def _sha(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _direction(value: Any) -> Literal["maximize", "minimize"]:
    if value not in _DIRECTIONS:
        raise ValueError("direction must be maximize or minimize")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _step_list(value: Any) -> Sequence[int]:
    if not isinstance(value, list):
        raise ValueError("rung_steps must be a list")
    return [_positive_int(step, "rung_steps entry") for step in value]
