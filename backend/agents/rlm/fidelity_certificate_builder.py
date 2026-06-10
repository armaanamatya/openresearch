"""U16 — Executable fidelity-certificate builder.

Produces ``<run_dir>/rlm_state/fidelity_certificate.json`` — the artifact that
``two_axis_report.load_certificate`` reads to decide whether the build is
certifiably ``faithful``.

The certificate has seven fields (matching the ``FidelityCertificate`` dataclass
in ``reproducibility_verdict.py`` and consumed by ``two_axis_report.load_certificate``):

  ``invariant_tests_ran``         — True iff code/test_reproduction.py was found and
                                    executed (even if it failed).
  ``invariant_tests_passed``      — True iff the test suite exited 0 (all green).
  ``mutation_confirmed``          — True iff EVERY registered invariant constant,
                                    when perturbed, caused at least one test to fail.
                                    Bounded to constants in PAPER_HINTS / ReproSpec.
  ``blinded_extraction_agreed``   — Passthrough from repro_spec.json
                                    (``blinded_extraction_agreed`` key); False when the
                                    artifact is absent (conservative — no blinded
                                    verifier has run yet).
  ``obligation_profile``          — One of {static, forward_pass, multi_step, trace,
                                    end_to_end}; inferred from repro_spec.json or
                                    paper type; defaults to ``end_to_end``.
  ``profile_satisfied``           — Whether the evidence demanded by the obligation
                                    profile was produced (currently: has_measured_metrics
                                    for end_to_end; always True for static/forward_pass
                                    when tests ran green).
  ``has_measured_metrics``        — True iff code/metrics.json or
                                    code/outputs/**/metrics.json exists.

Design principles (mirrors the locked spec):
  * Flag-gated: does nothing (and writes no file) unless OPENRESEARCH_TWO_AXIS_VERDICT
    is truthy.  The gate keeps existing behaviour byte-for-byte unchanged when off.
  * Fail-soft: any internal exception leaves the existing certificate intact (or
    writes a not-green one) rather than blocking the run.  A not-green cert means
    implementation_verdict caps at ``partial`` — not ``broken``.
  * Conservative: every uncertain field defaults False.  A not-green cert can never
    mint a false ``faithful`` verdict.  A green cert (all True) is only possible
    when executable evidence supports it.
  * Local-first: only touches local files and spawns subprocesses in the run's own
    code directory.  Does NOT modify runpod/docker exec paths.

A6b (execution-trace depth):
  The invariant tests are run against the PRODUCTION entry points in code/ — the
  actual ``train.py`` / ``train_cell.py`` the agent wrote, NOT a copy.  The tests
  must import them from the flat ``code/`` sandbox (no ``backend`` imports), so they
  naturally exercise the real code path.

  **What is NOT yet implemented (TODO):**
  Full execution-trace evidence (verifying the asserted PRODUCTION path actually
  ran, e.g. by checking a sentinel written by the production code, or by comparing
  call-graph fingerprints) is deferred.  The obligation_profile is set to
  ``forward_pass`` (not ``trace``) when no trace evidence is available, and
  ``profile_satisfied`` is True only when the tests ran green on the real code.
  A ``trace``-level certificate requires the harness to inject a sentinel call-hook
  into the production code and verify the hook fired — that is the next iteration's
  work (leave the TODO below, it is the honest boundary).

Mutation-testing strategy (A6b, Tier 4):
  For each constant named in ``PAPER_HINTS[arxiv_id].invariants``, we:
    1. Locate the constant in the agent's code via the same ``must_match`` regex
       the rubric uses (first match → concrete numeric value).
    2. Substitute a perturbed value (+1 for integers, *1.1 for floats, swapping
       sign, or a structurally different value).
    3. Write the perturbed file to a temp location, run test_reproduction.py
       against the perturbed code/, and assert it FAILS (exit != 0).
    4. Restore the original file.
  If a test PASSES under mutation, the invariant test is tautological → set
  ``mutation_confirmed=False`` and record which constant was tautologically tested.
  Bounded to registered constants: if no constants are registered (no PAPER_HINTS
  entry), mutation_confirmed defaults False (conservative — can't confirm without
  knowing what to mutate).

  Note: mutation operates on the SAME ``code/`` directory the tests run against,
  with a temporary file swap.  Thread-unsafe; the builder is not re-entrant for the
  same run_dir, but ``primitives.py`` always calls it from a single thread.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CERT_FILENAME = "fidelity_certificate.json"
_TEST_SCRIPT = "test_reproduction.py"

# How long to wait for a single test run (seconds).  Generous but bounded —
# the invariant tests run on CPU with tiny data.
_DEFAULT_TEST_TIMEOUT_S = int(os.environ.get("OPENRESEARCH_FIDELITY_TEST_TIMEOUT_S", "120"))

# How long to wait for a single MUTATED test run.  Should fail fast.
_MUTATION_TIMEOUT_S = int(os.environ.get("OPENRESEARCH_FIDELITY_MUTATION_TIMEOUT_S", "60"))

_VALID_PROFILES = frozenset({"static", "forward_pass", "multi_step", "trace", "end_to_end"})


# ---------------------------------------------------------------------------
# Flag gate (mirrors two_axis_report.is_enabled)
# ---------------------------------------------------------------------------

def _is_enabled() -> bool:
    return os.environ.get("OPENRESEARCH_TWO_AXIS_VERDICT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ---------------------------------------------------------------------------
# Metrics presence check
# ---------------------------------------------------------------------------

def _has_measured_metrics(run_dir: Path) -> bool:
    """True iff any metrics.json exists under code/ or code/outputs/."""
    code = run_dir / "code"
    if (code / "metrics.json").exists():
        return True
    outputs = code / "outputs"
    if outputs.exists():
        for p in outputs.rglob("metrics.json"):
            if p.exists():
                return True
    return False


# ---------------------------------------------------------------------------
# repro_spec.json loader (blinded_extraction_agreed + obligation_profile)
# ---------------------------------------------------------------------------

def _read_repro_spec(run_dir: Path) -> dict[str, Any]:
    """Load rlm_state/repro_spec.json; return {} when absent or unreadable."""
    path = run_dir / "rlm_state" / "repro_spec.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("fidelity_cert: could not read repro_spec.json (%s)", exc)
        return {}


def _blinded_extraction_agreed(spec: dict[str, Any]) -> bool:
    """Passthrough: True iff the repro_spec records extractor/verifier agreement."""
    # The blinded verifier (U13 Extractor + independent verifier) writes this key
    # when both agree on the claim set.  Until that unit lands, conservative False.
    return bool(spec.get("blinded_extraction_agreed", False))


def _infer_obligation_profile(spec: dict[str, Any], has_metrics: bool) -> str:
    """Infer the obligation profile from repro_spec or fall back to end_to_end.

    A7 — the profile scales the certificate to the claim type:
      * ``end_to_end``   — requires measured metrics (a full training run).
      * ``multi_step``   — requires at least a few forward/backward steps.
      * ``forward_pass`` — a single forward pass suffices (eval-only / analysis).
      * ``static``       — code inspection only (deterministic math, no GPU).

    When repro_spec does not specify a profile, default to ``end_to_end`` (most
    demanding) — the conservative choice.  Override by setting
    ``obligation_profile`` in repro_spec.json.
    """
    raw = str(spec.get("obligation_profile", "")).strip().lower()
    if raw in _VALID_PROFILES:
        return raw
    return "end_to_end"


def _profile_satisfied(profile: str, tests_ran: bool, tests_passed: bool, has_metrics: bool) -> bool:
    """Whether the evidence demanded by the obligation profile was produced.

    Conservative: only True when the REQUIRED evidence is present.

      ``static``       — tests ran green (no runtime evidence needed).
      ``forward_pass`` — tests ran green (a forward pass was exercised).
      ``multi_step``   — tests ran green (ditto; we don't yet count steps).
      ``trace``        — tests ran green AND has_metrics (trace implies a run happened).
                         TODO: full trace evidence (A6b) would check a sentinel.
      ``end_to_end``   — tests ran green AND has_metrics (a full run was observed).
    """
    if not tests_ran or not tests_passed:
        return False
    if profile in ("static", "forward_pass", "multi_step"):
        return True
    # trace / end_to_end both require observed runtime metrics.
    return has_metrics


# ---------------------------------------------------------------------------
# Running test_reproduction.py
# ---------------------------------------------------------------------------

def _find_python() -> str:
    """Return a usable Python interpreter path (prefer the running interpreter)."""
    # Use the same interpreter that is running us — this is the per-run venv
    # python that already has the paper's deps installed.
    return sys.executable or "python3"


def _run_tests(code_dir: Path, *, timeout_s: int, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run test_reproduction.py in code_dir; return (returncode, combined_output).

    The test file is run inside code_dir so relative imports of copied helpers
    (rubric_guard, sdar_env_base, etc.) resolve correctly — mirroring the flat
    sandbox where train.py runs.

    We set OPENRESEARCH_SMOKE_STEPS=1 so any forward-pass calls inside the test stay
    cheap (1 step), and CUDA_LAUNCH_BLOCKING=1 so device-side errors appear at
    the correct line.

    Returns (returncode, stdout+stderr output).  Never raises.
    """
    python = _find_python()
    env = dict(os.environ)
    env.update({
        "OPENRESEARCH_SMOKE_STEPS": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
    })
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            [python, "-m", "pytest", _TEST_SCRIPT, "-x", "--tb=short", "-q"],
            cwd=str(code_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        logger.info("fidelity_cert: test run timed out after %ds (soft pass)", timeout_s)
        return 124, f"TIMEOUT after {timeout_s}s"
    except Exception as exc:  # noqa: BLE001
        logger.warning("fidelity_cert: test run failed to launch: %s", exc)
        return -1, str(exc)


# ---------------------------------------------------------------------------
# Invariant-constant extraction from PAPER_HINTS
# ---------------------------------------------------------------------------

def _get_paper_hints_constants(arxiv_id: str | None) -> list[dict[str, Any]]:
    """Return registered invariant constant specs for mutation testing.

    Each entry has keys:
      ``invariant_name`` — the InvariantSpec.name
      ``must_match``     — list of regex patterns that matched the constant
      ``files``          — list of (Path, line_no, matched_text) tuples where
                           the constant was found in the agent's code

    Returns [] when:
      * No arxiv_id
      * No PAPER_HINTS entry for the arxiv_id
      * No invariants declared
    """
    if not arxiv_id:
        return []
    try:
        from backend.agents.prompts.paper_hints import PAPER_HINTS
        hint = PAPER_HINTS.get(arxiv_id)
        if hint is None:
            return []
        return [
            {"invariant_name": inv.name, "must_match": list(inv.must_match)}
            for inv in (hint.invariants or [])
            if inv.must_match
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("fidelity_cert: could not load PAPER_HINTS for %s: %s", arxiv_id, exc)
        return []


def _find_constant_in_code(code_dir: Path, patterns: list[str]) -> list[tuple[Path, int, str]]:
    """Scan code_dir/*.py for the first match of any of ``patterns``.

    Returns a list of (file_path, 1-based line, matched line text).
    Only returns the first match per file to keep the mutation surface bounded.
    Fail-soft: any I/O error skips that file.
    """
    matches: list[tuple[Path, int, str]] = []
    compiled = [re.compile(p) for p in patterns]
    for py_file in sorted(code_dir.glob("*.py")):
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: BLE001
            continue
        for lineno, line in enumerate(lines, 1):
            for pat in compiled:
                if pat.search(line):
                    matches.append((py_file, lineno, line))
                    break  # one match per file is enough
            else:
                continue
            break  # first match in this file; move to next file
    return matches


def _perturb_value(matched_line: str, pattern: str) -> str | None:
    """Return a perturbed version of the numeric constant found in ``matched_line``.

    Strategy (simple, deterministic, bounded to numeric constants):
      * Find the first decimal/integer literal in the match region.
      * If it's an integer N, replace with N+1.
      * If it's a float f, replace with f * 1.1 (one decimal place).
      * If the value is 0.0 / 0, replace with 1 instead.

    Returns None when no numeric literal is found (can't mutate → skip).
    """
    # Find numeric literal: optional sign, digits, optional decimal
    num_re = re.compile(r"(?<![a-zA-Z_])(-?\d+(?:\.\d+)?)(?![a-zA-Z_])")
    # Try to find numbers after the pattern match region
    m = re.search(pattern, matched_line)
    search_from = m.end() if m else 0
    tail = matched_line[search_from:]
    nm = num_re.search(tail)
    if nm is None:
        # Fall back: scan the whole line
        nm = num_re.search(matched_line)
    if nm is None:
        return None

    original = nm.group(0)
    try:
        if "." in original:
            val = float(original)
            if val == 0.0:
                perturbed = "1.0"
            else:
                perturbed = f"{val * 1.1:.6g}"
        else:
            val = int(original)
            if val == 0:
                perturbed = "1"
            else:
                perturbed = str(val + 1)
    except ValueError:
        return None

    # Replace the first occurrence of the original numeric literal (as a whole token)
    # in the line with the perturbed value.
    token_re = re.compile(r"(?<![a-zA-Z_])" + re.escape(original) + r"(?![a-zA-Z_])")
    perturbed_line, n = token_re.subn(perturbed, matched_line, count=1)
    if n == 0:
        return None
    return perturbed_line


# ---------------------------------------------------------------------------
# Mutation testing
# ---------------------------------------------------------------------------

def _run_mutation_check(
    code_dir: Path,
    constants: list[dict[str, Any]],
    *,
    timeout_s: int,
) -> tuple[bool, list[str]]:
    """Perturb each registered constant and confirm test_reproduction.py fails.

    Returns ``(mutation_confirmed, notes)`` where:
      * ``mutation_confirmed=True`` iff EVERY constant's perturbation caused at
        least one test to fail (exit != 0 AND exit != 124).  This is the "tests
        actually bite" guarantee (Tier 4).
      * ``mutation_confirmed=False`` with a non-empty ``notes`` list when any
        constant's perturbation did NOT cause a test failure.

    Conservatively returns ``(False, ["no_constants"])`` when:
      * ``constants`` is empty (nothing to mutate → can't confirm).
      * All constants were unfound in the code (patterns matched nothing).

    Always returns ``(False, ...)`` when the clean test suite itself did not pass
    (no point mutating a failing test suite).
    """
    if not constants:
        return False, ["no registered invariant constants to mutate"]

    notes: list[str] = []
    mutated_count = 0
    all_biting = True

    for const_spec in constants:
        inv_name = const_spec["invariant_name"]
        patterns = const_spec["must_match"]

        file_matches = _find_constant_in_code(code_dir, patterns)
        if not file_matches:
            logger.debug("fidelity_cert: mutation: no match for invariant %s", inv_name)
            # No match means we can't mutate → can't confirm → conservative False
            notes.append(f"constant_not_found:{inv_name}")
            all_biting = False
            continue

        # Mutate the first match only (deterministic, bounded)
        py_file, lineno, matched_line = file_matches[0]
        first_pattern = patterns[0]

        perturbed_line = _perturb_value(matched_line, first_pattern)
        if perturbed_line is None:
            logger.debug("fidelity_cert: mutation: could not perturb line for %s", inv_name)
            notes.append(f"perturb_failed:{inv_name}")
            all_biting = False
            continue

        # Swap the line in the file (minimal in-place edit, no temp dir needed for
        # a single-line swap).
        try:
            original_text = py_file.read_text(encoding="utf-8")
            lines = original_text.splitlines(keepends=True)
            original_line_text = lines[lineno - 1]
            # Preserve line ending
            ending = ""
            for end in ("\r\n", "\r", "\n"):
                if original_line_text.endswith(end):
                    ending = end
                    break
            lines[lineno - 1] = perturbed_line.rstrip("\r\n") + ending
            py_file.write_text("".join(lines), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("fidelity_cert: mutation: could not write perturbed file: %s", exc)
            notes.append(f"write_failed:{inv_name}")
            all_biting = False
            continue

        try:
            rc, output = _run_tests(code_dir, timeout_s=timeout_s)
            mutated_count += 1

            if rc == 124:
                # Timed out under mutation — ambiguous, treat as non-biting
                logger.info("fidelity_cert: mutation: test timed out for %s (tautological?)", inv_name)
                notes.append(f"mutation_timeout:{inv_name}")
                all_biting = False
            elif rc == 0:
                # Tests PASSED under mutation → tautological
                logger.info(
                    "fidelity_cert: mutation: tests PASSED under mutation of %s "
                    "(tautological — does not bite)", inv_name
                )
                notes.append(f"tautological:{inv_name}")
                all_biting = False
            else:
                # Tests failed under mutation — the test bites this constant
                logger.info("fidelity_cert: mutation: tests correctly failed under mutation of %s", inv_name)
        finally:
            # Always restore the original file, even on exception
            try:
                py_file.write_text(original_text, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning("fidelity_cert: mutation: could not restore %s: %s", py_file, exc)

    if mutated_count == 0:
        # All constants were missing or unparseable
        return False, notes if notes else ["no_mutable_constants_found"]

    return all_biting, notes


# ---------------------------------------------------------------------------
# Obligation-profile evidence check (A7)
# ---------------------------------------------------------------------------

def _check_profile_evidence(
    profile: str,
    code_dir: Path,
    tests_ran: bool,
    tests_passed: bool,
    has_metrics: bool,
) -> bool:
    """Check whether the evidence demanded by the obligation profile was produced.

    This is the first-cut implementation.  The ``trace`` profile is intentionally
    not fully satisfied here — see the module-level TODO on A6b.
    """
    # TODO (A6b full exec-trace): For `trace` and `end_to_end`, inject a harness
    # sentinel into the production entry point, run it, and verify the sentinel
    # file was written.  That proves the asserted production path ran, not a copy.
    # For now we require tests_passed + (has_metrics for runtime profiles).
    return _profile_satisfied(profile, tests_ran, tests_passed, has_metrics)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_certificate(
    run_dir: Path,
    *,
    arxiv_id: str | None = None,
    timeout_s: int = _DEFAULT_TEST_TIMEOUT_S,
    mutation_timeout_s: int = _MUTATION_TIMEOUT_S,
) -> dict[str, Any]:
    """Build the fidelity certificate for ``run_dir`` and write it to disk.

    Returns the certificate dict (regardless of whether it was written), so
    callers can inspect it without re-reading from disk.  The dict keys EXACTLY
    match what ``two_axis_report.load_certificate`` expects.

    When ``OPENRESEARCH_TWO_AXIS_VERDICT`` is off, returns a not-green stub dict
    (``invariant_tests_ran=False``) without writing any file — byte-for-byte
    preserves existing behaviour.

    Fail-soft: any exception during a sub-step is caught; the relevant field
    defaults to its conservative (False) value.  An exception during the final
    write is logged and the in-memory dict is returned.
    """
    # Always return a valid dict even when disabled or on error
    stub = _not_green_stub(has_measured_metrics=False)

    if not _is_enabled():
        return stub

    code_dir = run_dir / "code"
    if not code_dir.is_dir():
        logger.info("fidelity_cert: code/ not found in %s — no cert", run_dir)
        return stub

    has_metrics = _has_measured_metrics(run_dir)
    spec = _read_repro_spec(run_dir)
    blinded = _blinded_extraction_agreed(spec)
    profile = _infer_obligation_profile(spec, has_metrics)

    test_script = code_dir / _TEST_SCRIPT
    invariant_tests_ran = False
    invariant_tests_passed = False
    test_output = ""

    if test_script.exists():
        logger.info("fidelity_cert: running %s", test_script)
        try:
            rc, test_output = _run_tests(code_dir, timeout_s=timeout_s)
            if rc == 124:
                # Timed out — tests ran but result is ambiguous (soft pass for ran,
                # fail for passed: can't certify green when we don't know the outcome)
                invariant_tests_ran = True
                invariant_tests_passed = False
                logger.info("fidelity_cert: test run timed out — invariant_tests_passed=False")
            elif rc == 0:
                invariant_tests_ran = True
                invariant_tests_passed = True
                logger.info("fidelity_cert: tests PASSED")
            else:
                invariant_tests_ran = True
                invariant_tests_passed = False
                logger.info("fidelity_cert: tests FAILED (rc=%d)", rc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fidelity_cert: unexpected error running tests: %s", exc)
            invariant_tests_ran = False
            invariant_tests_passed = False
    else:
        # No test_reproduction.py → invariant_tests_ran=False → PARTIAL, not broken
        logger.info("fidelity_cert: no %s found — invariant_tests_ran=False", _TEST_SCRIPT)

    # Mutation check — only meaningful when tests passed (otherwise we already know
    # the code is unfaithful, no need to confirm biting).
    mutation_confirmed = False
    mutation_notes: list[str] = []

    if invariant_tests_ran and invariant_tests_passed:
        constants = _get_paper_hints_constants(arxiv_id)
        if not constants:
            # Try repro_spec.json for registered constants (U13/U14 path)
            constants = _extract_constants_from_spec(spec)

        if constants:
            try:
                mutation_confirmed, mutation_notes = _run_mutation_check(
                    code_dir,
                    constants,
                    timeout_s=mutation_timeout_s,
                )
                if mutation_confirmed:
                    logger.info("fidelity_cert: mutation confirmed (all constants bite)")
                else:
                    logger.info(
                        "fidelity_cert: mutation NOT confirmed: %s",
                        "; ".join(mutation_notes) or "unknown",
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("fidelity_cert: mutation check failed: %s", exc)
                mutation_confirmed = False
                mutation_notes = [f"exception:{exc}"]
        else:
            # No registered constants → can't confirm mutation → conservative False
            logger.info("fidelity_cert: no registered constants for mutation check — mutation_confirmed=False")
            mutation_notes = ["no_constants_registered"]
    elif not invariant_tests_ran:
        mutation_notes = ["tests_did_not_run"]
    else:
        mutation_notes = ["tests_did_not_pass"]

    profile_satisfied = _check_profile_evidence(
        profile, code_dir, invariant_tests_ran, invariant_tests_passed, has_metrics
    )

    cert: dict[str, Any] = {
        "invariant_tests_ran": invariant_tests_ran,
        "invariant_tests_passed": invariant_tests_passed,
        "mutation_confirmed": mutation_confirmed,
        "blinded_extraction_agreed": blinded,
        "obligation_profile": profile,
        "profile_satisfied": profile_satisfied,
        "has_measured_metrics": has_metrics,
        # Extra context for diagnostics (not read by load_certificate, but useful)
        "_mutation_notes": mutation_notes,
        "_test_output_tail": test_output[-2000:] if test_output else "",
    }

    # Write to rlm_state/fidelity_certificate.json (atomic-ish write)
    try:
        state_dir = run_dir / "rlm_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cert_path = state_dir / _CERT_FILENAME
        # Write to a temp file then rename for atomicity
        tmp = cert_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cert, indent=2), encoding="utf-8")
        tmp.replace(cert_path)
        logger.info("fidelity_cert: written to %s", cert_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fidelity_cert: could not write certificate: %s", exc)

    return cert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _not_green_stub(has_measured_metrics: bool = False) -> dict[str, Any]:
    """Return a conservative not-green certificate dict (all flags False)."""
    return {
        "invariant_tests_ran": False,
        "invariant_tests_passed": False,
        "mutation_confirmed": False,
        "blinded_extraction_agreed": False,
        "obligation_profile": "end_to_end",
        "profile_satisfied": False,
        "has_measured_metrics": has_measured_metrics,
        "_mutation_notes": [],
        "_test_output_tail": "",
    }


def _extract_constants_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract mutation-testable constant specs from a ReproSpec (U13/U14 path).

    Reads ``spec.get("invariants", [])`` where each entry may carry
    ``must_match`` patterns (same schema as InvariantSpec).  Falls back to []
    when absent or malformed.
    """
    result: list[dict[str, Any]] = []
    for inv in (spec.get("invariants") or []):
        if not isinstance(inv, dict):
            continue
        name = str(inv.get("name", "unnamed"))
        patterns = inv.get("must_match") or []
        if not isinstance(patterns, list) or not patterns:
            continue
        result.append({"invariant_name": name, "must_match": [str(p) for p in patterns]})
    return result


__all__ = [
    "build_certificate",
]
