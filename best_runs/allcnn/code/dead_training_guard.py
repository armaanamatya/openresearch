"""Dead-training early-stop guard — detect a cell whose loss is pinned (network not
learning) and stop it early as a REPAIRABLE ``training_diverged`` signal, instead of
burning the full epoch budget on a model that provably never learns.

Motivation (2026-06-09, All-CNN 1412.6806). The agent's max-pool architecture variants
(``base_a``, ``convpool_a``) trained for the FULL 350 epochs with ``train_loss`` pinned
at exactly ``ln(10) = 2.3026`` and ``test_acc = 0.1`` (random) — dead networks emitting
constant outputs, zero gradient signal — while the strided (all-conv) variant learned
fine (0.80). Two costs:

  1. ~19 min of GPU wasted **per dead cell**, across many cells (~8 h for the grid).
  2. Worse: a dead cell **exits 0**, so the cell runner records ``status="ok"`` and the
     matrix returns ``success`` with a quietly low rubric score and NO actionable
     signal. A model that ran to completion but never learned is indistinguishable from
     a good one at the exit-code level.

This guard converts that into an early, actionable, REPAIRABLE signal: a cell whose loss
is flat-and-high for ``window`` consecutive epochs (with no meaningful descent from where
it started) is killed and surfaced as ``training_diverged`` — so the orchestrator's
existing repair path drives the agent to FIX its own architecture bug on the next
iteration, rather than the harness silently scoring a broken implementation low.

Signal design — robust + near-zero false-positive, scale/num-classes-agnostic:
  A LIVE network's loss jitters epoch-to-epoch from minibatch noise; a DEAD network
  emits identical outputs → identical loss every epoch. So we early-stop ONLY when ALL
  of the following hold over the most recent ``window`` per-epoch loss readings:

    * we have observed >= ``window`` epochs of loss,
    * the window is FLAT:        ``max - min < flat_eps``   (healthy SGD never is),
    * the window value is HIGH:  ``value >= min_loss``       (a flat LOW loss is a
                                                              legitimate converged
                                                              plateau — never killed),
    * training never DESCENDED:  the best loss seen so far is still >= ``descent_frac``
                                 of the first loss (a model that fell 2.3 -> 0.1 then
                                 plateaus has best << first → NOT flagged; a model stuck
                                 at its starting loss has best ~= first → flagged).

All four conditions together make a false positive on healthy training essentially
impossible: a converging model fails the FLAT test early and the NO-DESCENT test later.

The whole guard is gated behind ``REPROLAB_DEAD_LOSS_EARLYSTOP`` (default OFF). When
off, the cell runner never instantiates a detector and behaviour is byte-for-byte
unchanged. It is ``local``/cell-path scoped — the monolithic and runpod/docker exec
paths do not call it.
"""
from __future__ import annotations

import math
import os
import re
from collections import deque

# A captured-output marker the cell runner appends when it early-stops a dead cell, so
# the post-hoc status classifier (mirror of ``_is_oom``) can recognise the divergence
# without threading a side-channel return value out of the streaming reader thread.
MARKER = "[gpu_cell_runner] DEAD-TRAINING early-stop"

# Prefer ``train_loss`` (the optimisation target) but fall back to a bare ``loss``. We
# deliberately do NOT match ``val_loss`` / ``test_loss`` first — a dead net's TRAIN loss
# is the cleanest stuck signal. ``(?<![\w])`` avoids matching inside another token.
_TRAIN_LOSS_RE = re.compile(r"(?<![\w])train[_ ]?loss\s*[=:]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_LOSS_RE = re.compile(r"(?<![\w])loss\s*[=:]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def is_enabled() -> bool:
    """True when ``REPROLAB_DEAD_LOSS_EARLYSTOP`` is truthy (default OFF — opt-in)."""
    return os.environ.get("REPROLAB_DEAD_LOSS_EARLYSTOP", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def extract_loss(line: str) -> float | None:
    """Extract a per-epoch loss value from one log line, or ``None`` if absent.

    Prefers ``train_loss`` over a bare ``loss``. Non-finite values (NaN/Inf — a
    separate failure mode the NaN guards already handle) are ignored here so this guard
    stays focused on the *flat-and-high* dead-training signature.
    """
    m = _TRAIN_LOSS_RE.search(line) or _LOSS_RE.search(line)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _random_guess_note(value: float) -> str:
    """If ``value`` ~= ln(k) for a small integer k, name the random-guess interpretation.

    ``ln(10) = 2.3026`` is the cross-entropy of uniform predictions over 10 classes —
    the canonical dead-classifier loss. Surfacing this in the diagnosis makes the bug
    self-evident to the agent ("you're at random-guess loss for 10 classes").
    """
    try:
        if value <= 0:
            return ""
        k = round(math.exp(value))
        if 2 <= k <= 1000 and abs(value - math.log(k)) < 0.02:
            return f" (~= ln({k}) = the uniform random-guess loss for {k} classes)"
    except (ValueError, OverflowError):
        pass
    return ""


class DeadTrainingDetector:
    """Streaming detector: feed it log lines; it flags provably-stuck training.

    Construct one per cell subprocess. Call :meth:`observe` with each streamed line; it
    returns a human-readable diagnosis string the first time the dead-training signature
    is met (and on every line thereafter), else ``None``. The caller early-stops the
    subprocess on the first non-``None`` return.

    Defaults are overridable via env (so an operator can tune sensitivity per fleet
    without a code change), but the defaults are chosen to make a false positive on
    healthy training essentially impossible:

      * ``REPROLAB_DEAD_LOSS_WINDOW``  (default 40)    consecutive flat epochs to trip
      * ``REPROLAB_DEAD_LOSS_EPS``     (default 1e-3)  max-min flatness threshold
      * ``REPROLAB_DEAD_LOSS_MIN``     (default 0.2)   only trip above this loss value
      * ``REPROLAB_DEAD_LOSS_DESCENT`` (default 0.9)   best must stay >= this * first
    """

    def __init__(
        self,
        *,
        window: int | None = None,
        flat_eps: float | None = None,
        min_loss: float | None = None,
        descent_frac: float | None = None,
    ) -> None:
        self.window = int(window if window is not None else _env_int("REPROLAB_DEAD_LOSS_WINDOW", 40))
        self.flat_eps = float(flat_eps if flat_eps is not None else _env_float("REPROLAB_DEAD_LOSS_EPS", 1e-3))
        self.min_loss = float(min_loss if min_loss is not None else _env_float("REPROLAB_DEAD_LOSS_MIN", 0.2))
        self.descent_frac = float(
            descent_frac if descent_frac is not None else _env_float("REPROLAB_DEAD_LOSS_DESCENT", 0.9)
        )
        self.window = max(self.window, 2)  # a window of <2 is meaningless
        self._recent: deque[float] = deque(maxlen=self.window)
        self._first: float | None = None
        self._best: float = math.inf
        self._n_seen = 0

    def observe_loss(self, loss: float) -> str | None:
        """Feed one numeric loss reading; return a diagnosis when dead, else ``None``."""
        if loss is None or not math.isfinite(loss):
            return None
        if self._first is None:
            self._first = loss
        self._best = min(self._best, loss)
        self._recent.append(loss)
        self._n_seen += 1

        if len(self._recent) < self.window:
            return None
        hi, lo = max(self._recent), min(self._recent)
        value = self._recent[-1]
        flat = (hi - lo) < self.flat_eps
        high = value >= self.min_loss
        # Training never meaningfully descended: the best loss ever seen is still within
        # ``descent_frac`` of where it started. A model that fell from ``first`` to a far
        # smaller ``best`` (then plateaued) is a CONVERGED model, not a dead one.
        no_descent = self._first is not None and self._best >= self.descent_frac * self._first
        if flat and high and no_descent:
            note = _random_guess_note(value)
            return (
                f"training diverged: loss flat at {value:.4f}{note} for "
                f"{self.window} consecutive epochs with no descent from the initial "
                f"{self._first:.4f} (best={self._best:.4f}) — the network is not "
                f"learning (likely dead activations, bad weight init, a missing "
                f"normalization layer, or a pooling/shape bug in this architecture)"
            )
        return None

    def observe(self, line: str) -> str | None:
        """Feed one streamed log line; return a diagnosis when dead, else ``None``."""
        loss = extract_loss(line)
        if loss is None:
            return None
        return self.observe_loss(loss)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def is_dead_training(output: str) -> bool:
    """True iff captured cell output carries the early-stop :data:`MARKER`."""
    return MARKER in (output or "")


__all__ = [
    "MARKER",
    "is_enabled",
    "extract_loss",
    "DeadTrainingDetector",
    "is_dead_training",
]
