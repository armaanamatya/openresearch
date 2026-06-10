"""Dead-training early-stop guard — the detector decision logic.

The All-CNN 2026-06-09 case: ``base_a``/``convpool_a`` cells ran the full 350 epochs
with ``train_loss`` pinned at ln(10)=2.3026 and test_acc=0.1, while ``strided_a``
learned (0.80). These tests pin the four-condition signature so a healthy converging
run is NEVER flagged and a dead flat-and-high run always is.
"""

from __future__ import annotations


from backend.agents.rlm.dead_training_guard import (
    DeadTrainingDetector,
    MARKER,
    extract_loss,
    is_dead_training,
    is_enabled,
)


# --------------------------------------------------------------------------- #
# extract_loss
# --------------------------------------------------------------------------- #

def test_extract_train_loss_preferred():
    assert extract_loss("[convpool_a|cifar10] epoch 311/350 train_loss=2.3027 test_acc=0.1") == 2.3027

def test_extract_bare_loss():
    assert extract_loss("[adam] ep 47/50 loss=0.2471 acc=0.9262") == 0.2471

def test_extract_loss_colon_form():
    assert extract_loss("step 10 loss: 1.5e-1") == 0.15

def test_extract_loss_absent():
    assert extract_loss("downloading cifar-10 ... 100%") is None

def test_extract_loss_ignores_nan_inf():
    assert extract_loss("epoch 5 loss=nan") is None
    assert extract_loss("epoch 5 train_loss=inf") is None


# --------------------------------------------------------------------------- #
# DeadTrainingDetector — the dead case (must flag)
# --------------------------------------------------------------------------- #

def test_flat_high_no_descent_is_flagged():
    """ln(10) pinned for window epochs from the start → dead."""
    det = DeadTrainingDetector(window=40, flat_eps=1e-3, min_loss=0.2, descent_frac=0.9)
    out = None
    for _ in range(45):
        out = det.observe("epoch x/350 train_loss=2.3027 test_acc=0.1000")
    assert out is not None
    assert "training diverged" in out
    assert "ln(10)" in out  # random-guess note for 10 classes

def test_flag_exactly_at_window():
    det = DeadTrainingDetector(window=10, flat_eps=1e-3, min_loss=0.2, descent_frac=0.9)
    results = [det.observe_loss(2.3027) for _ in range(10)]
    assert results[8] is None      # not yet (only 9 readings)
    assert results[9] is not None  # the 10th completes the window → flagged


# --------------------------------------------------------------------------- #
# DeadTrainingDetector — healthy cases (must NOT flag)
# --------------------------------------------------------------------------- #

def test_converging_run_is_not_flagged():
    """A descending loss never trips the flat test."""
    det = DeadTrainingDetector(window=40, flat_eps=1e-3, min_loss=0.2, descent_frac=0.9)
    flagged = False
    loss = 2.30
    for _ in range(100):
        loss *= 0.97  # steady descent
        if det.observe_loss(loss) is not None:
            flagged = True
    assert not flagged

def test_converged_low_plateau_is_not_flagged():
    """Descended 2.3 -> 0.05 then plateaus flat-and-LOW → converged, never flagged
    (fails both the min_loss gate and the no-descent gate)."""
    det = DeadTrainingDetector(window=40, flat_eps=1e-3, min_loss=0.2, descent_frac=0.9)
    flagged = False
    for v in [2.3, 1.5, 0.8, 0.3, 0.1]:        # clear descent first
        det.observe_loss(v)
    for _ in range(60):                         # then a flat low plateau at 0.05
        if det.observe_loss(0.05) is not None:
            flagged = True
    assert not flagged

def test_high_but_noisy_is_not_flagged():
    """High loss that still jitters (alive minibatch noise) is not flat → not flagged."""
    det = DeadTrainingDetector(window=40, flat_eps=1e-3, min_loss=0.2, descent_frac=0.9)
    flagged = False
    base = 2.3
    for i in range(80):
        # oscillate by ~0.05 — well above flat_eps
        if det.observe_loss(base + (0.05 if i % 2 else -0.05)) is not None:
            flagged = True
    assert not flagged

def test_descended_then_stuck_high_not_flagged_if_best_low():
    """If best loss fell well below first, the no-descent gate prevents a flag even if
    it later flatlines high (e.g. a metric reset) — we don't kill a model that proved it
    can learn."""
    det = DeadTrainingDetector(window=20, flat_eps=1e-3, min_loss=0.2, descent_frac=0.9)
    for v in [2.3, 1.0, 0.2]:  # best drops to 0.2 (<0.9*2.3)
        det.observe_loss(v)
    flagged = any(det.observe_loss(2.3027) is not None for _ in range(40))
    assert not flagged


# --------------------------------------------------------------------------- #
# non-loss lines + marker helpers
# --------------------------------------------------------------------------- #

def test_non_loss_lines_ignored():
    det = DeadTrainingDetector(window=5, flat_eps=1e-3, min_loss=0.2, descent_frac=0.9)
    for _ in range(20):
        assert det.observe("Files already downloaded and verified") is None

def test_is_dead_training_marker():
    assert is_dead_training(f"...\n{MARKER} cell=base_a\n") is True
    assert is_dead_training("clean run, no marker") is False

def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_DEAD_LOSS_EARLYSTOP", raising=False)
    assert is_enabled() is False
    monkeypatch.setenv("OPENRESEARCH_DEAD_LOSS_EARLYSTOP", "1")
    assert is_enabled() is True


# --------------------------------------------------------------------------- #
# End-to-end wiring: run_matrix early-stops a dead cell → status training_diverged
# --------------------------------------------------------------------------- #

def test_run_matrix_early_stops_dead_cell(tmp_path, monkeypatch):
    """A cell whose loss is pinned gets killed mid-run and recorded as
    training_diverged (not 'ok'), with the flag ON. Proves the reader→kill→status
    wiring end-to-end through a real subprocess."""
    monkeypatch.setenv("OPENRESEARCH_DEAD_LOSS_EARLYSTOP", "1")
    monkeypatch.setenv("OPENRESEARCH_DEAD_LOSS_WINDOW", "5")
    monkeypatch.setenv("OPENRESEARCH_DEAD_LOSS_EPS", "1e-3")
    monkeypatch.setenv("OPENRESEARCH_DEAD_LOSS_MIN", "0.2")
    from backend.agents.rlm import gpu_cell_runner as gcr

    # A cell that emits a pinned loss forever, then would sleep 600s (the guard must
    # kill it long before that).
    script = tmp_path / "dead_cell.py"
    script.write_text(
        "import time, sys\n"
        "for i in range(100):\n"
        "    print(f'epoch {i}/350 train_loss=2.3027 test_acc=0.1000', flush=True)\n"
        "    time.sleep(0.02)\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    cells = [{"id": "dead_a"}]
    monkeypatch.setattr(gcr, "discover_visible_gpus", lambda: ["0"])
    out = gcr.run_matrix(cells, script, output_root=tmp_path / "out",
                         per_cell_timeout_s=120, max_parallel=1)
    assert out["dead_a"]["status"] == "training_diverged"


def test_run_matrix_healthy_cell_not_flagged(tmp_path, monkeypatch):
    """A descending-loss cell completes normally as 'ok' even with the guard ON."""
    monkeypatch.setenv("OPENRESEARCH_DEAD_LOSS_EARLYSTOP", "1")
    monkeypatch.setenv("OPENRESEARCH_DEAD_LOSS_WINDOW", "5")
    from backend.agents.rlm import gpu_cell_runner as gcr
    script = tmp_path / "good_cell.py"
    script.write_text(
        "import os, json\n"
        "loss=2.3\n"
        "for i in range(30):\n"
        "    loss*=0.9\n"
        "    print(f'epoch {i} train_loss={loss:.4f}', flush=True)\n"
        "od=os.environ['OPENRESEARCH_CELL_OUTPUT_DIR']\n"
        "json.dump({'status':'ok','test_accuracy':0.8}, open(od+'/metrics.json','w'))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gcr, "discover_visible_gpus", lambda: ["0"])
    out = gcr.run_matrix([{"id": "good_a"}], script, output_root=tmp_path / "out",
                         per_cell_timeout_s=120, max_parallel=1)
    assert out["good_a"]["status"] == "ok"
