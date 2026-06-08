"""BUG-NEW-045: the RunStatus Literal must include the out-of-band terminal
states "killed" (CLI SIGTERM handler) and "interrupted" (run_liveness sweep),
or _load_run 500s on /runs/latest & /runs/{id}. leaderboard._TERMINAL_STATUSES
already lists both — this guards the producer side."""
from typing import get_args

from backend.services.events.live_runs import RunStatus


def test_runstatus_includes_terminal_states():
    members = set(get_args(RunStatus))
    assert {"killed", "interrupted"} <= members, members
    # the original five must remain
    assert {"queued", "running", "stopped", "completed", "failed"} <= members


def test_runstatus_matches_leaderboard_terminal_set():
    from backend.routes.leaderboard import _TERMINAL_STATUSES

    # every status the leaderboard treats as terminal must be a valid RunStatus
    assert _TERMINAL_STATUSES <= set(get_args(RunStatus))
