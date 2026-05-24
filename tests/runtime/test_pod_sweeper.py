"""Tests for backend.services.runtime.pod_sweeper.

Pins the sweeper's contract:
  * is_stale honours max_age_seconds and falls back safely on None age
  * sweep_stale_pods + dry_run never DELETEs
  * preserve_pod_ids pins runs so a routine sweep can't kill them
  * delete_pod fail-soft on every error path
  * Estimated savings tracker accumulates costPerHr only for swept pods
"""

from __future__ import annotations

from backend.services.runtime import pod_sweeper as ps


# ---------------------------------------------------------------------------
# PodInfo.is_stale
# ---------------------------------------------------------------------------


def test_is_stale_above_threshold() -> None:
    pod = ps.PodInfo(
        id="x", name="x", status="RUNNING", created_at="",
        age_seconds=7500, cost_per_hour=0.8, gpu_type="L40S",
    )
    assert pod.is_stale(max_age_seconds=7200) is True


def test_is_stale_at_threshold_inclusive() -> None:
    pod = ps.PodInfo(
        id="x", name="x", status="RUNNING", created_at="",
        age_seconds=7200, cost_per_hour=0.8, gpu_type="L40S",
    )
    # Strictly greater than threshold → not stale at the boundary.
    assert pod.is_stale(max_age_seconds=7200) is False


def test_is_stale_none_age_returns_false() -> None:
    pod = ps.PodInfo(
        id="x", name="x", status="RUNNING", created_at="",
        age_seconds=None, cost_per_hour=None, gpu_type="?",
    )
    assert pod.is_stale(max_age_seconds=0) is False


# ---------------------------------------------------------------------------
# sweep_stale_pods orchestration (using monkeypatched list_pods + delete_pod)
# ---------------------------------------------------------------------------


def _fake_pods(*ages: int | None, base_cost: float = 0.34) -> list[ps.PodInfo]:
    return [
        ps.PodInfo(
            id=f"p{i}", name=f"pod{i}", status="RUNNING", created_at="",
            age_seconds=age, cost_per_hour=base_cost, gpu_type="L40S",
        )
        for i, age in enumerate(ages)
    ]


def test_sweep_dry_run_does_not_delete(monkeypatch) -> None:
    monkeypatch.setattr(ps, "list_pods", lambda **_kw: _fake_pods(8000, 9000))
    delete_calls: list[str] = []
    monkeypatch.setattr(ps, "delete_pod", lambda pid, **_kw: delete_calls.append(pid) or True)

    report = ps.sweep_stale_pods(max_age_seconds=7200, dry_run=True)
    assert delete_calls == []  # no DELETEs in dry_run
    assert sorted(report.swept) == ["p0", "p1"]
    assert report.errors == []
    assert report.estimated_savings_per_hour > 0


def test_sweep_skips_young_pods(monkeypatch) -> None:
    monkeypatch.setattr(ps, "list_pods", lambda **_kw: _fake_pods(60, 9000))
    monkeypatch.setattr(ps, "delete_pod", lambda pid, **_kw: True)

    report = ps.sweep_stale_pods(max_age_seconds=7200)
    assert report.swept == ["p1"]
    assert len(report.skipped) == 1
    assert report.skipped[0][0] == "p0"


def test_sweep_preserves_pinned_ids(monkeypatch) -> None:
    monkeypatch.setattr(ps, "list_pods", lambda **_kw: _fake_pods(9000, 8500))
    monkeypatch.setattr(ps, "delete_pod", lambda pid, **_kw: True)

    report = ps.sweep_stale_pods(max_age_seconds=7200, preserve_pod_ids=("p0",))
    assert report.swept == ["p1"]
    assert ("p0", "preserved by caller") in report.skipped


def test_sweep_records_delete_failure(monkeypatch) -> None:
    monkeypatch.setattr(ps, "list_pods", lambda **_kw: _fake_pods(9000))
    monkeypatch.setattr(ps, "delete_pod", lambda pid, **_kw: False)

    report = ps.sweep_stale_pods(max_age_seconds=7200)
    assert report.swept == []
    assert len(report.errors) == 1
    assert report.errors[0][0] == "p0"


def test_sweep_no_pods(monkeypatch) -> None:
    monkeypatch.setattr(ps, "list_pods", lambda **_kw: [])
    report = ps.sweep_stale_pods()
    assert report.total_pods == 0
    assert report.summary().startswith("0/0 swept")


def test_sweep_estimated_savings_aggregates_swept_only(monkeypatch) -> None:
    pods = [
        ps.PodInfo(id="cheap", name="x", status="RUNNING", created_at="",
                   age_seconds=9000, cost_per_hour=0.34, gpu_type="4090"),
        ps.PodInfo(id="dear",  name="x", status="RUNNING", created_at="",
                   age_seconds=9000, cost_per_hour=0.86, gpu_type="L40S"),
        ps.PodInfo(id="young", name="x", status="RUNNING", created_at="",
                   age_seconds=60,   cost_per_hour=1.20, gpu_type="A100"),
    ]
    monkeypatch.setattr(ps, "list_pods", lambda **_kw: pods)
    monkeypatch.setattr(ps, "delete_pod", lambda pid, **_kw: True)

    report = ps.sweep_stale_pods(max_age_seconds=7200)
    assert sorted(report.swept) == ["cheap", "dear"]
    # ``young`` was skipped — its cost MUST NOT appear in savings.
    assert report.estimated_savings_per_hour == 0.34 + 0.86


# ---------------------------------------------------------------------------
# _parse_age_seconds
# ---------------------------------------------------------------------------


def test_parse_age_handles_iso_with_z() -> None:
    # 1970-01-01 is decades old → age in seconds is huge but finite.
    out = ps._parse_age_seconds("1970-01-01T00:00:00.000Z")
    assert out is not None and out > 10_000_000


def test_parse_age_handles_offset() -> None:
    out = ps._parse_age_seconds("1970-01-01T00:00:00.000+00:00")
    assert out is not None and out > 10_000_000


def test_parse_age_handles_go_time_with_utc_suffix() -> None:
    # The exact shape RunPod's REST API returns today: Go's time.String()
    # "YYYY-MM-DD HH:MM:SS.fff +0000 UTC".  This shape broke the original
    # parser and made the sweeper silently report 0/0 swept.
    out = ps._parse_age_seconds("1970-01-01 00:00:00.000 +0000 UTC")
    assert out is not None and out > 10_000_000


def test_parse_age_handles_go_time_negative_offset() -> None:
    out = ps._parse_age_seconds("1970-01-01 00:00:00.000 -0500 EST")
    # The trailing tz name is best-effort; the offset is authoritative.
    # Strip "EST" before the offset, parse "-0500" → "-05:00".
    assert out is not None and out > 10_000_000


def test_parse_age_handles_malformed() -> None:
    assert ps._parse_age_seconds("not-a-date") is None
    assert ps._parse_age_seconds(None) is None
    assert ps._parse_age_seconds("") is None


# ---------------------------------------------------------------------------
# list_pods + delete_pod fail-soft on missing API key
# ---------------------------------------------------------------------------


def test_list_pods_no_api_key_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("REPROLAB_RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert ps.list_pods(api_key=None) == []


def test_delete_pod_no_api_key_returns_false(monkeypatch) -> None:
    monkeypatch.delenv("REPROLAB_RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert ps.delete_pod("any-id", api_key=None) is False
