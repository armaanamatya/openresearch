"""Takeover-safe durable-controller submit + sweeper (WS3).

All logic here is pure over an injected ``ControllerCluster`` double — no GCS,
no K8s, no network. Covers the Codex-flagged correctness cases: contended
adopt, submit ordering (submit→ready→reap), not-ready-with-confirmed-delete vs
stuck, and the split-brain-safe sweeper (distinct owner, expired-only).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.rlm import controller_cluster as cc


class _Token:
    def __init__(self, fence_epoch=1, generation=1, acquired_epoch=100.0, owner_id="prj_x"):
        self.fence_epoch = fence_epoch
        self.generation = generation
        self.acquired_epoch = acquired_epoch
        self.owner_id = owner_id


class _FakeCluster:
    """Records the call order and lets each step's outcome be configured."""

    def __init__(self, *, token=None, current=True, ready=True, delete_ok=True,
                 acquire_raises=False, submit_raises=False):
        self._token = token if token is not None else _Token()
        self._current = current
        self._ready = ready
        self._delete_ok = delete_ok
        self._acquire_raises = acquire_raises
        self._submit_raises = submit_raises
        self.calls: list[str] = []
        self.submitted_manifest = None
        self.reaped = False

    def now(self):
        return 100.0

    def acquire(self, project_id, owner_id, now_epoch):
        self.calls.append("acquire")
        if self._acquire_raises:
            raise RuntimeError("gcs unreachable")
        return self._token

    def is_current(self, token):
        self.calls.append("is_current")
        return self._current

    def submit(self, manifest):
        self.calls.append("submit")
        if self._submit_raises:
            raise RuntimeError("k8s submit failed")
        self.submitted_manifest = manifest

    def wait_ready(self, job_name, *, timeout_s):
        self.calls.append("wait_ready")
        return self._ready

    def delete_confirmed(self, job_name):
        self.calls.append("delete")
        return self._delete_ok

    def reap(self, project_id, token):
        self.calls.append("reap")
        self.reaped = True
        return 0


def _mf(fence):
    return {"metadata": {"name": f"controller-prj_x-fe{fence}"},
            "_fence": fence}


# ---------------------------------------------------------------------------
# submit_controller — happy path + ordering
# ---------------------------------------------------------------------------

def test_happy_path_orders_submit_then_ready_then_reap():
    cluster = _FakeCluster(token=_Token(fence_epoch=2, acquired_epoch=100.0))
    handle = cc.submit_controller(
        cluster, build_manifest=_mf, project_id="prj_x", owner_id="prj_x",
        ready_timeout_s=180.0,
    )
    assert handle == {"jobName": "controller-prj_x-fe2", "fenceEpoch": 2, "submittedEpoch": 100.0}
    # reap must come AFTER submit and ready (never before the successor is up).
    assert cluster.calls.index("submit") < cluster.calls.index("wait_ready") < cluster.calls.index("reap")
    assert cluster.reaped is True


def test_manifest_built_with_acquired_fence_epoch():
    cluster = _FakeCluster(token=_Token(fence_epoch=7))
    cc.submit_controller(cluster, build_manifest=_mf, project_id="prj_x",
                         owner_id="prj_x", ready_timeout_s=180.0)
    assert cluster.submitted_manifest["_fence"] == 7   # fence came from the lease, post-acquire


# ---------------------------------------------------------------------------
# contended — another driver owns it: adopt, never submit
# ---------------------------------------------------------------------------

def test_lease_none_returns_none_no_submit():
    cluster = _FakeCluster(token=None)
    cluster._token = None  # acquire returns None
    handle = cc.submit_controller(cluster, build_manifest=_mf, project_id="prj_x",
                                  owner_id="prj_x", ready_timeout_s=180.0)
    assert handle is None
    assert "submit" not in cluster.calls


def test_lost_is_current_between_acquire_and_use_returns_none():
    cluster = _FakeCluster(current=False)   # acquired but immediately superseded
    handle = cc.submit_controller(cluster, build_manifest=_mf, project_id="prj_x",
                                  owner_id="prj_x", ready_timeout_s=180.0)
    assert handle is None
    assert "submit" not in cluster.calls


# ---------------------------------------------------------------------------
# not ready — confirmed delete (safe) vs stuck (never fall back)
# ---------------------------------------------------------------------------

def test_not_ready_confirmed_delete_raises_not_ready():
    cluster = _FakeCluster(ready=False, delete_ok=True)
    with pytest.raises(cc.ControllerNotReady):
        cc.submit_controller(cluster, build_manifest=_mf, project_id="prj_x",
                             owner_id="prj_x", ready_timeout_s=180.0)
    assert "delete" in cluster.calls and not cluster.reaped


def test_not_ready_unconfirmed_delete_raises_stuck():
    cluster = _FakeCluster(ready=False, delete_ok=False)
    with pytest.raises(cc.ControllerStuck):
        cc.submit_controller(cluster, build_manifest=_mf, project_id="prj_x",
                             owner_id="prj_x", ready_timeout_s=180.0)


def test_submit_error_with_confirmed_delete_is_retryable_remote_failure():
    cluster = _FakeCluster(submit_raises=True, delete_ok=True)
    with pytest.raises(cc.ControllerNotReady):
        cc.submit_controller(cluster, build_manifest=_mf, project_id="prj_x",
                             owner_id="prj_x", ready_timeout_s=180.0)
    assert "wait_ready" not in cluster.calls
    assert "delete" in cluster.calls


def test_submit_error_without_confirmed_delete_fails_closed():
    cluster = _FakeCluster(submit_raises=True, delete_ok=False)
    with pytest.raises(cc.ControllerStuck):
        cc.submit_controller(cluster, build_manifest=_mf, project_id="prj_x",
                             owner_id="prj_x", ready_timeout_s=180.0)


# ---------------------------------------------------------------------------
# sweeper — split-brain safe: distinct owner, only acts on acquirable (expired)
# ---------------------------------------------------------------------------

def test_sweeper_resubmits_when_lease_acquirable():
    cluster = _FakeCluster(token=_Token(fence_epoch=3))
    done = cc.sweep_orphaned_controllers(
        cluster,
        list_durable_runs=lambda: ["prj_x"],
        build_manifest_for=lambda pid, owner_id: _mf,
        ready_timeout_s=180.0,
        sweeper_owner="sweeper",
    )
    assert done == ["prj_x"]


def test_sweeper_skips_live_controller_when_lease_held():
    # acquire returns None → the live controller still holds the lease (different owner,
    # not expired). The sweeper must NOT submit — no takeover of a healthy controller.
    cluster = _FakeCluster()
    cluster._token = None
    done = cc.sweep_orphaned_controllers(
        cluster,
        list_durable_runs=lambda: ["prj_x"],
        build_manifest_for=lambda pid, owner_id: _mf,
        ready_timeout_s=180.0,
        sweeper_owner="sweeper",
    )
    assert done == []
    assert "submit" not in cluster.calls


def test_sweeper_is_fail_soft_per_run():
    cluster = _FakeCluster(ready=False, delete_ok=False)  # every submit ends Stuck
    done = cc.sweep_orphaned_controllers(
        cluster,
        list_durable_runs=lambda: ["prj_a", "prj_b"],
        build_manifest_for=lambda pid, owner_id: _mf,
        ready_timeout_s=180.0,
        sweeper_owner="sweeper",
    )
    assert done == []   # both failed, but the sweep did not raise


def test_sweeper_threads_a_fresh_owner_into_each_replacement_manifest():
    owners: list[str] = []
    cluster = _FakeCluster(token=_Token(fence_epoch=3))

    def build(project_id, owner_id):
        owners.append(owner_id)
        return _mf

    done = cc.sweep_orphaned_controllers(
        cluster,
        list_durable_runs=lambda: ["prj_a", "prj_b"],
        build_manifest_for=build,
        ready_timeout_s=180.0,
        sweeper_owner="sweeper",
    )
    assert done == ["prj_a", "prj_b"]
    assert len(set(owners)) == 2
    assert all(owner.startswith("sweeper-") for owner in owners)


def test_k8s_cluster_lazy_loads_shared_batch_api(monkeypatch):
    from backend.services.runtime import k8s_job_backend

    sentinel = object()
    calls = []
    monkeypatch.setattr(
        k8s_job_backend,
        "_load_kubernetes_batch_api",
        lambda: calls.append("load") or sentinel,
    )
    cluster = cc.K8sControllerCluster(
        lease=SimpleNamespace(), namespace="reprolab"
    )

    assert cluster._batch() is sentinel
    assert cluster._batch() is sentinel
    assert calls == ["load"]


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("Pending", False), ("Running", True), ("Succeeded", True)],
)
def test_k8s_controller_readiness_requires_started_pod(phase, expected):
    class Batch:
        def read_namespaced_job_status(self, **kwargs):
            return SimpleNamespace(
                status=SimpleNamespace(succeeded=0, conditions=[])
            )

    class Core:
        def list_namespaced_pod(self, **kwargs):
            return SimpleNamespace(
                items=[SimpleNamespace(status=SimpleNamespace(phase=phase))]
            )

    ticks = iter([0.0, 0.0, 1.0])
    cluster = cc.K8sControllerCluster(
        lease=SimpleNamespace(),
        namespace="reprolab",
        batch_api=Batch(),
        core_api=Core(),
        clock=lambda: next(ticks),
        sleep=lambda _: None,
    )
    assert cluster.wait_ready("controller-prj-fe1", timeout_s=0.5) is expected


def test_k8s_reaper_accepts_controller_and_cell_fence_labels():
    deleted: list[str] = []

    class Lease:
        def reap_stale_fence_epochs(
            self, project_id, token, *, list_jobs, delete_job
        ):
            jobs = list_jobs(project_id)
            for name, fence in jobs:
                if fence < token.fence_epoch:
                    delete_job(name)
            return len(deleted)

    jobs = SimpleNamespace(
        items=[
            SimpleNamespace(
                metadata=SimpleNamespace(
                    name="controller-old", labels={"reprolab-generation": "1"}
                )
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    name="cell-old", labels={"reprolab/fence-epoch": "2"}
                )
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    name="current", labels={"reprolab-generation": "3"}
                )
            ),
        ]
    )

    class Batch:
        def list_namespaced_job(self, **kwargs):
            return jobs

        def delete_namespaced_job(self, *, name, **kwargs):
            deleted.append(name)

    cluster = cc.K8sControllerCluster(
        lease=Lease(), namespace="reprolab", batch_api=Batch()
    )
    count = cluster.reap("prj_x", SimpleNamespace(fence_epoch=3))
    assert count == 2
    assert deleted == ["controller-old", "cell-old"]
