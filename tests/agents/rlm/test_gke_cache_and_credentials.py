"""GKE cell-Job cache volume + credential injection (defects 1 and 2).

DEFECT 1 (P0, pure money) -- every GPU cell re-downloaded multi-GB model weights.
``gcp_files_cache_enabled`` defaulted FALSE, so ``_cache_volume_spec`` returned an
ephemeral ``emptyDir``; the in-pod entrypoint points HF_HOME at it. emptyDir is
per-pod and destroyed on exit, so EVERY cell -- and every re-run -- re-pulled
weights + datasets + pip wheels while an A100 metered. A 6-cell grid paid the
model download 6 times. Fix: the persistent PVC is the DEFAULT, and when no cache
is actually provisioned we say so LOUDLY instead of silently degrading.

DEFECT 2 (P0, capability) -- cell pods had no HF_TOKEN, so gated/private datasets
were unreachable. Nothing in the GKE path imported the ``CredentialBroker`` seam.
A gated dataset therefore failed in-pod as an uncontrolled training error AFTER
paying pod + schedule cost. Fix: inject credentials via CredentialBroker (the
canonical resolver), and NEVER let a secret value reach a log, an event, or an
artifact.

All tests are hermetic: no real K8s, no real GCS, no network, no real secrets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

import backend.agents.rlm.k8s_job_cell_runner as kjcr
from backend.agents.rlm.k8s_job_cell_runner import run_matrix
from backend.services.runtime.credential_broker import CredentialBroker

from .test_k8s_job_cell_runner import (  # reuse the proven hermetic doubles
    FakeK8sBatch,
    FakeK8sCore,
    _FakePod,
    _patch_blob,
    _succeeded_job,
)
from backend.agents.rlm.k8s_job_cell_runner import _K8sClients

# A canary secret. If this string EVER appears in a log record, an emitted event,
# or an on-disk artifact, we have leaked a credential.
_CANARY = "hf_SUPERSECRETCANARY_do_not_leak"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _CoreWithPvc(FakeK8sCore):
    """CoreV1Api double whose PVC lookup is configurable.

    ``pvc_exists=True``  -> the shared cache PVC is provisioned (the happy path).
    ``pvc_exists=False`` -> `get` raises, exactly as a real 404 / RBAC-403 would.
    """

    def __init__(self, *, pvc_exists: bool, **kw: Any) -> None:
        super().__init__(**kw)
        self._pvc_exists = pvc_exists
        self.pvc_get_calls = 0

    def read_namespaced_persistent_volume_claim(self, name: str, namespace: str) -> Any:
        self.pvc_get_calls += 1
        if not self._pvc_exists:
            raise RuntimeError(f'persistentvolumeclaims "{name}" not found')
        return object()


_GCP_SETTINGS: dict[str, Any] = {
    "gcp_namespace": "reprolab",
    "gcp_service_account": "reprolab-sa",
    "gcp_node_pool_name": "gpua100",
    "gcp_base_image": "us-docker.pkg.dev/proj/repo/gke-cell-base:v1",
    "gcp_gcs_bucket": "my-bucket",
    "gcp_max_nodes": 4,
    "gcp_gpu_usd_per_hour": 3.93,
    "gcp_pending_timeout_seconds": 900,
    "gcp_gpu_skus": ["gcp_a100_80", "gcp_a100_80x8"],
    "gcp_ttl_seconds_after_finished": 3600,
    "gcp_job_backoff_limit": 0,
    "gcp_cache_mount_path": "/mnt/reprolab-cache",
    "gcp_watch_poll_interval_s": 0.001,
    "gcp_files_share": "reprolab-cache",
    "gcp_files_cache_enabled": True,   # the NEW default
    "gcp_use_spot": False,
    "gcp_spot_backoff_limit": 3,
    "gcp_cell_preempt_grace_s": 20,
    "gcp_bootstrap_pip_timeout_s": 600,
    "dynamic_gpu_max_escalations": 2,
}


def _run_gke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pvc_exists: bool,
    settings: dict[str, Any] | None = None,
    events: list[tuple[str, dict]] | None = None,
) -> tuple[dict, _CoreWithPvc, FakeK8sBatch]:
    """Drive run_matrix once through the GCP path; return (results, core, batch)."""
    cfg = dict(_GCP_SETTINGS)
    cfg.update(settings or {})
    monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: cfg.get(name, default))

    batch = FakeK8sBatch(_succeeded_job())
    core = _CoreWithPvc(pvc_exists=pvc_exists, pods=[_FakePod(exit_code=0)])
    kjcr._k8s_clients_override = _K8sClients(batch=batch, core=core, watch_cls=None)
    _patch_blob(monkeypatch, metrics={"metric": 0.5})

    sink = (lambda t, p: events.append((t, p))) if events is not None else (lambda t, p: None)

    with kjcr._bind_settings_prefix("gcp"), kjcr.bind_run_context(event_sink=sink):
        results = run_matrix(
            [{"id": "c0"}], tmp_path / "train_cell.py", output_root=tmp_path / "out"
        )
    return results, core, batch


def _volume(batch: FakeK8sBatch) -> dict:
    return batch.created_jobs[0]["spec"]["template"]["spec"]["volumes"][0]


def _env_map(batch: FakeK8sBatch) -> dict[str, str]:
    env = batch.created_jobs[0]["spec"]["template"]["spec"]["containers"][0]["env"]
    return {e["name"]: e["value"] for e in env}


# ---------------------------------------------------------------------------
# DEFECT 1 -- the cache volume
# ---------------------------------------------------------------------------

class TestPersistentCacheIsTheDefault:
    def test_config_default_enables_the_persistent_cache(self, monkeypatch):
        """The money fix: default ON. A cost fix that ships default-OFF saves nothing.

        Self-isolating: backend/config.py sets ``env_file=".env"`` AND a pytest
        plugin (deepeval) calls ``load_dotenv()`` on the repo ``.env`` at session
        start -- so the developer's ``.env`` lands in BOTH the Settings file source
        and ``os.environ``. ``_env_file=None`` only closes the first door; without
        the delenv this would assert the ambient environment rather than the
        shipped code default (and could pass while the real default is broken).
        Both prefixes, because config._apply_legacy_env_aliases bridges
        REPROLAB_* <-> OPENRESEARCH_*.
        """
        for name in (
            "OPENRESEARCH_GCP_FILES_CACHE_ENABLED",
            "REPROLAB_GCP_FILES_CACHE_ENABLED",
        ):
            monkeypatch.delenv(name, raising=False)
        import backend.config as _config
        monkeypatch.setattr(_config, "_settings_cache", None, raising=False)

        assert _config.Settings(_env_file=None).gcp_files_cache_enabled is True

    def test_cache_volume_is_a_persistent_claim_by_default(self, monkeypatch, tmp_path):
        """With the PVC provisioned, the cell mounts it -- weights download ONCE."""
        _results, _core, batch = _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        vol = _volume(batch)
        assert "persistentVolumeClaim" in vol, (
            f"cell cache must be a persistent claim, got {vol!r} -- an emptyDir is "
            f"destroyed per-pod, so every cell re-downloads multi-GB weights"
        )
        assert vol["persistentVolumeClaim"]["claimName"] == "reprolab-cache"
        assert "emptyDir" not in vol

    def test_cache_volume_is_mounted_at_the_cache_mount_path(self, monkeypatch, tmp_path):
        _results, _core, batch = _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        container = batch.created_jobs[0]["spec"]["template"]["spec"]["containers"][0]
        mount = container["volumeMounts"][0]
        assert mount["name"] == "reprolab-cache"
        assert mount["mountPath"] == "/mnt/reprolab-cache"

    def test_pvc_is_checked_once_per_run_not_once_per_cell(self, monkeypatch, tmp_path):
        """Scale: the live check must not add an API call per cell."""
        cfg = dict(_GCP_SETTINGS)
        monkeypatch.setattr(kjcr, "_setting", lambda n, d=None: cfg.get(n, d))
        batch = FakeK8sBatch(_succeeded_job())
        core = _CoreWithPvc(pvc_exists=True, pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = _K8sClients(batch=batch, core=core, watch_cls=None)
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        cells = [{"id": f"c{i}"} for i in range(6)]
        with kjcr._bind_settings_prefix("gcp"), kjcr.bind_run_context():
            run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert core.pvc_get_calls == 1, (
            f"expected ONE PVC existence check per run, got {core.pvc_get_calls}"
        )


class TestMissingCacheIsLoudNotSilent:
    def test_missing_pvc_emits_a_loud_run_warning(self, monkeypatch, tmp_path):
        """The honesty requirement: no silent emptyDir fallback."""
        events: list[tuple[str, dict]] = []
        _run_gke(monkeypatch, tmp_path, pvc_exists=False, events=events)

        warnings = [p for t, p in events if t == "run_warning"]
        codes = {p.get("code") for p in warnings}
        assert "persistent_cache_unavailable" in codes, (
            f"a missing cache provision must emit a loud run_warning, got {codes!r}"
        )

    def test_the_warning_names_the_cost_impact(self, monkeypatch, tmp_path):
        """A warning that doesn't say 'this costs you money' gets ignored."""
        events: list[tuple[str, dict]] = []
        _run_gke(monkeypatch, tmp_path, pvc_exists=False, events=events)

        msg = next(
            p["message"] for t, p in events
            if t == "run_warning" and p.get("code") == "persistent_cache_unavailable"
        ).lower()
        assert "re-download" in msg
        assert "cost" in msg
        assert "emptydir" in msg
        # ...and it must tell the operator how to actually fix it.
        assert "filestore" in msg

    def test_missing_pvc_still_falls_back_to_emptydir(self, monkeypatch, tmp_path):
        """Loud, but not fatal: referencing a nonexistent claim would strand the pod
        Pending forever, which is strictly worse than a slow emptyDir."""
        _results, _core, batch = _run_gke(monkeypatch, tmp_path, pvc_exists=False)
        vol = _volume(batch)
        assert "emptyDir" in vol
        assert "persistentVolumeClaim" not in vol

    def test_missing_pvc_does_not_fail_the_cell(self, monkeypatch, tmp_path):
        results, _core, _batch = _run_gke(monkeypatch, tmp_path, pvc_exists=False)
        assert results["c0"]["status"] == "ok"

    def test_deliberate_opt_out_is_silent(self, monkeypatch, tmp_path):
        """files_cache_enabled=false is an INFORMED choice -- don't nag about it."""
        events: list[tuple[str, dict]] = []
        _results, core, batch = _run_gke(
            monkeypatch, tmp_path, pvc_exists=False,
            settings={"gcp_files_cache_enabled": False}, events=events,
        )
        codes = {p.get("code") for t, p in events if t == "run_warning"}
        assert "persistent_cache_unavailable" not in codes
        assert core.pvc_get_calls == 0, "opted out => don't even probe the API"
        assert "emptyDir" in _volume(batch)


# ---------------------------------------------------------------------------
# DEFECT 2 -- HF_TOKEN reaches the pod, and never leaks
# ---------------------------------------------------------------------------

class TestCredentialsReachTheCellPod:
    def test_hf_token_is_injected_into_the_cell_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", _CANARY)
        _results, _core, batch = _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        env = _env_map(batch)
        assert env.get("HF_TOKEN") == _CANARY, (
            "HF_TOKEN must reach the cell pod, else a gated/private dataset is "
            "unreachable and fails in-pod AFTER paying pod + schedule cost"
        )

    def test_hf_token_resolves_through_the_alternate_alias(self, monkeypatch, tmp_path):
        """CredentialBroker registers HUGGING_FACE_HUB_TOKEN as an HF_TOKEN alias;
        it must still land under the canonical name the HF libs read."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", _CANARY)
        _results, _core, batch = _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        assert _env_map(batch).get("HF_TOKEN") == _CANARY

    def test_no_credential_env_when_nothing_is_configured(self, monkeypatch, tmp_path):
        """Byte-identical env when no secret is configured (today's common case)."""
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_KEY",
                     "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(name, raising=False)
        _results, _core, batch = _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        env = _env_map(batch)
        assert "HF_TOKEN" not in env
        assert "KAGGLE_KEY" not in env

    def test_llm_provider_keys_are_never_injected(self, monkeypatch, tmp_path):
        """A cell pod runs train_cell.py, never an LLM call -- injecting the
        Anthropic/OpenAI keys would only widen the secret blast radius."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-travel")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-should-not-travel")
        _results, _core, batch = _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        env = _env_map(batch)
        assert "ANTHROPIC_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env
        assert "sk-ant-should-not-travel" not in json.dumps(batch.created_jobs)

    def test_broker_is_the_resolver(self, monkeypatch, tmp_path):
        """Use the CredentialBroker seam -- do not hand-roll a second secret path."""
        assert CredentialBroker(env={"HF_TOKEN": _CANARY}).resolve_env(["hf_token"]) == [
            ("HF_TOKEN", _CANARY)
        ]


class TestNoSecretLeaks:
    """The one that matters most: a secret must never reach a log, an event, or disk."""

    def test_secret_never_appears_in_emitted_events(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", _CANARY)
        events: list[tuple[str, dict]] = []
        _run_gke(monkeypatch, tmp_path, pvc_exists=False, events=events)  # warning path too
        blob = json.dumps(events)
        assert _CANARY not in blob, "secret leaked into an emitted SSE event"

    def test_secret_never_appears_in_logs(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("HF_TOKEN", _CANARY)
        with caplog.at_level(logging.DEBUG):
            _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        assert _CANARY not in caplog.text, "secret leaked into a log record"

    def test_injection_logs_only_the_names(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("HF_TOKEN", _CANARY)
        with caplog.at_level(logging.INFO):
            _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        assert "HF_TOKEN" in caplog.text          # the NAME is fine (and useful)
        assert _CANARY not in caplog.text         # the VALUE is not

    def test_secret_never_appears_in_on_disk_artifacts(self, monkeypatch, tmp_path):
        """cell_manifest.json / metrics.json / logs are all persisted -- sweep them."""
        monkeypatch.setenv("HF_TOKEN", _CANARY)
        _run_gke(monkeypatch, tmp_path, pvc_exists=True)

        scanned = 0
        for path in tmp_path.rglob("*"):
            if not path.is_file():
                continue
            scanned += 1
            assert _CANARY not in path.read_text(errors="replace"), (
                f"secret leaked into on-disk artifact {path}"
            )
        assert scanned > 0, "sweep found no artifacts -- the test would vacuously pass"

    def test_secret_never_appears_in_the_cell_result(self, monkeypatch, tmp_path):
        """CellResult.error is persisted into cell_manifest.json."""
        monkeypatch.setenv("HF_TOKEN", _CANARY)
        results, _core, _batch = _run_gke(monkeypatch, tmp_path, pvc_exists=True)
        assert _CANARY not in json.dumps(results)

    def test_submit_failure_message_is_redacted(self, monkeypatch, tmp_path):
        """A K8s API-server validation error can echo request fields back. The
        manifest now carries HF_TOKEN, so the error string must be redacted before
        it reaches CellResult.error (which IS written to cell_manifest.json)."""
        monkeypatch.setenv("HF_TOKEN", _CANARY)
        cfg = dict(_GCP_SETTINGS)
        monkeypatch.setattr(kjcr, "_setting", lambda n, d=None: cfg.get(n, d))

        class _EchoingBatch(FakeK8sBatch):
            def create_namespaced_job(self, namespace: str, body: dict) -> None:
                # Simulate an API server echoing the offending manifest back.
                raise RuntimeError(f"Job rejected; env was HF_TOKEN={_CANARY}")

        batch = _EchoingBatch(_succeeded_job())
        core = _CoreWithPvc(pvc_exists=True, pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = _K8sClients(batch=batch, core=core, watch_cls=None)
        _patch_blob(monkeypatch, metrics=None)

        with kjcr._bind_settings_prefix("gcp"), kjcr.bind_run_context():
            results = run_matrix(
                [{"id": "c0"}], tmp_path / "train_cell.py", output_root=tmp_path / "out"
            )

        assert results["c0"]["status"] == "error"
        assert _CANARY not in json.dumps(results), "secret leaked via the submit-error path"
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert _CANARY not in path.read_text(errors="replace")

    def test_redact_env_drops_the_injected_secret(self, monkeypatch):
        """The broker's redaction contract, applied to the exact env we inject."""
        injected = {e["name"]: e["value"] for e in [{"name": "HF_TOKEN", "value": _CANARY}]}
        injected["OPENRESEARCH_CELL_ID"] = "c0"
        red = CredentialBroker.redact_env(injected)
        assert _CANARY not in json.dumps(red)
        assert red == {"OPENRESEARCH_CELL_ID": "c0"}
