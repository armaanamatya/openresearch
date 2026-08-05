"""Pure builder for the durable-controller K8s Job (WS3).

The durable controller is itself a CPU workload — it drives the campaign and
delegates GPU training to fenced cell Jobs — so its manifest is CPU-by
construction: no ``nvidia.com/gpu`` toleration/resources, a CPU nodeSelector,
and a fenced name embedding the stable ``fence_epoch`` so two controller
generations can never collide. No cloud SDK call of any kind lives here; the
returned dict is materialised by the caller via ``create_namespaced_job``.

Design: ``docs/history/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`` §3.1.
"""

from __future__ import annotations


def _cpu_node_selector(label: str) -> dict[str, str]:
    """Parse a ``"key=value"`` node-pool label into a nodeSelector dict."""
    key, _, value = label.partition("=")
    if not key.strip() or not value.strip():
        raise ValueError("cpu_pool_label must use the non-empty 'key=value' form")
    return {key: value}


def build_controller_job_manifest(
    *,
    paper: str,
    project_id: str,
    fence_epoch: int,
    image: str,
    cpu_pool_label: str,
    namespace: str,
    service_account: str,
    env: dict[str, str],
    command: list[str],
    backoff_limit: int = 3,
    pod_labels: dict[str, str] | None = None,
    secret_provider_class: str | None = None,
    secret_mount_path: str = "/mnt/orchestrator-secrets",
    runs_pvc_name: str | None = None,
    runs_mount_path: str = "/mnt/reprolab",
) -> dict:
    """Return a CPU-only, fenced ``batch/v1`` Job manifest for the controller Pod.

    ``command`` is the argv the Pod runs (``run_controller.build_controller_command``);
    ``env`` carries the run's config/credentials plus the fence stamp
    (``OPENRESEARCH_CELL_FENCE_EPOCH``) the in-Pod campaign reads to fence its
    cell Jobs. ``backoff_limit`` gives K8s-native restart-on-crash; a restarted
    Pod reacquires the same lease (stable ``owner_id``), preserving ``fence_epoch``.
    """
    forbidden = [
        name
        for name in env
        if "API_KEY" in name
        or name.endswith("_TOKEN")
        or name.endswith("_PASSWORD")
        or name == "OPENRESEARCH_DEMO_SECRET"
    ]
    if forbidden:
        raise ValueError(
            "controller Job env must not embed credential values; use the CSI "
            f"secret volume instead (forbidden names: {sorted(forbidden)})"
        )

    job_name = f"controller-{project_id}-fe{fence_epoch}"
    env_list = [{"name": k, "value": str(v)} for k, v in sorted(env.items())]
    template_labels = {"app": "reprolab-controller", **(pod_labels or {})}
    volume_mounts: list[dict[str, object]] = []
    volumes: list[dict[str, object]] = []
    if runs_pvc_name:
        volume_mounts.append({"name": "controller-state", "mountPath": runs_mount_path})
        volumes.append(
            {
                "name": "controller-state",
                "persistentVolumeClaim": {"claimName": runs_pvc_name},
            }
        )
    if secret_provider_class:
        volume_mounts.append(
            {
                "name": "orchestrator-secrets",
                "mountPath": secret_mount_path,
                "readOnly": True,
            }
        )
        volumes.append(
            {
                "name": "orchestrator-secrets",
                "csi": {
                    "driver": "secrets-store.csi.k8s.io",
                    "readOnly": True,
                    "volumeAttributes": {
                        "secretProviderClass": secret_provider_class,
                    },
                },
            }
        )

    container: dict[str, object] = {
        "name": "controller",
        "image": image,
        "command": command,
        "env": env_list,
        "resources": {"requests": {"cpu": "2", "memory": "8Gi"}},
    }
    if volume_mounts:
        container["volumeMounts"] = volume_mounts

    pod_spec: dict[str, object] = {
        "serviceAccountName": service_account,
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 30,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "fsGroupChangePolicy": "OnRootMismatch",
        },
        "nodeSelector": _cpu_node_selector(cpu_pool_label),
        "containers": [container],
    }
    if volumes:
        pod_spec["volumes"] = volumes

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "app": "reprolab-controller",
                "reprolab/project": project_id,
                "reprolab/fence-epoch": str(fence_epoch),
                "reprolab-generation": str(fence_epoch),
            },
        },
        "spec": {
            "backoffLimit": backoff_limit,
            "template": {
                "metadata": {"labels": template_labels},
                "spec": pod_spec,
            },
        },
    }
