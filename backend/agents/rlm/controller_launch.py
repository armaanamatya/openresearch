"""Pure builder for the durable-controller K8s Job (WS3).

The durable controller is itself a CPU workload — it drives the campaign and
delegates GPU training to fenced cell Jobs — so its manifest is CPU-by
construction: no ``nvidia.com/gpu`` toleration/resources, a CPU nodeSelector,
and a fenced name embedding the stable ``fence_epoch`` so two controller
generations can never collide. No cloud SDK call of any kind lives here; the
returned dict is materialised by the caller via ``create_namespaced_job``.

Design: ``docs/superpowers/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`` §3.1.
"""

from __future__ import annotations


def _cpu_node_selector(label: str) -> dict[str, str]:
    """Parse a ``"key=value"`` node-pool label into a nodeSelector dict."""
    key, _, value = label.partition("=")
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
) -> dict:
    """Return a CPU-only, fenced ``batch/v1`` Job manifest for the controller Pod.

    ``command`` is the argv the Pod runs (``run_controller.build_controller_command``);
    ``env`` carries the run's config/credentials plus the fence stamp
    (``OPENRESEARCH_CELL_FENCE_EPOCH``) the in-Pod campaign reads to fence its
    cell Jobs. ``backoff_limit`` gives K8s-native restart-on-crash; a restarted
    Pod reacquires the same lease (stable ``owner_id``), preserving ``fence_epoch``.
    """
    job_name = f"controller-{project_id}-fe{fence_epoch}"
    env_list = [{"name": k, "value": str(v)} for k, v in sorted(env.items())]
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
            },
        },
        "spec": {
            "backoffLimit": backoff_limit,
            "template": {
                "metadata": {"labels": {"app": "reprolab-controller"}},
                "spec": {
                    "serviceAccountName": service_account,
                    "restartPolicy": "Never",
                    "nodeSelector": _cpu_node_selector(cpu_pool_label),
                    "containers": [
                        {
                            "name": "controller",
                            "image": image,
                            "command": command,
                            "env": env_list,
                            "resources": {"requests": {"cpu": "2", "memory": "8Gi"}},
                        }
                    ],
                },
            },
        },
    }
