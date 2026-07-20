# GCP/GKE execution backend — shared-K8s-base design

> **Status:** in progress (2026-06-16). Branch `feat/gcp-gke-backend`.
> Sequencing: **code layer first** (this doc), then `infra/gcp` Terraform/Helm (follow-up).
> Architecture decision (user-approved): **shared K8s base + cloud adapters**, NOT a duplicated GkeJobBackend.

## Goal

Make `provider='gcp'` runnable end-to-end alongside Azure, so `SandboxMode.{azure,gcp,runpod,local}`
are true peers. The selection layer (`gpu_catalog` GCP A2 rows + `gpu_resolver.resolve(provider='gcp')`)
already exists. This adds the **execution** layer on the same modular seam.

Both K8s job paths — the `exec` RuntimeBackend (`AksJobBackend`) and the cell-matrix runner
(`k8s_job_cell_runner`) — are ~90% identical to what GKE needs (it is plain Kubernetes). The cloud
deltas are exactly four: **object store** (Blob→GCS), **credential preflight** (DefaultAzureCredential→ADC),
**settings prefix** (`azure_*`→`gcp_*`), and **Workload-Identity binding form** (Azure needs a pod label;
GKE binds via the KSA annotation, cluster-side — no manifest label).

## Invariants to preserve (regression guard)

- The 189 tests in `tests/services/runtime/test_{aks_job_backend,azure_blob,gpu_capacity,gpu_capacity_azure,gpu_resolver_gcp,aks_cell_entrypoint}.py` MUST stay green.
- Back-compat: `aks_job_backend.AksJobBackend`, `.ensure_azure_available`, `._build_exec_job_manifest`, `._safe_name` must remain importable from `backend.services.runtime.aks_job_backend`.
- Blob I/O must stay patchable at the module level (`azure_blob.upload_prefix` etc.) — the Azure backend test patches it there.
- No behaviour change for any existing sandbox under existing config (GCP is purely additive, default sandbox stays `runpod`).

## New / changed files

### NEW `backend/services/runtime/gcs_blob.py` — GCS object store (mirror of `azure_blob.py`)
Same public function names + analogous signatures, `google.cloud.storage` lazily imported (module imports without the SDK; tests inject a duck-typed `client`). `account_name`→unused/`project`, `container_name`→`bucket`. Keep `_validate_blob_name`, the upload_prefix exclude set (`outputs/.git/__pycache__/.venv`, `.pyc`, escaping symlinks), and the bounded `ThreadPoolExecutor`.
- `upload_prefix(local_root, *, blob_prefix, bucket, project=None, client=None) -> list[str]`
- `upload_bytes(data, *, blob_name, bucket, project=None, client=None) -> None`
- `download_artifact(blob_name, destination, *, bucket, project=None, client=None) -> Path`
- `download_bytes(blob_name, *, bucket, project=None, client=None) -> bytes`
- `_make_bucket_client(bucket, project=None) -> Any` (lazy `storage.Client(project).bucket(bucket)`)
- Duck-typed test client contract: `client.blob(name).upload_from_string(data)` / `client.blob(name).download_as_bytes()` (mirror how `azure_blob`'s injected client is shaped; match whatever `test_azure_blob` does, adapted to GCS).

### NEW `backend/services/runtime/k8s_job_backend.py` — shared base + adapters
Holds everything cloud-agnostic, moved out of `aks_job_backend.py`:
- Helpers (moved): `_safe_name`, `_job_name`, `_blob_code_prefix`, `_blob_artifact_key`, `_settings_get`, `_extract_exit_code`.
- `_build_job_manifest(*, ..., pod_template_extra_labels: dict|None=None, sandbox_label: str, node_selector_label: str="reprolab/sku") -> dict` — the generic manifest builder (current `_build_exec_job_manifest` body, with the hardcoded `azure.workload.identity/use` label and `reprolab/sandbox: aks` label parameterized out).
- `class ObjectStore` (Protocol) + `AzureBlobStore(account_name, container_name, client)` and `GcsStore(bucket, project, client)` adapters. Methods: `upload_prefix(local_root, *, blob_prefix)->list[str]`, `upload_bytes(data, *, blob_name)->None`, `download_bytes(blob_name)->bytes`, `download_artifact(blob_name, destination)->Path`. Each delegates to the corresponding module function (`azure_blob.*` / `gcs_blob.*`) so module-level patching still works.
- `@dataclass(frozen=True) class CloudSpec`: `provider:str`, `settings_prefix:str`, `sandbox_prefix:str` (sandbox_id prefix: "aks"/"gke"), `sandbox_label:str` (manifest `reprolab/sandbox` value), `pod_template_extra_labels:dict` (Azure WI label / {} for GKE), `make_object_store: Callable[[settings, client], ObjectStore]`, `ensure_available: Callable[[], None]`, `base_image_setting:str` (e.g. "azure_base_image").
- `class _KubernetesJobBackend(RuntimeBackend)`: the full current `AksJobBackend` body, generalized — reads `f"{self._cloud.settings_prefix}_namespace"` etc. via `_settings_get`; builds the manifest with `pod_template_extra_labels=self._cloud.pod_template_extra_labels`, `sandbox_label=self._cloud.sandbox_label`; routes blob I/O through `self._store` (built lazily from `self._cloud.make_object_store(settings, injected_client)`). `__init__` keeps the exact kwargs `(*, run_budget=None, gpu_plan=None, batch_api=None, core_api=None, blob_client=None, settings=None)`.
- `ensure_k8s_cloud_available(cloud: CloudSpec)`: the shared preflight (SDK imports per cloud, settings non-empty, credential probe, kubeconfig). The cloud-specific SDK/credential checks come from the CloudSpec.

### CHANGED `backend/services/runtime/aks_job_backend.py` — slim to Azure adapter
- `from backend.services.runtime.k8s_job_backend import (_KubernetesJobBackend, _build_job_manifest, _safe_name, _job_name, _extract_exit_code, AzureBlobStore, CloudSpec)` (+ re-export for back-compat).
- Keep `_build_exec_job_manifest(...)` as a thin Azure wrapper → `_build_job_manifest(..., pod_template_extra_labels={"azure.workload.identity/use":"true"}, sandbox_label="aks")` so the existing direct-call test stays green unchanged.
- `_AZURE_CLOUD = CloudSpec(provider="azure", settings_prefix="azure", sandbox_prefix="aks", sandbox_label="aks", pod_template_extra_labels={"azure.workload.identity/use":"true"}, make_object_store=<AzureBlobStore from azure_storage_account/azure_blob_container>, ensure_available=ensure_azure_available, base_image_setting="azure_base_image")`.
- `class AksJobBackend(_KubernetesJobBackend): def __init__(self, **kw): super().__init__(cloud=_AZURE_CLOUD, **kw)`.
- `ensure_azure_available()`: keep current body (or thin-wrap `ensure_k8s_cloud_available(_AZURE_CLOUD)` if the CloudSpec carries the Azure SDK/cred checks). Either way the test's expectations are unchanged.
- `__all__ = ["AksJobBackend", "ensure_azure_available"]`.

### NEW `backend/services/runtime/gke_job_backend.py` — GCP adapter (mirror of slim aks)
- `_GCP_CLOUD = CloudSpec(provider="gcp", settings_prefix="gcp", sandbox_prefix="gke", sandbox_label="gke", pod_template_extra_labels={}, make_object_store=<GcsStore from gcp_gcs_bucket/gcp_project>, ensure_available=ensure_gcp_available, base_image_setting="gcp_base_image")`.
- `class GkeJobBackend(_KubernetesJobBackend): def __init__(self, **kw): super().__init__(cloud=_GCP_CLOUD, **kw)`.
- `ensure_gcp_available()`: SDK checks `google.cloud.storage` + `google.auth` + `kubernetes`; settings `gcp_gcs_bucket`/`gcp_project` non-empty; ADC probe `google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_write"])`; kubeconfig load (shared loader). Actionable error messages (`gcloud auth application-default login`, `gcloud container clusters get-credentials`).
- `__all__ = ["GkeJobBackend", "ensure_gcp_available"]`.

### CHANGED wiring (small, additive — each mirrors the azure line)
- `backend/agents/execution.py`: `SandboxMode.gcp = "gcp"` (enum); `ensure_sandbox_mode_available` add `elif resolved is SandboxMode.gcp: from backend.services.runtime import ensure_gcp_available; ensure_gcp_available()`.
- `backend/config.py`: add `"gcp"` to `default_sandbox` + `force_sandbox` Literals; add the full `gcp_*` block mirroring `azure_*` (24 fields; defaults: `gcp_region="us-central1"`, `gcp_blob_container`→`gcp_gcs_bucket=""`, `gcp_namespace="reprolab"`, `gcp_service_account="reprolab-sa"`, `gcp_node_pool_name="gpua100"`, `gcp_per_gpu_vram_gb=80.0`, `gcp_max_nodes=4`, `gcp_base_image=""`, `gcp_gpu_usd_per_hour=3.93`, `gcp_gpu_skus=["gcp_a100_80"]`, plus the timeout/oom/cache knobs identical to azure).
- `backend/services/runtime/__init__.py`: import + `__all__` add `GkeJobBackend`, `ensure_gcp_available`.
- `backend/services/runtime/gpu_capacity.py`: add `"gcp"` to the `_backend_kind` substring tuple; dispatch `kind == "gcp" → _describe_gcp(ctx)`; add `_describe_gcp` mirroring `_describe_azure` with `gcp_*` settings and **`short_name.startswith("gcp_")`** plan disambiguation (NOT `cloud_type=="ONDEMAND"`, which matches both clouds).
- `backend/agents/rlm/primitives.py`:
  - Backend factory `_backend_for_sandbox_mode` (~2723): add `elif mode is SandboxMode.gcp:` → `from backend.services.runtime.gke_job_backend import GkeJobBackend; _runtime.ensure_gcp_available(); return GkeJobBackend(run_budget=run_budget, gpu_plan=gpu_plan)`.
  - `resolve_gpu_requirements` (~1238/1246): when sandbox is gcp, set `_provider="gcp"`, `_provisioned_skus=settings.gcp_gpu_skus`, `cloud_types=("ONDEMAND",)` — mirror the `_is_azure` arm.
- `backend/cli.py` (~2119): add `"gcp"` to `--sandbox` choices.

### Cell-matrix path (`k8s_job_cell_runner` generalization) — sub-increment 1e
Generalize `k8s_job_cell_runner.py` to be cloud-parameterized (settings-prefix + ObjectStore + `default_sku` + `pod_template_extra_labels`), and add the `gcp` arm in `primitives.py::_execute_cell_matrix` (~5592, currently `_sb_key_ecm == "azure"`). Default `default_sku` becomes cloud-derived (`azure_a100_80` / `gcp_a100_80`). Guarded by `test_aks_cell_entrypoint` staying green + a new GKE cell test. (Done after the backend layer is green.)

## Tests (new)
- `tests/services/runtime/test_gcs_blob.py` — mirror `test_azure_blob.py` (imports w/o SDK, upload/download via fake client, name validation, prefix exclude set).
- `tests/services/runtime/test_gke_job_backend.py` — mirror `test_aks_job_backend.py` (create_sandbox uploads to GCS, exec submits a Job w/ correct manifest + **no** Azure WI label + `reprolab/sku` nodeSelector + GPU toleration, watch/timeout/pending paths, copy_in/out, destroy, `ensure_gcp_available` failure modes, module imports w/o SDK).
- `tests/services/runtime/test_gpu_capacity_gcp.py` — mirror `test_gpu_capacity_azure.py` (`_describe_gcp` reads `gcp_*`, gpu_plan.json override gated on `gcp_` prefix).
- Config test: `gcp_*` fields parse + `OPENRESEARCH_GCP_*` env round-trip; `SandboxMode("gcp")`; factory returns `GkeJobBackend` for `SandboxMode.gcp` (with deps faked).

## Acceptance gate
1. `pytest tests/services/runtime tests/config tests/rlm/test_registry.py -q` green (incl. the 189 baseline).
2. `ruff check backend/services/runtime backend/agents/execution.py backend/config.py` clean.
3. `SandboxMode.gcp` resolvable; `_backend_for_sandbox_mode(SandboxMode.gcp, ...)` returns `GkeJobBackend` (deps faked); `describe_capacity` with `ctx.sandbox_mode="gcp"` returns a `gcp` `GpuCapacity`.
4. No diff to behaviour for existing sandboxes (azure baseline tests unchanged).
