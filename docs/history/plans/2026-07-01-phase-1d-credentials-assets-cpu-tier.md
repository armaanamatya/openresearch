# Phase 1d — CredentialBroker + generic AssetResolver + cpu_warm_disk_then_gpu_attach — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** (A) a `CredentialBroker` that resolves configured secrets for resolve/triage/stage/provision and never puts a raw `.env` on the wire; (B) a generic `AssetResolver` that resolves the Phase-1b `RequiredAsset` list into the shared `AssetCache` for ANY paper (unresolved → verified `Exclusion`, never a fake-0); (C) the `cpu_warm_disk_then_gpu_attach` two-VM tiering strategy in `VmComputeProvider`. All default-OFF / unwired ⇒ live path byte-identical.

**Architecture:** `CredentialBroker` follows the canonical `foundry_endpoint._env_or_settings` resolver (os.environ → Settings, fail-soft) — the cloud secret stores already inject secrets as env vars via CSI, so reading `os.environ` is uniform across VM + cluster. `AssetResolver` WRAPS the existing fetchers (`dataset_recipes.find_recipe`, `asset_provisioning.warm_hf_models`, `huggingface_hub.snapshot_download`, `env_pin`) behind one `resolve(asset, cache)` dispatch — it does NOT rewrite them (the live SDAR provisioning path stays intact). The tiering strategy extends `VmComputeProvider` with argv builders only (live validation operator-gated).

**Tech Stack:** Python 3.12 / floor 3.11; `pytest` socket-hermetic; stdlib + existing modules. No new third-party deps. All fetch/subprocess calls injected in tests.

## Global Constraints
- **Everything default-OFF / unwired ⇒ byte-identical.** No live `run.py`/provisioning path changes. New modules are consumed by the (unwired) `ReproductionRun`/triage seams, not the live SDAR path.
- **Do NOT rewrite** `asset_provisioning.py` / `dataset_recipes.py` / `environment_detective.py` / `env_cache.py` — `AssetResolver` reuses them. Do NOT extend the K8s `CloudSpec`.
- **CredentialBroker red line:** no raw `.env` on the wire; no secret-shaped value in any staged bundle or logged command (redaction tests enforce). It resolves secrets os.environ → Settings (never prints/returns them into a bundle).
- **Fail-soft:** an unresolved asset → a verified `Exclusion` (fairness); a needed-but-absent credential → a `gated` `Exclusion`, never a hang or raise.
- Hermetic: inject every fetcher/subprocess/HTTP call; no real downloads/gcloud in tests. Env naming `OPENRESEARCH_*`.
- Commit at the phase milestone; no CC prefix; no `Co-Authored-By`; author `lolout1`. Confirm before committing.

## Component → file map
| Component | New/extends | Where |
|---|---|---|
| `CredentialBroker` + canonical redaction | NEW | `backend/services/runtime/credential_broker.py` |
| `AssetResolver` + `ResolveResult` + data-driven framework matrix | NEW | `backend/services/runtime/asset_resolver.py` |
| `cpu_warm_disk_then_gpu_attach` strategy | extends | `backend/services/runtime/vm_compute_provider.py` (additive branch) |

---

## Unit A — CredentialBroker

**Files:** Create `backend/services/runtime/credential_broker.py`; Test `tests/services/runtime/test_credential_broker.py`.

**Interfaces:**
```python
# canonical registry: logical name -> tuple of env-var candidates (first non-empty wins), + optional Settings attr
_SECRET_REGISTRY: dict[str, tuple[tuple[str, ...], str | None]] = {
    "hf_token":            (("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"), None),
    "anthropic_api_key":   (("ANTHROPIC_API_KEY",), "anthropic_api_key"),
    "openai_api_key":      (("OPENAI_API_KEY",), "openai_api_key"),
    "runpod_api_key":      (("RUNPOD_API_KEY",), "runpod_api_key"),
    "azure_foundry_api_key": (("AZURE_FOUNDRY_API_KEY",), "azure_foundry_api_key"),
    "aws_access_key_id":   (("AWS_ACCESS_KEY_ID",), None),
    "aws_secret_access_key": (("AWS_SECRET_ACCESS_KEY",), None),
    "kaggle_key":          (("KAGGLE_KEY",), None),
}
_SECRET_KEY_RE = re.compile(r"key|token|secret|password", re.IGNORECASE)

class CredentialBroker:
    def __init__(self, *, env: "dict | None" = None, settings_getter: "Callable[[], object] | None" = None) -> None: ...
    def get(self, name: str) -> str | None: ...       # os.environ candidates → Settings attr; None if unset (fail-soft)
    def available(self, name: str) -> bool: ...        # get(name) is a non-empty string
    def require(self, name: str) -> str: ...           # get() or raise KeyError(name) — callers prefer available()+gated_exclusion
    @staticmethod
    def redact_env(env: "dict | None") -> dict: ...     # drop keys matching _SECRET_KEY_RE (the canonical impl)
    @staticmethod
    def redact_text(text: str) -> str: ...              # drop whole lines mentioning key/token/secret/password
    def gated_exclusion(self, *, item: str, secret_name: str, axis: str = "dataset") -> "Exclusion | None": ...
        # returns a verified `gated` Exclusion (kind=KIND_ENV_SETUP_FAILED, reason "gated: needs <secret_name>")
        # when the secret is absent; None when it's available.
```
`get` resolution mirrors `foundry_endpoint._env_or_settings`: try each env candidate in `self._env` (default `os.environ`) first, then `getattr(settings, attr)` if `attr` is set; strip; fail-soft (Settings import error → skip). Never logs the value.

- [ ] **Step 1: Write the failing tests**
```python
import pytest
from backend.services.runtime.credential_broker import CredentialBroker


def test_get_prefers_env_then_settings():
    b = CredentialBroker(env={"HF_TOKEN": "hf_xxx"})
    assert b.get("hf_token") == "hf_xxx" and b.available("hf_token") is True


def test_get_settings_fallback():
    class _S:  # fake settings
        anthropic_api_key = "sk-ant"
    b = CredentialBroker(env={}, settings_getter=lambda: _S())
    assert b.get("anthropic_api_key") == "sk-ant"


def test_absent_secret_is_none_and_unavailable():
    b = CredentialBroker(env={}, settings_getter=lambda: object())
    assert b.get("hf_token") is None and b.available("hf_token") is False


def test_redact_env_drops_secret_keys():
    out = CredentialBroker.redact_env({"ANTHROPIC_API_KEY": "x", "HF_TOKEN": "y", "SEED": "0", "MODEL": "qwen"})
    assert out == {"SEED": "0", "MODEL": "qwen"}


def test_redact_text_drops_secret_lines():
    text = "line ok\nANTHROPIC_API_KEY=sk-secret\nother fine"
    red = CredentialBroker.redact_text(text)
    assert "sk-secret" not in red and "line ok" in red and "other fine" in red


def test_gated_exclusion_when_absent_else_none():
    from backend.agents.rlm import exclusion as X
    b = CredentialBroker(env={})
    exc = b.gated_exclusion(item="gated-ds", secret_name="hf_token", axis=X.AXIS_DATASET)
    assert exc is not None and exc.verified and exc.kind == X.KIND_ENV_SETUP_FAILED and "gated" in exc.reason.lower()
    b2 = CredentialBroker(env={"HF_TOKEN": "hf_xxx"})
    assert b2.gated_exclusion(item="gated-ds", secret_name="hf_token", axis=X.AXIS_DATASET) is None


def test_never_logs_or_returns_secret_in_redaction(caplog):
    # redact_env must not leak the value anywhere it returns.
    out = CredentialBroker.redact_env({"OPENAI_API_KEY": "sk-leak"})
    assert "sk-leak" not in repr(out)
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `credential_broker.py` (registry + resolver mirroring `foundry_endpoint._env_or_settings`; `redact_env`/`redact_text` matching `failure_capsule._redact_env`/`_redact_text` semantics — this becomes the canonical home; `gated_exclusion` builds a verified `Exclusion` via `backend.agents.rlm.exclusion`). Confirm `X.AXIS_DATASET`/`X.KIND_ENV_SETUP_FAILED` exist. **Step 4:** Run → PASS. Lint clean.

---

## Unit B — generic AssetResolver

**Files:** Create `backend/services/runtime/asset_resolver.py`; Test `tests/services/runtime/test_asset_resolver.py`.

**Consumes:** `RequiredAsset` (1b `run_plan.py`), `AssetCache` (`asset_cache.py`), `CredentialBroker` (Unit A), reuses `dataset_recipes.find_recipe`, `asset_provisioning.warm_hf_models`/`warm_datasets`, `env_pin` (torch-core), `exclusion`.

**Design:** `AssetResolver.resolve(asset, cache)` dispatches by `asset.kind`, reusing the existing fetchers behind injected callables (hermetic). It does NOT rewrite the existing modules. Data-driven framework matrix replaces the hardcoded `_FRAMEWORK_COMPATIBILITY`.

```python
_FRAMEWORK_MATRIX: dict[str, dict[str, dict[str, str]]] = {   # data-driven (extendable) replacement
    "pytorch":    {"2.5.1": {"python": "3.12", "cuda": "12.1"}, "2.2.0": {"python": "3.11", "cuda": "12.1"},
                   "2.1.0": {"python": "3.11", "cuda": "11.8"}, "2.0.0": {"python": "3.10", "cuda": "11.7"}},
    "tensorflow": {"2.15.0": {"python": "3.11", "cuda": "12.2"}, "2.14.0": {"python": "3.11", "cuda": "11.8"}},
    "jax":        {"0.4.25": {"python": "3.11", "cuda": "12.1"}},
}
def resolve_framework(name: str, version: str | None = None) -> dict:
    """Data-driven framework→{python,cuda}; graceful fallback (latest known, then a safe default)."""

@dataclass(frozen=True)
class ResolveResult:
    ok: bool
    asset: "RequiredAsset"
    local_path: str | None = None
    env_vars: dict = field(default_factory=dict)
    exclusion: "Exclusion | None" = None
    detail: str = ""

class AssetResolver:
    def __init__(self, *, broker=None,
                 hf_snapshot: "Callable[[str], str] | None" = None,   # (repo_id) -> local dir  (injected; default snapshot_download)
                 url_fetch: "Callable[[str, str], str] | None" = None, # (url, dest) -> path      (injected; default urllib)
                 recipe_lookup: "Callable[[str], object | None] | None" = None) -> None: ...  # default dataset_recipes.find_recipe
    def resolve(self, asset: "RequiredAsset", cache: "AssetCache") -> ResolveResult: ...
    def resolve_all(self, assets, cache) -> list[ResolveResult]: ...
```
Dispatch (`resolve`):
- `asset.gated` (or a known-gated identifier) AND `not broker.available("hf_token")` → `ResolveResult(ok=False, exclusion=broker.gated_exclusion(...))`.
- `kind in ("dataset","weights")`: a `recipe_lookup(identifier)` hit → resolve via the recipe (torchvision/registry, mark `ok=True` + detail); else an HF-repo-shaped id (`owner/name`) → `hf_snapshot(identifier)` into cache; else a URL → `url_fetch`; else → unresolved `Exclusion`.
- `kind == "framework"`: `resolve_framework(identifier)` → `ok=True` with `env_vars` carrying the `{python,cuda}` (no download); reuse `env_pin` for the torch-core note.
- `kind in ("image","service")`: not an AssetResolver concern (env adapters handle services) → `ok=True, detail="handled elsewhere"` (no-op, not an exclusion).
- Any fetch raise → fail-soft verified `Exclusion` (never propagate, never fake-0).

- [ ] **Step 1: Write the failing tests** (all fetchers injected — no network)
```python
from backend.services.runtime.asset_resolver import AssetResolver, ResolveResult, resolve_framework
from backend.services.runtime.run_plan import RequiredAsset
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.credential_broker import CredentialBroker
from backend.agents.rlm import exclusion as X


def test_hf_weights_resolved_via_injected_snapshot(tmp_path):
    calls = []
    r = AssetResolver(broker=CredentialBroker(env={}),
                      hf_snapshot=lambda repo: calls.append(repo) or f"/cache/{repo}")
    res = r.resolve(RequiredAsset("weights", "Qwen/Qwen3-1.7B"), AssetCache(tmp_path))
    assert res.ok and res.local_path == "/cache/Qwen/Qwen3-1.7B" and calls == ["Qwen/Qwen3-1.7B"]


def test_recipe_dataset_resolved_without_download(tmp_path):
    r = AssetResolver(broker=CredentialBroker(env={}),
                      recipe_lookup=lambda name: object() if name.lower() == "cifar-10" else None)
    res = r.resolve(RequiredAsset("dataset", "CIFAR-10"), AssetCache(tmp_path))
    assert res.ok and res.exclusion is None


def test_gated_dataset_without_cred_is_gated_exclusion(tmp_path):
    r = AssetResolver(broker=CredentialBroker(env={}))   # no HF_TOKEN
    res = r.resolve(RequiredAsset("dataset", "secret/gated", gated=True), AssetCache(tmp_path))
    assert not res.ok and res.exclusion is not None and res.exclusion.verified
    assert "gated" in res.exclusion.reason.lower()


def test_unresolvable_asset_is_verified_exclusion_not_fake_ok(tmp_path):
    def _boom(_): raise RuntimeError("network down")
    r = AssetResolver(broker=CredentialBroker(env={}), hf_snapshot=_boom)
    res = r.resolve(RequiredAsset("weights", "owner/model"), AssetCache(tmp_path))
    assert not res.ok and res.exclusion is not None and res.exclusion.verified   # fail-soft, never fake-ok


def test_framework_matrix_is_data_driven():
    assert resolve_framework("pytorch", "2.2.0") == {"python": "3.11", "cuda": "12.1"}
    assert resolve_framework("pytorch", "9.9.9")["cuda"]        # unknown version → graceful fallback, non-empty
    assert resolve_framework("unknown-fw")                       # unknown framework → safe default dict, no raise


def test_service_kind_is_noop_not_exclusion(tmp_path):
    r = AssetResolver(broker=CredentialBroker(env={}))
    res = r.resolve(RequiredAsset("service", "webshop-server"), AssetCache(tmp_path))
    assert res.ok and res.exclusion is None
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `asset_resolver.py` (dispatch as above; default `hf_snapshot` lazy-imports `huggingface_hub.snapshot_download`, default `url_fetch` uses `urllib`, default `recipe_lookup` = `dataset_recipes.find_recipe`; a known-gated identifier heuristic may consult `asset.gated`). **Step 4:** Run → PASS. Lint clean.

---

## Unit C — cpu_warm_disk_then_gpu_attach tiering (VmComputeProvider extension)

**Files:** Modify `backend/services/runtime/vm_compute_provider.py` (additive strategy branch); Test add to `tests/services/runtime/test_vm_compute_provider.py`.

**Consumes:** the existing `VmComputeProvider` (1c) + `VmSpec.tiering_strategy`/`cache_disk_name`/`cpu_machine_type`.

**Design (argv builders only — live validation is operator-gated Phase-1f):** when `profile.vm.tiering_strategy == "cpu_warm_disk_then_gpu_attach"`:
- `provision_cpu` → create a CHEAP CPU VM (`cpu_machine_type`, e.g. `e2-standard-16`, no accelerator) + `disks create` (if absent) + `attach-disk` the cache disk + an `ssh` warm step (the `.warm_ok`-sentinel-gated stage) + `detach-disk`. Returns a lease with `disk=<cache_disk_name>`, `cpu=<cpu-vm>`, `meta["strategy"]="cpu_warm_disk_then_gpu_attach"`, and NO `gpu` (no GPU billing yet).
- `acquire_gpu` → create the GPU VM (`gpu_machine_type`) + `attach-disk` the warmed cache disk (SAME zone — assert `disk_zone == zone` or fail-soft to `stage_on_gpu`). Stamps `gpu`.
- Reuse the existing `_gcloud`/`_ssh_argv`/`_stop_with_retry`/argv helpers. `stage_on_gpu` (default) path is unchanged.
- Add a `detach-disk` argv builder. Same-zone check mirrors the bash (`gcp_sdar_preflight.sh:289-294`).

- [ ] **Step 1: Write the failing tests** (recording fake runner; argv parity)
```python
def test_cpu_warm_disk_provision_uses_cheap_cpu_vm_and_attaches_disk(tmp_path):
    calls = []
    from backend.services.runtime.cloud_profile import CloudProfile, VmSpec
    from backend.services.runtime.vm_compute_provider import VmComputeProvider
    prof = CloudProfile(cloud="gcp", vm=VmSpec(
        zone="us-central1-b", cpu_machine_type="e2-standard-16", gpu_machine_type="a2-highgpu-4g",
        machine_image="sdar-ultra", cache_disk_name="sdar-cache",
        tiering_strategy="cpu_warm_disk_then_gpu_attach"))
    prov = VmComputeProvider(prof, runner=lambda a: calls.append(a) or _ok())
    lease = prov.provision_cpu(_run_plan())
    joined = [" ".join(a) for a in calls]
    assert any("e2-standard-16" in j and "instances create" in j for j in joined)   # cheap CPU VM
    assert any("disks create" in j and "sdar-cache" in j for j in joined) or \
           any("attach-disk" in j and "sdar-cache" in j for j in joined)            # disk warmed
    assert any("detach-disk" in j for j in joined)                                  # detached after warm
    assert lease.gpu is None and lease.disk == "sdar-cache"                          # no GPU billing yet


def test_cpu_warm_disk_acquire_gpu_attaches_warmed_disk(tmp_path):
    calls = []
    prov = _cpu_warm_provider(lambda a: calls.append(a) or _ok())
    lease = prov.provision_cpu(_run_plan()); calls.clear()
    lease = prov.acquire_gpu(lease)
    joined = [" ".join(a) for a in calls]
    assert any("a2-highgpu-4g" in j and "instances create" in j for j in joined)    # GPU VM created
    assert any("attach-disk" in j and "sdar-cache" in j for j in joined)            # warmed disk attached
    assert lease.gpu is not None


def test_stage_on_gpu_default_unchanged(tmp_path):
    # the default strategy still folds provision_cpu into the GPU create (1c behavior).
    calls = []
    prov = _stage_on_gpu_provider(lambda a: calls.append(a) or _ok())
    prov.provision_cpu(_run_plan())
    joined = [" ".join(a) for a in calls]
    assert any("a2-highgpu-4g" in j and "instances create" in j for j in joined)
    assert not any("e2-standard-16" in j for j in joined)                           # no cheap CPU VM under stage_on_gpu
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement the additive `cpu_warm_disk_then_gpu_attach` branch in `provision_cpu`/`acquire_gpu` + a `_detach_disk_argv` builder; keep the `stage_on_gpu` branch byte-identical. **Step 4:** Run the FULL `test_vm_compute_provider.py` (new + the 7 existing 1c tests unchanged) → PASS. Lint clean.

---

## Validation
- [ ] `.venv/bin/python -m pytest tests/services/runtime/test_credential_broker.py tests/services/runtime/test_asset_resolver.py tests/services/runtime/test_vm_compute_provider.py -q`
- [ ] Broad regression: `.venv/bin/python -m pytest tests/services/runtime/ -q`
- [ ] Ruff clean on all new/changed files.
- [ ] Import smoke: `.venv/bin/python -c "import backend.agents.rlm.run, backend.cli; print('ok')"`
- [ ] Docs: CLAUDE.md note + memory update.

## Self-Review (against spec §6.1/§6.2/§5.5)
- §6.2 `CredentialBroker` scoped to resolve/triage/stage/provision, no raw `.env`, redaction tests → Unit A. ✓
- §6.2 `AssetResolver` generic Tier-1 (HF/URL/torchvision, data-driven matrix, env_pin), consumes 1b assets, unresolved → `Exclusion` → Unit B. ✓
- §5.5 `cpu_warm_disk_then_gpu_attach` two-VM handoff (same-zone) → Unit C (argv-parity; live validation operator-gated). ✓
- **Deferred (honest):** wiring the broker into `VmComputeProvider.stage` (1c already redacts by-construction — consolidation is a follow-up, not a rewrite); the `EnvironmentAdapter` Tier-2 already exists (1a); live two-VM validation + the default-flip (operator-gated).
