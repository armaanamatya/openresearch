"""Pluggable accelerator-provider abstraction for the RLM orchestrator.

The accelerator provider resolves WHERE the RLM's high-volume, cheap LLM calls
run.  It returns an :class:`AcceleratorEndpoint` — an OpenAI-compatible endpoint
descriptor — so that any downstream code can wire an :class:`OpenAILlmClient`
(or Azure variant) against it without knowing the provisioning details.

Supported providers (``mode`` arg to :func:`resolve_accelerator`):

* ``"off"``      — disable; returns ``None`` so callers keep the default
                   Sonnet/OAuth path.
* ``"auto"``     — dynamic best-effort pick: local GPU → RunPod proxy →
                   Azure → ``None``.  Never raises; returns ``None`` on any
                   miss.
* ``"local"``    — on-device vLLM server expected at
                   ``REPROLAB_ACCELERATOR_BASE_URL`` (default
                   ``http://127.0.0.1:8001/v1``).  The server itself is started
                   by ``scripts/serve_local_llm.py``; this module just resolves
                   and probes.  Returns ``None`` when the probe fails (server not
                   up) — for explicit ``"local"`` the caller gets ``None``, not an
                   exception, because the server may simply not be running yet.
* ``"runpod"``   — scaffold: if ``REPROLAB_ACCELERATOR_BASE_URL`` is already set
                   to a RunPod proxy URL, uses it; otherwise raises
                   :class:`AcceleratorError` for explicit mode, or returns
                   ``None`` for ``"auto"``.  Full auto-provisioning is a future
                   task (see TODO below).
* ``"azure"``    — Azure OpenAI endpoint from ``AZURE_OPENAI_API_KEY``,
                   ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_DEPLOYMENT``.  Raises
                   :class:`AcceleratorError` when creds are missing in explicit
                   mode; returns ``None`` in ``"auto"``.
* ``"endpoint"`` — arbitrary user-supplied OpenAI-compatible endpoint from
                   ``REPROLAB_ACCELERATOR_BASE_URL``.  Raises
                   :class:`AcceleratorError` when the env var is absent.

Adding a new provider
---------------------
1. Add a branch in :func:`resolve_accelerator` that reads env/settings, calls
   :func:`probe_endpoint` (or skips for cloud providers where a round-trip is
   expensive), constructs an :class:`AcceleratorEndpoint`, and either returns it
   or raises / returns ``None`` according to the explicit/auto contract.
2. Document it in the docstring above and in the ``"auto"`` branch.
3. Add a test in ``tests/rlm/test_accelerator.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

__all__ = [
    "AcceleratorEndpoint",
    "AcceleratorError",
    "build_accelerator_client",
    "probe_endpoint",
    "resolve_accelerator",
]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorEndpoint:
    """Descriptor for an OpenAI-compatible accelerator endpoint.

    Attributes
    ----------
    base_url:
        Root URL of the OpenAI-compatible API, e.g.
        ``"http://127.0.0.1:8001/v1"``.  Must include the version suffix.
    model:
        Model identifier served at *base_url*, e.g.
        ``"Qwen/Qwen2.5-Coder-32B-Instruct"``.
    api_key:
        API key for the endpoint.  Defaults to ``"local"`` for on-device
        servers that do not require authentication.
    kind:
        Originating provider: ``"local"``, ``"runpod"``, ``"azure"``, or
        ``"endpoint"``.
    is_azure:
        ``True`` only for Azure OpenAI endpoints.
        :func:`build_accelerator_client` routes to
        :class:`~backend.services.context.workspace.tools.azure_openai_client.AzureOpenAILlmClient`
        when this flag is set.
    """

    base_url: str
    model: str
    api_key: str = "local"
    kind: str = "endpoint"
    is_azure: bool = False


class AcceleratorError(RuntimeError):
    """Raised when an explicitly requested accelerator provider cannot be satisfied.

    *Not* raised in ``"auto"`` mode — that path returns ``None`` on any miss.
    """


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------

# Default API version used when probing / connecting to Azure endpoints.
_DEFAULT_AZURE_API_VERSION = "2024-10-21"

# Default local vLLM address that ``scripts/serve_local_llm.py`` binds.
_DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8001/v1"

# Default model name expected by the local vLLM server.
_DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


def probe_endpoint(base_url: str, *, timeout: float = 3.0) -> bool:
    """Return ``True`` iff ``GET {base_url}/models`` returns a 2xx response.

    Handles the common mis-specification where *base_url* already ends with
    ``/models`` (passed through as-is) or ends with ``/v1`` (appends
    ``/models``).  Any network error, timeout, or non-2xx status is treated as
    ``False`` — strictly safe-by-default so callers can fall back cleanly.

    Uses only :mod:`urllib.request` from the stdlib to avoid importing
    ``requests``/``httpx`` at module level and keep the import cost negligible.

    Parameters
    ----------
    base_url:
        The root of the OpenAI-compatible API, e.g.
        ``"http://127.0.0.1:8001/v1"``.
    timeout:
        Connection + read timeout in seconds.  Defaults to ``3.0`` so the
        probe does not stall startup for more than a few seconds.
    """
    import urllib.request
    import urllib.error

    url = base_url.rstrip("/")
    # Normalise: /v1 -> /v1/models; already /v1/models -> keep; other -> append /models.
    if url.endswith("/models"):
        probe_url = url
    elif url.endswith("/v1"):
        probe_url = url + "/models"
    else:
        probe_url = url + "/models"

    try:
        req = urllib.request.Request(probe_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # noqa: BLE001 — catch-all: network, timeout, HTTP errors
        _log.debug("accelerator: probe %s failed: %s", probe_url, exc)
        return False


# ---------------------------------------------------------------------------
# Provider sub-resolvers
# ---------------------------------------------------------------------------


def _resolve_local(*, explicit: bool) -> AcceleratorEndpoint | None:
    """Resolve the on-device vLLM provider.

    Reads ``REPROLAB_ACCELERATOR_BASE_URL`` (default ``http://127.0.0.1:8001/v1``)
    and ``REPROLAB_ACCELERATOR_MODEL`` (default Qwen2.5-Coder-32B-Instruct).

    Probes the endpoint; returns ``None`` when the probe fails regardless of
    whether the call was explicit or from ``"auto"`` — the server simply may
    not be running yet, and a ``None`` return lets callers fall back to the
    default Sonnet/OAuth path without a hard error.
    """
    base_url = os.environ.get("REPROLAB_ACCELERATOR_BASE_URL", _DEFAULT_LOCAL_BASE_URL)
    model = os.environ.get("REPROLAB_ACCELERATOR_MODEL", _DEFAULT_LOCAL_MODEL)

    if probe_endpoint(base_url):
        return AcceleratorEndpoint(
            base_url=base_url,
            model=model,
            api_key="local",
            kind="local",
        )

    level = logging.WARNING if explicit else logging.INFO
    _log.log(
        level,
        "accelerator[local]: probe failed at %s — server not running; returning None",
        base_url,
    )
    return None


def _resolve_runpod(*, explicit: bool) -> AcceleratorEndpoint | None:
    """Resolve the RunPod accelerator provider.

    Scaffold implementation.  If ``REPROLAB_ACCELERATOR_BASE_URL`` is already
    set to a RunPod vLLM proxy URL, validate it with a probe and return the
    endpoint.  Otherwise:

    * explicit mode  → raise :class:`AcceleratorError` with an actionable
      message.
    * auto mode      → return ``None`` (graceful fallback).

    TODO (full auto-provisioning):
    --------------------------------
    Full auto-provisioning would call
    :class:`~backend.services.runtime.runpod_backend.RunpodBackend` to:

    1. Request a GPU pod with a pre-built vLLM image (e.g.
       ``vllm/vllm-openai:latest``) sized to the model's VRAM requirement.
    2. Bootstrap the vLLM server on the pod's public IP at port 8001 (or
       behind a proxy).
    3. Wait for the ``/v1/models`` probe to return 2xx.
    4. Return an :class:`AcceleratorEndpoint` with ``base_url=<pod_proxy_url>``,
       ``kind="runpod"``.
    5. Register an atexit / context-manager hook to delete the pod on run
       completion so cost is bounded.

    Hook point: instantiate ``RunpodBackend`` from
    ``backend.services.runtime.runpod_backend``, call ``create_sandbox`` with a
    minimal ``SandboxConfig`` whose ``image`` is the vLLM container, then
    extract the public IP from the returned ``Sandbox`` object and store it in
    ``REPROLAB_ACCELERATOR_BASE_URL`` for the remainder of the process so
    subsequent ``_resolve_runpod`` calls hit the existing pod.
    """
    proxy_url = os.environ.get("REPROLAB_ACCELERATOR_BASE_URL", "").strip()
    model = os.environ.get("REPROLAB_ACCELERATOR_MODEL", _DEFAULT_LOCAL_MODEL)

    if proxy_url:
        if probe_endpoint(proxy_url):
            return AcceleratorEndpoint(
                base_url=proxy_url,
                model=model,
                api_key=os.environ.get("REPROLAB_ACCELERATOR_API_KEY", "local"),
                kind="runpod",
            )
        _log.warning(
            "accelerator[runpod]: probe failed at %s — endpoint set but unreachable",
            proxy_url,
        )
        if explicit:
            raise AcceleratorError(
                f"RunPod accelerator endpoint set to {proxy_url!r} but probe failed. "
                "Ensure the vLLM server is running on the pod and the proxy URL is correct."
            )
        return None

    # No proxy URL at all.
    if explicit:
        raise AcceleratorError(
            "runpod accelerator auto-provisioning not yet implemented — "
            "set REPROLAB_ACCELERATOR_BASE_URL to a running RunPod vLLM endpoint, "
            "or use --accelerator local"
        )
    _log.info(
        "accelerator[runpod]: REPROLAB_ACCELERATOR_BASE_URL not set; "
        "skipping runpod provider in auto mode"
    )
    return None


def _resolve_azure(*, explicit: bool) -> AcceleratorEndpoint | None:
    """Resolve the Azure OpenAI accelerator provider.

    Reads ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``, and
    ``AZURE_OPENAI_DEPLOYMENT`` from the environment.  ``AZURE_OPENAI_API_VERSION``
    may also be set; defaults to the same GA version used by
    :class:`~backend.services.context.workspace.tools.azure_openai_client.AzureOpenAILlmClient`.

    No endpoint probe is performed for Azure — the SDK handles auth and
    connectivity; an invalid key/endpoint surfaces at the first completion call
    with a typed error.

    Raises :class:`AcceleratorError` in explicit mode when required credentials
    (key + endpoint) are absent; returns ``None`` in ``"auto"`` mode.
    """
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()

    if not (api_key and endpoint):
        if explicit:
            missing = []
            if not api_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            raise AcceleratorError(
                f"Azure accelerator provider requires {' and '.join(missing)} to be set. "
                "Set these environment variables and retry."
            )
        _log.info(
            "accelerator[azure]: credentials not present (AZURE_OPENAI_API_KEY / "
            "AZURE_OPENAI_ENDPOINT); skipping in auto mode"
        )
        return None

    model = deployment or "gpt-4o"
    return AcceleratorEndpoint(
        base_url=endpoint.rstrip("/"),
        model=model,
        api_key=api_key,
        kind="azure",
        is_azure=True,
    )


def _resolve_endpoint(*, explicit: bool) -> AcceleratorEndpoint | None:
    """Resolve a user-supplied arbitrary OpenAI-compatible endpoint.

    Reads ``REPROLAB_ACCELERATOR_BASE_URL`` (required),
    ``REPROLAB_ACCELERATOR_MODEL`` (required; no default because the caller
    must know what is served), and ``REPROLAB_ACCELERATOR_API_KEY``
    (default ``"local"``).

    Raises :class:`AcceleratorError` when ``REPROLAB_ACCELERATOR_BASE_URL`` is
    absent.
    """
    base_url = os.environ.get("REPROLAB_ACCELERATOR_BASE_URL", "").strip()
    if not base_url:
        if explicit:
            raise AcceleratorError(
                "accelerator mode 'endpoint' requires REPROLAB_ACCELERATOR_BASE_URL to be set."
            )
        return None

    model = os.environ.get("REPROLAB_ACCELERATOR_MODEL", "").strip()
    if not model:
        if explicit:
            raise AcceleratorError(
                "accelerator mode 'endpoint' requires REPROLAB_ACCELERATOR_MODEL to be set."
            )
        return None

    api_key = os.environ.get("REPROLAB_ACCELERATOR_API_KEY", "local")

    if not probe_endpoint(base_url):
        if explicit:
            raise AcceleratorError(
                f"Accelerator endpoint {base_url!r} did not respond to a health probe. "
                "Ensure the server is running and REPROLAB_ACCELERATOR_BASE_URL is correct."
            )
        _log.info("accelerator[endpoint]: probe failed at %s; returning None", base_url)
        return None

    return AcceleratorEndpoint(
        base_url=base_url,
        model=model,
        api_key=api_key,
        kind="endpoint",
    )


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def resolve_accelerator(
    mode: str,
    *,
    sandbox_mode: object = None,
    settings: object = None,
) -> AcceleratorEndpoint | None:
    """Resolve WHERE cheap LLM calls should run for the current environment.

    Parameters
    ----------
    mode:
        One of ``"off"``, ``"auto"``, ``"local"``, ``"runpod"``, ``"azure"``,
        ``"endpoint"``.
    sandbox_mode:
        The active sandbox mode (a :class:`~backend.agents.execution.SandboxMode`
        enum or plain string).  Used by ``"auto"`` to prefer RunPod when the
        execution sandbox is already RunPod.
    settings:
        Optional pydantic-settings ``Settings`` object.  Reserved for future
        use when accelerator credentials migrate into the settings hierarchy.

    Returns
    -------
    AcceleratorEndpoint | None
        A descriptor for the resolved endpoint, or ``None`` meaning "use the
        default Sonnet/OAuth path".

    Raises
    ------
    AcceleratorError
        When an explicitly named provider (anything other than ``"auto"``)
        cannot be satisfied (missing credentials, server unreachable, etc.).
    ValueError
        When *mode* is not one of the recognised values.
    """
    mode_lower = (mode or "").lower().strip()

    # --- off ---
    if mode_lower == "off":
        return None

    # --- explicit providers ---
    if mode_lower == "local":
        return _resolve_local(explicit=True)

    if mode_lower == "runpod":
        return _resolve_runpod(explicit=True)

    if mode_lower == "azure":
        return _resolve_azure(explicit=True)

    if mode_lower == "endpoint":
        return _resolve_endpoint(explicit=True)

    # --- auto ---
    if mode_lower == "auto":
        return _resolve_auto(sandbox_mode=sandbox_mode)

    raise ValueError(
        f"Unknown accelerator mode {mode!r}. "
        "Valid values: off, auto, local, runpod, azure, endpoint."
    )


def _resolve_auto(*, sandbox_mode: object) -> AcceleratorEndpoint | None:
    """Dynamic best-effort accelerator selection for ``mode="auto"``.

    Priority order:
    1. On-device NVIDIA GPU present AND local endpoint probes OK → ``"local"``.
    2. Sandbox is RunPod AND ``REPROLAB_ACCELERATOR_BASE_URL`` is set → ``"runpod"``.
    3. Azure credentials present → ``"azure"``.
    4. None (caller keeps Sonnet/OAuth path).

    Never raises.
    """
    # --- 1. Local GPU + local server ---
    try:
        from backend.services.runtime.gpu_resolution import host_supports_nvidia_gpu

        if host_supports_nvidia_gpu():
            ep = _resolve_local(explicit=False)
            if ep is not None:
                _log.info("accelerator[auto]: selected local (NVIDIA GPU + probe OK)")
                return ep
    except Exception as exc:  # noqa: BLE001
        _log.debug("accelerator[auto]: GPU check failed: %s", exc)

    # --- 2. RunPod sandbox + published proxy URL ---
    try:
        sandbox_str = str(sandbox_mode).lower() if sandbox_mode is not None else ""
        is_runpod_sandbox = "runpod" in sandbox_str
        if is_runpod_sandbox:
            ep = _resolve_runpod(explicit=False)
            if ep is not None:
                _log.info("accelerator[auto]: selected runpod (sandbox=runpod + probe OK)")
                return ep
    except Exception as exc:  # noqa: BLE001
        _log.debug("accelerator[auto]: runpod check failed: %s", exc)

    # --- 3. Azure credentials ---
    try:
        ep = _resolve_azure(explicit=False)
        if ep is not None:
            _log.info("accelerator[auto]: selected azure (credentials present)")
            return ep
    except Exception as exc:  # noqa: BLE001
        _log.debug("accelerator[auto]: azure check failed: %s", exc)

    _log.info(
        "accelerator[auto]: no provider satisfied; returning None (using default Sonnet/OAuth)"
    )
    return None


def build_accelerator_client(ep: AcceleratorEndpoint) -> object:
    """Return an LlmClient bound to *ep*.

    The returned object has a ``.complete(*, system: str, user: str) -> str``
    method matching the interface of
    :class:`~backend.services.context.workspace.tools.openai_client.OpenAILlmClient`.

    * Non-Azure endpoints → :class:`OpenAILlmClient` with ``base_url`` +
      ``api_key`` + ``model`` from the endpoint descriptor.
    * Azure endpoints (``ep.is_azure is True``) →
      :class:`AzureOpenAILlmClient` with ``azure_endpoint`` = ``ep.base_url``,
      ``azure_deployment`` = ``ep.model``, ``api_key`` = ``ep.api_key``.

    Parameters
    ----------
    ep:
        A resolved :class:`AcceleratorEndpoint` (never ``None``).

    Returns
    -------
    object
        An LlmClient with ``.complete(*, system, user) -> str``.
    """
    if ep.is_azure:
        from backend.services.context.workspace.tools.azure_openai_client import (
            AzureOpenAILlmClient,
        )

        return AzureOpenAILlmClient(
            model=ep.model,
            api_key=ep.api_key,
            azure_endpoint=ep.base_url,
            azure_deployment=ep.model,
        )

    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    return OpenAILlmClient(
        model=ep.model,
        api_key=ep.api_key,
        base_url=ep.base_url,
    )
