"""Unit tests for the generic AssetResolver (Phase 1d, Unit B).

Verbatim from ``docs/superpowers/plans/2026-07-01-phase-1d-credentials-assets-cpu-tier.md``
(Unit B). Hermetic: every fetcher (``hf_snapshot``/``url_fetch``/``recipe_lookup``)
is injected — no real network, no real HuggingFace Hub call.
"""

from backend.services.runtime.asset_resolver import AssetResolver, resolve_framework
from backend.services.runtime.run_plan import RequiredAsset
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.credential_broker import CredentialBroker


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
