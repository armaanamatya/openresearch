"""Unit tests for AssetResolverV2 (OPENRESEARCH_ASSET_RESOLVER_V2, default OFF).

All tests use injected fakes — no real GCS, no real network, no real HF Hub.

Coverage:
  1. GCS-cache hit returns immediately without calling any network fetcher.
  2. GCS miss → first successful source is used + write-through to fake GCS store.
  3. Fetcher priority: HF tried before URL before gdrive; None falls through.
  4. A network fetcher that raises is treated as None (fail-soft, not propagated).
  5. All sources exhausted → verified Exclusion, never a fake path.
  6. Flag OFF (OPENRESEARCH_ASSET_RESOLVER_V2 unset) → V1 path, V2 code not taken.
  7. content_key is stable and collision-resistant for distinct coordinates.
  8. Gated asset without credential → gated Exclusion (same as V1).
  9. Framework / image / service kinds → same fast paths as V1.
  10. Checksum mismatch on URL fetch → Exclusion (not ok).
  11. resolve_all processes every asset; one failure does not abort the rest.
  12. GCS download failure falls through to network sources (fail-soft).
"""

from backend.agents.rlm.exclusion import KIND_ENV_SETUP_FAILED
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.asset_resolver_v2 import (
    AssetResolverV2,
    InMemoryGcsStore,
    content_key,
    v2_enabled,
)
from backend.services.runtime.credential_broker import CredentialBroker
from backend.services.runtime.run_plan import RequiredAsset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolver(tmp_path, *, gcs=None, hf=None, url=None, gdrive=None, recipe=None):
    """Convenience factory: always passes a no-HF-token broker so gated checks work."""
    return AssetResolverV2(
        gcs_store=gcs if gcs is not None else InMemoryGcsStore(),
        broker=CredentialBroker(env={}),
        hf_source=hf,
        url_source=url,
        gdrive_source=gdrive,
        recipe_lookup=recipe if recipe is not None else (lambda _: None),
    )


def _noop_hf(ident, dest):
    """HF source that always returns None (can't find / wrong shape)."""
    return None


def _noop_url(ident, dest):
    """URL source that always returns None."""
    return None


def _file_writer(content: bytes = b"asset-data"):
    """Returns a source callable that writes ``content`` to dest and returns it."""
    def _source(ident, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest
    return _source


# ---------------------------------------------------------------------------
# 1. GCS cache HIT — no network fetcher is called
# ---------------------------------------------------------------------------


def test_gcs_cache_hit_returns_immediately_without_network_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    network_calls: list[str] = []

    def _tracking_hf(ident, dest):
        network_calls.append("hf")
        return None

    def _tracking_url(ident, dest):
        network_calls.append("url")
        return None

    gcs = InMemoryGcsStore()
    asset = RequiredAsset("dataset", "https://example.com/data.tar.gz")
    # Pre-seed the GCS cache with the expected key
    ck = content_key(asset)
    gcs_key = f"assets/{ck}/data.tar.gz"
    gcs.seed(gcs_key, b"cached-content")

    resolver = _resolver(tmp_path, gcs=gcs, hf=_tracking_hf, url=_tracking_url)
    res = resolver.resolve(asset, AssetCache(tmp_path))

    assert res.ok, f"expected ok; detail={res.detail}"
    assert res.local_path is not None
    assert "gcs cache" in res.detail
    # No network fetcher was invoked
    assert network_calls == [], f"unexpected network calls: {network_calls}"


# ---------------------------------------------------------------------------
# 2. GCS miss → source fetch + write-through
# ---------------------------------------------------------------------------


def test_gcs_miss_fetches_source_and_writes_through(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    gcs = InMemoryGcsStore()
    asset = RequiredAsset("dataset", "https://example.com/dataset.zip")
    ck = content_key(asset)
    expected_gcs_key = f"assets/{ck}/dataset.zip"

    # GCS is empty → miss → should fall through to URL source
    resolver = _resolver(tmp_path, gcs=gcs, hf=_noop_hf, url=_file_writer(b"zip-content"))
    res = resolver.resolve(asset, AssetCache(tmp_path))

    assert res.ok, f"expected ok; detail={res.detail}"
    # Write-through: upload must have happened
    assert len(gcs.upload_calls) == 1, "expected exactly one GCS upload (write-through)"
    uploaded_key, uploaded_data = gcs.upload_calls[0]
    assert uploaded_key == expected_gcs_key, (
        f"upload key mismatch: got {uploaded_key!r}, want {expected_gcs_key!r}"
    )
    assert uploaded_data == b"zip-content"
    # Subsequent resolve hits GCS
    gcs.upload_calls.clear()
    network2: list[str] = []
    resolver2 = _resolver(
        tmp_path,
        gcs=gcs,
        hf=lambda i, d: network2.append("hf") or None,
        url=lambda i, d: network2.append("url") or None,
    )
    res2 = resolver2.resolve(asset, AssetCache(tmp_path))
    assert res2.ok
    assert network2 == [], "expected GCS hit on second resolve; no network calls"


# ---------------------------------------------------------------------------
# 3. Priority order: HF → URL → gdrive; None falls through
# ---------------------------------------------------------------------------


def test_fetcher_priority_hf_before_url_before_gdrive(tmp_path, monkeypatch):
    """Sources are tried in strict order; a None return falls through to the next."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    call_log: list[str] = []

    def tracking_hf(ident, dest):
        call_log.append("hf")
        return None  # can't handle → fall through

    def tracking_url(ident, dest):
        call_log.append("url")
        return None  # also can't → fall through

    def tracking_gdrive(ident, dest):
        call_log.append("gdrive")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"gdrive-data")
        return dest  # gdrive succeeds

    resolver = AssetResolverV2(
        gcs_store=InMemoryGcsStore(),
        broker=CredentialBroker(env={}),
        hf_source=tracking_hf,
        url_source=tracking_url,
        gdrive_source=tracking_gdrive,
        recipe_lookup=lambda _: None,
    )
    res = resolver.resolve(RequiredAsset("dataset", "some-identifier"), AssetCache(tmp_path))

    assert res.ok, f"expected ok; detail={res.detail}"
    assert call_log == ["hf", "url", "gdrive"], (
        f"expected [hf, url, gdrive] in strict order, got {call_log}"
    )


def test_hf_success_short_circuits_url_and_gdrive(tmp_path, monkeypatch):
    """If HF succeeds, URL and gdrive are never tried."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    call_log: list[str] = []

    def tracking_hf(ident, dest):
        call_log.append("hf")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"hf-data")
        return dest

    def tracking_url(ident, dest):
        call_log.append("url")
        return None

    def tracking_gdrive(ident, dest):
        call_log.append("gdrive")
        return None

    resolver = AssetResolverV2(
        gcs_store=InMemoryGcsStore(),
        broker=CredentialBroker(env={}),
        hf_source=tracking_hf,
        url_source=tracking_url,
        gdrive_source=tracking_gdrive,
        recipe_lookup=lambda _: None,
    )
    res = resolver.resolve(RequiredAsset("weights", "owner/model"), AssetCache(tmp_path))

    assert res.ok
    assert call_log == ["hf"], f"expected only hf called; got {call_log}"


# ---------------------------------------------------------------------------
# 4. Raising fetcher treated as None (fail-soft)
# ---------------------------------------------------------------------------


def test_raising_fetcher_treated_as_none_falls_through(tmp_path, monkeypatch):
    """A source that raises must not propagate the exception — it is treated as None."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    def exploding_hf(ident, dest):
        raise RuntimeError("simulated HF network failure")

    def working_url(ident, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"url-data")
        return dest

    resolver = _resolver(tmp_path, hf=exploding_hf, url=working_url)
    # Must succeed (exploding HF treated as None, URL takes over)
    res = resolver.resolve(RequiredAsset("dataset", "https://example.com/d.tar"), AssetCache(tmp_path))

    assert res.ok, f"expected ok after HF raise; detail={res.detail}"
    assert res.exclusion is None


def test_all_raisers_produces_verified_exclusion(tmp_path, monkeypatch):
    """If every source raises, the result is a verified Exclusion (never fake-ok)."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    def boom(ident, dest):
        raise ConnectionError("network down")

    resolver = _resolver(tmp_path, hf=boom, url=boom)
    res = resolver.resolve(RequiredAsset("dataset", "something"), AssetCache(tmp_path))

    assert not res.ok
    assert res.exclusion is not None
    assert res.exclusion.verified, "Exclusion must be verified=True"


# ---------------------------------------------------------------------------
# 5. All sources exhausted → verified Exclusion
# ---------------------------------------------------------------------------


def test_all_sources_exhausted_yields_verified_exclusion(tmp_path, monkeypatch):
    """When every tier returns None the result is a verified Exclusion, not a fake path."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    resolver = _resolver(tmp_path, hf=_noop_hf, url=_noop_url)
    # No GCS content, no sources succeed
    res = resolver.resolve(RequiredAsset("dataset", "totally-unknown"), AssetCache(tmp_path))

    assert not res.ok, "expected failure when all tiers exhausted"
    assert res.local_path is None, "must not return a fake path"
    assert res.exclusion is not None, "must produce an Exclusion"
    assert res.exclusion.verified, "Exclusion must be harness-verified"
    assert res.exclusion.kind == KIND_ENV_SETUP_FAILED


# ---------------------------------------------------------------------------
# 6. Flag OFF → V1 path, V2 code not taken
# ---------------------------------------------------------------------------


def test_flag_off_delegates_to_v1_not_v2(tmp_path, monkeypatch):
    """Flag OFF: resolve() delegates to V1; the GCS store is never touched."""
    monkeypatch.delenv("OPENRESEARCH_ASSET_RESOLVER_V2", raising=False)

    assert not v2_enabled(), "precondition: flag must be off"

    gcs = InMemoryGcsStore()
    v2_called: list[bool] = []

    # We inject a recipe_lookup that marks whether V2's _resolve_downloadable ran.
    # In V1 delegation, the V1 resolver's own recipe_lookup (default) is used,
    # NOT the V2 resolver's injectable recipe_lookup.  So if v2_called is populated
    # it means V2's own dispatch ran — which it must NOT do when the flag is off.

    class _MarkingGcs(InMemoryGcsStore):
        def exists(self, key):
            v2_called.append(True)
            return False

    marking_gcs = _MarkingGcs()
    resolver = AssetResolverV2(
        gcs_store=marking_gcs,
        broker=CredentialBroker(env={}),
        # V1 fallback will use its own default hf_snapshot (which will fail or
        # return quickly for a non-HF identifier, but crucially NOT hit the gcs tier)
        hf_source=_noop_hf,
        url_source=_noop_url,
        recipe_lookup=lambda _: None,
    )

    # With flag OFF, the _MarkingGcs.exists() must never be called (V2 not entered)
    resolver.resolve(RequiredAsset("framework", "pytorch"), AssetCache(tmp_path))

    assert v2_called == [], (
        "GCS store was accessed even though flag is OFF — V2 code must not run"
    )


def test_flag_on_uses_v2_path(tmp_path, monkeypatch):
    """Flag ON: V2 code runs; GCS store is queried."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    assert v2_enabled()

    gcs_queried: list[str] = []

    class _TrackingGcs(InMemoryGcsStore):
        def exists(self, key):
            gcs_queried.append(key)
            return False  # always miss

    resolver = _resolver(
        tmp_path,
        gcs=_TrackingGcs(),
        hf=_file_writer(b"data"),
        url=_noop_url,
    )
    res = resolver.resolve(RequiredAsset("dataset", "owner/data"), AssetCache(tmp_path))
    assert res.ok
    assert len(gcs_queried) >= 1, "GCS.exists must be called in V2 path"


# ---------------------------------------------------------------------------
# 7. content_key stability and collision resistance
# ---------------------------------------------------------------------------


def test_content_key_is_stable():
    """Same asset always produces the same key."""
    a = RequiredAsset("dataset", "owner/data")
    assert content_key(a) == content_key(a)
    assert content_key(a) == content_key(RequiredAsset("dataset", "owner/data"))


def test_content_key_collision_resistant_distinct_coordinates():
    """Distinct coordinates produce distinct keys."""
    a1 = RequiredAsset("dataset", "alpha/beta")
    a2 = RequiredAsset("dataset", "alpha/gamma")
    a3 = RequiredAsset("weights", "alpha/beta")
    assert content_key(a1) != content_key(a2), "different identifiers must differ"
    assert content_key(a1) != content_key(a3), "different kinds must differ"


def test_content_key_incorporates_optional_revision_and_checksum():
    """Optional revision/checksum fields change the key."""
    import types

    base = RequiredAsset("dataset", "owner/data")
    # Manufacture a variant with extra attrs (simulating a future RequiredAsset extension)
    with_rev = types.SimpleNamespace(kind="dataset", identifier="owner/data", revision="v2", checksum="")
    with_csum = types.SimpleNamespace(kind="dataset", identifier="owner/data", revision="", checksum="sha256:abc")
    assert content_key(base) != content_key(with_rev)
    assert content_key(base) != content_key(with_csum)
    assert content_key(with_rev) != content_key(with_csum)


def test_content_key_never_raises_on_malformed_input():
    """content_key must never raise even with a malformed asset."""
    import types

    bad = types.SimpleNamespace()  # no kind, no identifier
    result = content_key(bad)
    assert isinstance(result, str) and len(result) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# 8. Gated asset without credential
# ---------------------------------------------------------------------------


def test_gated_asset_without_hf_token_is_gated_exclusion(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    resolver = _resolver(tmp_path)  # broker has no HF_TOKEN
    res = resolver.resolve(
        RequiredAsset("dataset", "org/private-dataset", gated=True), AssetCache(tmp_path)
    )

    assert not res.ok
    assert res.exclusion is not None and res.exclusion.verified
    assert "gated" in res.exclusion.reason.lower()


# ---------------------------------------------------------------------------
# 9. Framework / image / service fast paths
# ---------------------------------------------------------------------------


def test_framework_resolved_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    resolver = _resolver(tmp_path)
    res = resolver.resolve(RequiredAsset("framework", "pytorch"), AssetCache(tmp_path))

    assert res.ok
    assert res.env_vars.get("cuda"), "expected cuda env var from framework resolution"


def test_image_and_service_kinds_are_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    resolver = _resolver(tmp_path)
    for kind in ("image", "service"):
        res = resolver.resolve(RequiredAsset(kind, "some-thing"), AssetCache(tmp_path))
        assert res.ok and res.exclusion is None, f"{kind} kind should be a no-op ok"


def test_unknown_kind_yields_verified_exclusion(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    resolver = _resolver(tmp_path)
    res = resolver.resolve(RequiredAsset("magical-unicorn", "x"), AssetCache(tmp_path))

    assert not res.ok
    assert res.exclusion is not None and res.exclusion.verified


# ---------------------------------------------------------------------------
# 10. Checksum mismatch → Exclusion
# ---------------------------------------------------------------------------


def test_checksum_mismatch_produces_exclusion(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    import types

    # Asset with a declared checksum that won't match the written bytes
    asset = types.SimpleNamespace(
        kind="dataset",
        identifier="https://example.com/data.bin",
        gated=False,
        size_hint_gb=None,
        checksum="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        revision=None,
    )

    resolver = AssetResolverV2(
        gcs_store=InMemoryGcsStore(),
        broker=CredentialBroker(env={}),
        hf_source=_noop_hf,
        url_source=_file_writer(b"real-data"),  # writes different bytes
        recipe_lookup=lambda _: None,
    )
    res = resolver.resolve(asset, AssetCache(tmp_path))  # type: ignore[arg-type]

    assert not res.ok, "checksum mismatch must not produce ok=True"
    assert res.exclusion is not None and res.exclusion.verified


# ---------------------------------------------------------------------------
# 11. resolve_all — one failure does not abort the rest
# ---------------------------------------------------------------------------


def test_resolve_all_one_failure_does_not_abort(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    def selective_url(ident, dest):
        if "good" in ident:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"ok")
            return dest
        return None  # bad asset returns None → Exclusion

    resolver = _resolver(tmp_path, hf=_noop_hf, url=selective_url)
    assets = [
        RequiredAsset("dataset", "https://example.com/good.zip"),
        RequiredAsset("dataset", "https://example.com/missing.zip"),
        RequiredAsset("dataset", "https://example.com/good2.zip"),
    ]
    results = resolver.resolve_all(assets, AssetCache(tmp_path))

    assert len(results) == 3
    assert results[0].ok, "first asset should succeed"
    assert not results[1].ok, "second asset should fail"
    assert results[2].ok, "third asset should succeed"


# ---------------------------------------------------------------------------
# 12. GCS download failure falls through to network sources
# ---------------------------------------------------------------------------


def test_gcs_download_failure_falls_through_to_network(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    class _BrokenGcs(InMemoryGcsStore):
        def exists(self, key):
            return True  # lie: pretend data exists

        def download(self, key, dest):
            raise IOError("GCS download failure")

    def working_url(ident, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"url-fallback")
        return dest

    resolver = _resolver(
        tmp_path, gcs=_BrokenGcs(), hf=_noop_hf, url=working_url
    )
    res = resolver.resolve(
        RequiredAsset("dataset", "https://example.com/d.zip"), AssetCache(tmp_path)
    )

    assert res.ok, f"expected ok (GCS error fell through to URL); detail={res.detail}"


# ---------------------------------------------------------------------------
# 13. Recipe registry hit (tier 0) — no GCS, no download
# ---------------------------------------------------------------------------


def test_recipe_hit_bypasses_gcs_and_network(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    gcs_queried: list[str] = []

    class _TrackGcs(InMemoryGcsStore):
        def exists(self, key):
            gcs_queried.append(key)
            return False

    network_called: list[str] = []

    resolver = AssetResolverV2(
        gcs_store=_TrackGcs(),
        broker=CredentialBroker(env={}),
        hf_source=lambda i, d: network_called.append("hf") or None,
        url_source=lambda i, d: network_called.append("url") or None,
        recipe_lookup=lambda name: object() if name == "cifar-10" else None,
    )
    res = resolver.resolve(RequiredAsset("dataset", "cifar-10"), AssetCache(tmp_path))

    assert res.ok
    assert "recipe" in res.detail
    assert gcs_queried == [], "GCS must not be queried on recipe hit"
    assert network_called == [], "network must not be called on recipe hit"
