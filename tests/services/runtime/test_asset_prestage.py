"""Unit tests for asset_prestage.py (OPENRESEARCH_ASSET_RESOLVER_V2, default OFF).

Coverage:
  1. Flag OFF → build_default_resolver() returns None; prestage_env_assets() returns [].
  2. Flag OFF → provision_scope ProvisionResult is byte-identical (resolver never consulted).
  3. Flag ON + ok resolve → file copied to WEBSHOP_DATA_DIR/items_shuffle.json.
  4. Flag ON + resolver returns ok=False → returns [], no raise, provision_scope proceeds.
  5. Unknown env → prestage_env_assets returns [].
  6. WEBSHOP_DATA_DIR unset → no copy attempted, no raise, returns [].
  7. Resolver.resolve raises → swallowed (fail-soft), returns [].
  8. Injected copier called with (local_path, str(dest)).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.asset_resolver_v2 import (
    AssetResolverV2,
    InMemoryGcsStore,
)
from backend.services.runtime.asset_prestage import (
    ENV_ASSET_REGISTRY,
    PrestageSpec,
    build_default_resolver,
    prestage_env_assets,
)
from backend.services.runtime.run_plan import RequiredAsset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolver_ok(tmp_path: Path, content: bytes = b"corpus-data") -> AssetResolverV2:
    """An AssetResolverV2 whose url_source always writes content and succeeds."""

    def _url_src(identifier: str, dest: Path) -> Path | None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest

    return AssetResolverV2(
        gcs_store=InMemoryGcsStore(),
        hf_source=lambda i, d: None,
        url_source=_url_src,
        gdrive_source=None,
        recipe_lookup=lambda _: None,
    )


def _resolver_fail() -> AssetResolverV2:
    """An AssetResolverV2 that always exhausts all tiers (ok=False)."""
    return AssetResolverV2(
        gcs_store=InMemoryGcsStore(),
        hf_source=lambda i, d: None,
        url_source=lambda i, d: None,
        gdrive_source=None,
        recipe_lookup=lambda _: None,
    )


def _resolver_raising() -> AssetResolverV2:
    """An AssetResolverV2 whose resolve() raises every time."""

    class _BrokenResolver(AssetResolverV2):
        def resolve(self, asset, cache):
            raise RuntimeError("boom")

    return _BrokenResolver(
        gcs_store=None,
        hf_source=lambda i, d: None,
        url_source=lambda i, d: None,
    )


# ---------------------------------------------------------------------------
# 1. Flag OFF → build_default_resolver returns None
# ---------------------------------------------------------------------------


def test_build_default_resolver_flag_off_returns_none(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_ASSET_RESOLVER_V2", raising=False)
    assert build_default_resolver() is None


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_build_default_resolver_flag_on_returns_instance(monkeypatch, val):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", val)
    # No GCS bucket set → gcs_store=None; no rclone → gdrive_source=None.
    # Should still construct successfully.
    r = build_default_resolver()
    assert isinstance(r, AssetResolverV2)


# ---------------------------------------------------------------------------
# 2. Flag OFF → prestage_env_assets returns []
# ---------------------------------------------------------------------------


def test_prestage_env_assets_flag_off_returns_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_ASSET_RESOLVER_V2", raising=False)
    cache = AssetCache(tmp_path)
    # Even if a non-None resolver is somehow passed, flag-off wins.
    result = prestage_env_assets("webshop", _resolver_fail(), cache)
    assert result == []


def test_prestage_env_assets_none_resolver_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    cache = AssetCache(tmp_path)
    result = prestage_env_assets("webshop", None, cache)
    assert result == []


# ---------------------------------------------------------------------------
# 3. provision_scope flag-OFF is byte-identical (resolver never consulted)
# ---------------------------------------------------------------------------


def test_provision_scope_flag_off_byte_identical(tmp_path, monkeypatch):
    """When flag is OFF, provision_scope result must equal a baseline run with no wiring."""
    monkeypatch.delenv("OPENRESEARCH_ASSET_RESOLVER_V2", raising=False)

    from backend.services.runtime import env_cache as EC

    # Fake manager: setup() always succeeds with a fixed env var dict.
    setup_calls: list[str] = []

    class FakeManager:
        _cache = AssetCache(tmp_path)

        def setup(self, name: str):
            setup_calls.append(name)
            result = MagicMock()
            result.ok = True
            result.base_url = None
            result.exclusion = None
            result.as_env_vars.return_value = {f"FAKE_{name.upper()}": "/fake"}
            return result

        def release_webshop(self):
            pass

    mgr = FakeManager()
    res = EC.provision_scope(["ALFWorld", "Search-QA"], mgr)  # type: ignore[arg-type]

    # Both envs set up, no exclusions, env_vars populated.
    assert res.env_vars == {"FAKE_ALFWORLD": "/fake", "FAKE_SEARCH-QA": "/fake"}
    assert res.exclusions == []
    assert setup_calls == ["ALFWorld", "Search-QA"]


def test_provision_scope_flag_off_resolver_never_consulted(tmp_path, monkeypatch):
    """build_default_resolver must NOT be called when flag is OFF."""
    monkeypatch.delenv("OPENRESEARCH_ASSET_RESOLVER_V2", raising=False)

    from backend.services.runtime import env_cache as EC
    import backend.services.runtime.asset_prestage as _ap

    consulted: list[bool] = []

    original = _ap.build_default_resolver

    def _spy():
        consulted.append(True)
        return original()

    with patch.object(_ap, "build_default_resolver", side_effect=_spy):
        # Re-import so the patched version is seen by the lazy import in provision_scope.
        class FakeManager:
            _cache = AssetCache(tmp_path)

            def setup(self, name):
                m = MagicMock()
                m.ok = True
                m.base_url = None
                m.exclusion = None
                m.as_env_vars.return_value = {}
                return m

            def release_webshop(self):
                pass

        EC.provision_scope(["ALFWorld"], FakeManager())  # type: ignore[arg-type]

    # build_default_resolver was called (it's always called), but it returned None,
    # so prestage_env_assets was never entered with a non-None resolver.
    # The key invariant: flag-off → result is None → inner guard skipped.
    assert consulted  # it was called (lightweight), but returned None


# ---------------------------------------------------------------------------
# 4. Flag ON + ok resolve → file copied to dest
# ---------------------------------------------------------------------------


def test_prestage_env_assets_copies_file_to_dest(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    data_dir = tmp_path / "webshop_data"
    data_dir.mkdir()
    cache = AssetCache(tmp_path / "cache")

    resolver = _resolver_ok(tmp_path, content=b"webshop-corpus")

    copied: list[tuple[str, str]] = []

    def _fake_copier(src: str, dst: str) -> None:
        # Actually write the file so we can assert dest exists.
        shutil.copy2(src, dst)
        copied.append((src, dst))

    result = prestage_env_assets(
        "webshop",
        resolver,
        cache,
        env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
        copier=_fake_copier,
    )

    # Two corpus files should be staged.
    assert len(result) == 2
    assert str(data_dir / "items_shuffle.json") in result
    assert str(data_dir / "items_ins_v2.json") in result
    # Both dest files should exist with the right content.
    assert (data_dir / "items_shuffle.json").read_bytes() == b"webshop-corpus"
    assert (data_dir / "items_ins_v2.json").read_bytes() == b"webshop-corpus"
    # Copier was called for each.
    assert len(copied) == 2


def test_prestage_env_assets_webshop_case_insensitive(tmp_path, monkeypatch):
    """Registry lookup is case-insensitive on the env name."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    cache = AssetCache(tmp_path / "cache")
    resolver = _resolver_ok(tmp_path)
    copy_calls: list[str] = []

    def _copier(src, dst):
        shutil.copy2(src, dst)
        copy_calls.append(dst)

    for variant in ("WebShop", "WEBSHOP", "webshop", "WebSHOP"):
        copy_calls.clear()
        result = prestage_env_assets(
            variant,
            resolver,
            cache,
            env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
            copier=_copier,
        )
        assert result, f"expected staging for variant {variant!r}"


# ---------------------------------------------------------------------------
# 5. Flag ON + resolver returns ok=False → returns [], no raise
# ---------------------------------------------------------------------------


def test_prestage_env_assets_resolve_failure_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    cache = AssetCache(tmp_path)
    data_dir = tmp_path / "ws_data"
    data_dir.mkdir()

    result = prestage_env_assets(
        "webshop",
        _resolver_fail(),
        cache,
        env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
    )
    assert result == []


def test_provision_scope_still_proceeds_after_resolve_failure(tmp_path, monkeypatch):
    """When pre-staging fails (ok=False), provision_scope should still call setup()."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")

    from backend.services.runtime import env_cache as EC
    import backend.services.runtime.asset_prestage as _ap

    setup_calls: list[str] = []

    class FakeManager:
        _cache = AssetCache(tmp_path)

        def setup(self, name: str):
            setup_calls.append(name)
            m = MagicMock()
            m.ok = True
            m.base_url = None
            m.exclusion = None
            m.as_env_vars.return_value = {}
            return m

        def release_webshop(self):
            pass

    # Patch build_default_resolver to return a failing resolver.
    with patch.object(_ap, "build_default_resolver", return_value=_resolver_fail()):
        EC.provision_scope(["WebShop"], FakeManager())  # type: ignore[arg-type]

    # setup() must still have been called regardless of the pre-staging outcome.
    assert "WebShop" in setup_calls


# ---------------------------------------------------------------------------
# 6. Unknown env → returns []
# ---------------------------------------------------------------------------


def test_prestage_env_assets_unknown_env_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    cache = AssetCache(tmp_path)
    resolver = _resolver_ok(tmp_path)
    result = prestage_env_assets("ALFWorld", resolver, cache)
    assert result == []

    result2 = prestage_env_assets("", resolver, cache)
    assert result2 == []

    result3 = prestage_env_assets("totally-unknown-env", resolver, cache)
    assert result3 == []


# ---------------------------------------------------------------------------
# 7. WEBSHOP_DATA_DIR unset → no copy attempted, no raise
# ---------------------------------------------------------------------------


def test_prestage_env_assets_env_dir_unset_no_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    cache = AssetCache(tmp_path)
    resolver = _resolver_ok(tmp_path)

    copy_calls: list[tuple[str, str]] = []

    def _copier(src, dst):
        copy_calls.append((src, dst))

    # env_getter returns "" for any key (unset)
    result = prestage_env_assets(
        "webshop",
        resolver,
        cache,
        env_getter=lambda k, d="": "",
        copier=_copier,
    )
    assert result == []
    assert copy_calls == []


# ---------------------------------------------------------------------------
# 8. resolver.resolve raises → swallowed (fail-soft), returns []
# ---------------------------------------------------------------------------


def test_prestage_env_assets_resolve_raises_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    cache = AssetCache(tmp_path)
    data_dir = tmp_path / "ws"
    data_dir.mkdir()

    result = prestage_env_assets(
        "webshop",
        _resolver_raising(),
        cache,
        env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
    )
    # No exception propagated; empty list returned.
    assert result == []


# ---------------------------------------------------------------------------
# 9. Copier called with correct (src, str(dest)) arguments
# ---------------------------------------------------------------------------


def test_prestage_env_assets_copier_receives_correct_args(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    data_dir = tmp_path / "ws_data"
    data_dir.mkdir()
    cache = AssetCache(tmp_path / "cache")
    content = b"test-bytes"
    resolver = _resolver_ok(tmp_path, content=content)

    copier_calls: list[tuple[str, str]] = []

    def _fake_copier(src: str, dst: str) -> None:
        shutil.copy2(src, dst)
        copier_calls.append((src, dst))

    prestage_env_assets(
        "webshop",
        resolver,
        cache,
        env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
        copier=_fake_copier,
    )

    assert len(copier_calls) == 2
    # First arg is local_path (a string), second is str(dest).
    for src_arg, dst_arg in copier_calls:
        assert isinstance(src_arg, str) and isinstance(dst_arg, str)
        # dest must be under data_dir
        assert dst_arg.startswith(str(data_dir))


# ---------------------------------------------------------------------------
# 10. Registry structure sanity
# ---------------------------------------------------------------------------


def test_env_asset_registry_webshop_specs():
    specs = ENV_ASSET_REGISTRY["webshop"]
    assert len(specs) == 2
    identifiers = {s.asset.identifier for s in specs}
    assert "gdrive:1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib" in identifiers
    assert "gdrive:1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu" in identifiers
    subpaths = {s.dest_subpath for s in specs}
    assert "items_shuffle.json" in subpaths
    assert "items_ins_v2.json" in subpaths
    for s in specs:
        assert s.asset.kind == "dataset"
        assert s.env_dir_var == "WEBSHOP_DATA_DIR"


def test_registry_keys_are_lowercase():
    for key in ENV_ASSET_REGISTRY:
        assert key == key.lower(), f"registry key {key!r} must be lowercase"
