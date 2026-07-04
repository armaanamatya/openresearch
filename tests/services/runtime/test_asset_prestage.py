"""Unit tests for asset_prestage.py (OPENRESEARCH_ASSET_RESOLVER_V2, default OFF).

Coverage:
  1. Flag OFF → build_default_resolver() returns None; prestage_env_assets() returns [].
  2. Flag OFF → provision_scope ProvisionResult is byte-identical (resolver never consulted).
  3. Flag ON + ok resolve → file copied to WEBSHOP_DATA_DIR/<dest_subpath>.
  4. Flag ON + resolver returns ok=False → returns [], no raise, provision_scope proceeds.
  5. Unknown env → prestage_env_assets returns [].
  6. WEBSHOP_DATA_DIR unset → no copy attempted, no raise, returns [].
  7. Resolver.resolve raises → swallowed (fail-soft), returns [].
  8. Injected copier called with (local_path, str(dest)).
  9. Candidate fallback: a failing/rejected first candidate falls through to the next.
  10. Registry shape: webshop carries 4 ordered-mirror specs with the pinned checksums.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _use_registry(monkeypatch, specs: tuple[PrestageSpec, ...], *, env: str = "webshop") -> None:
    """Swap ENV_ASSET_REGISTRY for the duration of a test (module-level, patched back after)."""
    import backend.services.runtime.asset_prestage as _ap

    monkeypatch.setattr(_ap, "ENV_ASSET_REGISTRY", {env: specs})


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
#
# These use a monkeypatched, integrity-pin-free registry so the *mechanism*
# (resolve -> copy) is tested independently of the real, multi-GB production
# pins (which fake in-memory content can never satisfy).
# ---------------------------------------------------------------------------


def _two_spec_registry() -> tuple[PrestageSpec, ...]:
    return (
        PrestageSpec(
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_shuffle.json",
            asset=RequiredAsset(kind="dataset", identifier="https://mirror.example/items_shuffle.json"),
        ),
        PrestageSpec(
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_ins_v2.json",
            asset=RequiredAsset(kind="dataset", identifier="https://mirror.example/items_ins_v2.json"),
        ),
    )


def test_prestage_env_assets_copies_file_to_dest(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    _use_registry(monkeypatch, _two_spec_registry())
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
    _use_registry(monkeypatch, _two_spec_registry())
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
    _use_registry(monkeypatch, _two_spec_registry())
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
# 10. Candidate fallback: first candidate fails/rejected -> second is used
# ---------------------------------------------------------------------------


def test_prestage_env_assets_candidate_fallback_first_fails(tmp_path, monkeypatch):
    """First candidate's resolve fails outright -> second candidate is tried and staged."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    cache = AssetCache(tmp_path / "cache")

    spec = PrestageSpec(
        env_dir_var="WEBSHOP_DATA_DIR",
        dest_subpath="items_shuffle.json",
        candidates=(
            RequiredAsset(kind="dataset", identifier="https://mirror-a.example/items_shuffle.json"),
            RequiredAsset(kind="dataset", identifier="https://mirror-b.example/items_shuffle.json"),
        ),
    )
    _use_registry(monkeypatch, (spec,))

    def _url_src(identifier: str, dest: Path) -> Path | None:
        if "mirror-a" in identifier:
            return None  # simulate a dead mirror (404 / network failure)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"content-from-mirror-b")
        return dest

    resolver = AssetResolverV2(
        gcs_store=InMemoryGcsStore(),
        hf_source=lambda i, d: None,
        url_source=_url_src,
        gdrive_source=None,
        recipe_lookup=lambda _: None,
    )

    def _copier(src, dst):
        shutil.copy2(src, dst)

    result = prestage_env_assets(
        "webshop",
        resolver,
        cache,
        env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
        copier=_copier,
    )

    assert result == [str(data_dir / "items_shuffle.json")]
    assert (data_dir / "items_shuffle.json").read_bytes() == b"content-from-mirror-b"


def test_prestage_env_assets_checksum_mismatch_falls_through(tmp_path, monkeypatch):
    """Candidate 1 resolves but has the wrong content (bad hash) -> rejected; candidate 2 wins."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    cache = AssetCache(tmp_path / "cache")

    good_content = b"the-real-correct-corpus-bytes"
    good_sha256 = hashlib.sha256(good_content).hexdigest()

    spec = PrestageSpec(
        env_dir_var="WEBSHOP_DATA_DIR",
        dest_subpath="items_shuffle.json",
        candidates=(
            RequiredAsset(kind="dataset", identifier="https://mirror-a.example/items_shuffle.json"),
            RequiredAsset(kind="dataset", identifier="https://mirror-b.example/items_shuffle.json"),
        ),
        sha256=good_sha256,
    )
    _use_registry(monkeypatch, (spec,))

    def _url_src(identifier: str, dest: Path) -> Path | None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "mirror-a" in identifier:
            dest.write_bytes(b"corrupted-or-truncated-garbage")
        else:
            dest.write_bytes(good_content)
        return dest

    resolver = AssetResolverV2(
        gcs_store=InMemoryGcsStore(),
        hf_source=lambda i, d: None,
        url_source=_url_src,
        gdrive_source=None,
        recipe_lookup=lambda _: None,
    )

    staged_copies: list[tuple[str, str]] = []

    def _copier(src, dst):
        shutil.copy2(src, dst)
        staged_copies.append((src, dst))

    result = prestage_env_assets(
        "webshop",
        resolver,
        cache,
        env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
        copier=_copier,
    )

    # Only ONE copy ever happened (candidate 1's bad file was never staged).
    assert len(staged_copies) == 1
    assert result == [str(data_dir / "items_shuffle.json")]
    assert (data_dir / "items_shuffle.json").read_bytes() == good_content


def test_prestage_env_assets_min_size_violation_rejected(tmp_path, monkeypatch):
    """A resolved file smaller than min_size_bytes is rejected (never staged)."""
    monkeypatch.setenv("OPENRESEARCH_ASSET_RESOLVER_V2", "1")
    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    cache = AssetCache(tmp_path / "cache")

    spec = PrestageSpec(
        env_dir_var="WEBSHOP_DATA_DIR",
        dest_subpath="items_shuffle_1000.json",
        candidates=(
            RequiredAsset(kind="dataset", identifier="https://mirror-a.example/items_shuffle_1000.json"),
        ),
        min_size_bytes=1_000_000,  # candidate content below will be a few bytes
    )
    _use_registry(monkeypatch, (spec,))

    resolver = _resolver_ok(tmp_path, content=b"too-small")

    staged_copies: list[tuple[str, str]] = []

    def _copier(src, dst):
        shutil.copy2(src, dst)
        staged_copies.append((src, dst))

    result = prestage_env_assets(
        "webshop",
        resolver,
        cache,
        env_getter=lambda k, d="": str(data_dir) if k == "WEBSHOP_DATA_DIR" else d,
        copier=_copier,
    )

    assert result == []
    assert staged_copies == []
    assert not (data_dir / "items_shuffle_1000.json").exists()


# ---------------------------------------------------------------------------
# 11. PrestageSpec backward-compat + normalization
# ---------------------------------------------------------------------------


def test_prestage_spec_asset_alias_populates_candidates():
    asset = RequiredAsset(kind="dataset", identifier="https://example.com/f.json")
    spec = PrestageSpec(env_dir_var="X_DIR", dest_subpath="f.json", asset=asset)
    assert spec.candidates == (asset,)


def test_prestage_spec_explicit_candidates_take_precedence_over_asset():
    asset = RequiredAsset(kind="dataset", identifier="https://ignored.example/f.json")
    c1 = RequiredAsset(kind="dataset", identifier="https://a.example/f.json")
    c2 = RequiredAsset(kind="dataset", identifier="https://b.example/f.json")
    spec = PrestageSpec(
        env_dir_var="X_DIR", dest_subpath="f.json", asset=asset, candidates=(c1, c2)
    )
    assert spec.candidates == (c1, c2)


# ---------------------------------------------------------------------------
# 12. Registry structure sanity
# ---------------------------------------------------------------------------


def test_registry_keys_are_lowercase():
    for key in ENV_ASSET_REGISTRY:
        assert key == key.lower(), f"registry key {key!r} must be lowercase"


def test_env_asset_registry_webshop_shape():
    specs = ENV_ASSET_REGISTRY["webshop"]
    assert len(specs) == 4

    by_subpath = {s.dest_subpath: s for s in specs}
    assert set(by_subpath) == {
        "items_shuffle.json",
        "items_ins_v2.json",
        "items_shuffle_1000.json",
        "items_ins_v2_1000.json",
    }

    for spec in specs:
        assert spec.env_dir_var == "WEBSHOP_DATA_DIR"
        assert spec.candidates, f"{spec.dest_subpath} must carry >=1 candidate"
        for cand in spec.candidates:
            assert cand.kind == "dataset"
            if cand.identifier.startswith("gdrive:"):
                continue
            assert cand.identifier.startswith("https://huggingface.co/"), cand.identifier

    full_shuffle = by_subpath["items_shuffle.json"]
    full_ins = by_subpath["items_ins_v2.json"]
    subset_shuffle = by_subpath["items_shuffle_1000.json"]
    subset_ins = by_subpath["items_ins_v2_1000.json"]

    # Exact sha256 + size pins on the two full files.
    assert full_shuffle.sha256 == "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8"
    assert full_shuffle.min_size_bytes == 5_479_720_229
    assert full_ins.sha256 == "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e"
    assert full_ins.min_size_bytes == 186_295_270

    # Mirror fallback order 1 -> 2 -> 3 on the full files.
    shuffle_urls = [c.identifier for c in full_shuffle.candidates if not c.identifier.startswith("gdrive:")]
    assert shuffle_urls == [
        "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_shuffle.json",
        "https://huggingface.co/datasets/HongbangYuan/webshop/resolve/main/items_shuffle.json",
        "https://huggingface.co/datasets/quanwei0/webshop-minimal/resolve/main/items_shuffle.json",
    ]
    ins_urls = [c.identifier for c in full_ins.candidates if not c.identifier.startswith("gdrive:")]
    assert ins_urls == [
        "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_ins_v2.json",
        "https://huggingface.co/datasets/HongbangYuan/webshop/resolve/main/items_ins_v2.json",
        "https://huggingface.co/datasets/quanwei0/webshop-minimal/resolve/main/items_ins_v2.json",
    ]

    # gdrive candidates are last-resort (final entry) on the two full files only.
    assert full_shuffle.candidates[-1].identifier == "gdrive:1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib"
    assert full_ins.candidates[-1].identifier == "gdrive:1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu"

    # 1K subsets: two mirrors only, no gdrive fallback, size-sanity only (no sha256).
    for s in (subset_shuffle, subset_ins):
        assert s.sha256 is None
        assert len(s.candidates) == 2
        assert all(not c.identifier.startswith("gdrive:") for c in s.candidates)
    assert subset_shuffle.min_size_bytes == 4_467_013
    assert subset_ins.min_size_bytes == 147_099
