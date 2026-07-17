"""GKE default GPU resolution — the single-GPU common case must resolve to ONE GPU.

Defect (production, 6/6 recent GKE runs died before training):
``OPENRESEARCH_FORCE_SINGLE_GPU`` defaults True while the ONLY provisioned GKE
pool was ``gcp_a100_80x8`` (8xA100-80). That gave a single-GPU paper two ways to
fail and no way to succeed:

  * with a usable VRAM estimate -> ``_provisioned_ladder`` filters to
    ``gpu_count == 1`` rows, the x8 pool is excluded, the ladder is EMPTY ->
    ``GpuResolutionError``;
  * with no/low-confidence estimate -> the fallback path picked the only
    provisioned pool and built the plan with ``gpu_count=sku.gpu_count`` --
    silently handing back **8 GPUs** despite force_single_gpu=True.

The fix ships a multi-SKU default pool set (a 1-GPU ``gcp_a100_80`` pool PLUS the
8-GPU ``gcp_a100_80x8`` pool, same nvidia-a100-80gb quota family = one quota ask)
and clamps ``gpu_count`` to 1 on every force_single_gpu path.

Also locks the per-GPU price semantics: ``max_gpu_usd_per_hour`` is a PER-GPU cap
but the catalog stores the WHOLE-MACHINE rate, so comparing them directly made a
genuine multi-GPU node ($31.44/machine = $3.93/GPU) fail the default $10/GPU cap
-- i.e. a real multi-GPU paper could not resolve either.

Pure tests: no GCP, no network, no GPU.
"""
from __future__ import annotations

import pytest

from backend.agents.schemas import GpuRequirements
from backend.config import Settings
from backend.services.runtime import gpu_catalog as cat
from backend.services.runtime import gpu_resolver as r

# The shipped default pool set (config.gcp_gpu_skus == terraform gpu_skus).
_DEFAULT_POOLS: tuple[str, ...] = ("gcp_a100_80", "gcp_a100_80x8")

# Production defaults from backend/config.py.
_DEFAULT_CAP = 10.0          # max_gpu_usd_per_hour (PER-GPU)
_DEFAULT_HEADROOM = 1.25     # dynamic_gpu_headroom


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch):
    """Assert the true SHIPPED code defaults, not this machine's .env.

    A pytest plugin loads the repo ``.env`` into ``os.environ`` at session start,
    and the repo ``.env`` pins ``OPENRESEARCH_GCP_GPU_SKUS`` -- which would shadow
    the Field default and make these assertions test the local operator's config
    instead of what actually ships. Same guard the existing
    tests/config/test_gcp_sku_pool_invariant.py fixture uses (both prefixes,
    because config._apply_legacy_env_aliases bridges REPROLAB_* <-> OPENRESEARCH_*).
    """
    for name in (
        "OPENRESEARCH_GCP_GPU_SKUS", "REPROLAB_GCP_GPU_SKUS",
        "OPENRESEARCH_FORCE_SINGLE_GPU", "REPROLAB_FORCE_SINGLE_GPU",
        "OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "REPROLAB_MAX_GPU_USD_PER_HOUR",
        "OPENRESEARCH_DYNAMIC_GPU_HEADROOM", "REPROLAB_DYNAMIC_GPU_HEADROOM",
    ):
        monkeypatch.delenv(name, raising=False)
    import backend.config as _config
    monkeypatch.setattr(_config, "_settings_cache", None, raising=False)
    yield
    monkeypatch.setattr(_config, "_settings_cache", None, raising=False)


def _req(vram: float | None, conf: float = 0.9, count: int = 1) -> GpuRequirements:
    return GpuRequirements(
        estimated_vram_gb=vram, confidence=conf, paper_gpu_count=count, rationale="t"
    )


def _resolve_gke(req: GpuRequirements, *, force_single_gpu: bool = True, **kw):
    """Resolve exactly as primitives.resolve_gpu_requirements does on --sandbox gke."""
    base = dict(
        dynamic_gpu_enabled=True,
        force_single_gpu=force_single_gpu,
        max_gpu_usd_per_hour=_DEFAULT_CAP,
        headroom_multiplier=_DEFAULT_HEADROOM,
        fallback_vram_gb=24,
        provider="gcp",
        provisioned_skus=_DEFAULT_POOLS,
    )
    base.update(kw)
    return r.resolve(req, **base)


def _sku(short_name: str) -> cat.GpuSku:
    return next(s for s in cat.CATALOG if s.short_name == short_name and s.provider == "gcp")


# ---------------------------------------------------------------------------
# The defaults themselves: config <-> terraform <-> catalog must agree
# ---------------------------------------------------------------------------

class TestDefaultPoolSet:
    def test_config_default_includes_a_single_gpu_pool(self):
        """The whole defect: with no 1-GPU pool, a 1-GPU paper cannot resolve."""
        s = Settings(_env_file=None)
        single = [n for n in s.gcp_gpu_skus if _sku(n).gpu_count == 1]
        assert single, (
            f"config.gcp_gpu_skus={s.gcp_gpu_skus!r} provisions no 1-GPU pool, but "
            f"force_single_gpu defaults True -> every single-GPU paper either raises "
            f"GpuResolutionError or gets silently over-provisioned."
        )

    def test_config_default_is_the_expected_pool_set(self):
        s = Settings(_env_file=None)
        assert s.gcp_gpu_skus == list(_DEFAULT_POOLS)

    def test_force_single_gpu_still_defaults_true(self):
        """We fixed the POOL SET, not by weakening the single-GPU default."""
        assert Settings(_env_file=None).force_single_gpu is True

    def test_default_pools_are_one_quota_family(self):
        """Both default pools are nvidia-a100-80gb => ONE quota request, not two."""
        assert all(_sku(n).vram_gb == 80 for n in _DEFAULT_POOLS)

    def test_module_constants_mirror_the_real_settings_defaults(self):
        """Guard the test's own premises.

        The resolution tests below assert behaviour "under the DEFAULT cap /
        headroom" using the module constants. If a real default drifts and these
        constants don't, those tests keep passing while no longer testing the
        shipped configuration at all -- a silently vacuous suite.
        """
        s = Settings(_env_file=None)
        assert s.max_gpu_usd_per_hour == pytest.approx(_DEFAULT_CAP)
        assert s.dynamic_gpu_headroom == pytest.approx(_DEFAULT_HEADROOM)


# ---------------------------------------------------------------------------
# THE headline case: a 1-GPU paper resolves to a 1-GPU SKU
# ---------------------------------------------------------------------------

class TestSingleGpuPaperResolvesToOneGpu:
    def test_single_gpu_paper_resolves_to_one_gpu(self):
        """Not an error, and NOT 8xA100."""
        plan = _resolve_gke(_req(30))
        assert plan.gpu_count == 1
        assert plan.short_name == "gcp_a100_80"
        assert _sku(plan.short_name).gpu_count == 1
        assert plan.runpod_id == "a2-ultragpu-1g"

    def test_single_gpu_paper_does_not_raise(self):
        """Regression: this raised GpuResolutionError with the x8-only pool set."""
        try:
            _resolve_gke(_req(30))
        except r.GpuResolutionError as exc:  # pragma: no cover - failure path
            pytest.fail(f"a 1-GPU paper must resolve, not raise: {exc}")

    @pytest.mark.parametrize("vram", [8, 16, 24, 40, 60, 64])
    def test_common_single_gpu_sizes_all_resolve_to_one_gpu(self, vram: int):
        plan = _resolve_gke(_req(vram))
        assert plan.gpu_count == 1, f"{vram}GB paper over-provisioned to {plan.gpu_count} GPUs"

    def test_low_confidence_fallback_is_not_eight_gpus(self):
        """The silent-8xA100 path: unestimatable paper fell back to the only pool (x8)
        and built the plan with gpu_count=8 despite force_single_gpu=True."""
        plan = _resolve_gke(_req(30, conf=0.1))
        assert plan.source == "fallback"
        assert plan.gpu_count == 1, "force_single_gpu was ignored on the fallback path"
        assert plan.short_name == "gcp_a100_80"

    def test_no_estimate_fallback_is_not_eight_gpus(self):
        plan = _resolve_gke(_req(None))
        assert plan.source == "fallback"
        assert plan.gpu_count == 1

    def test_dynamic_disabled_informational_is_not_eight_gpus(self):
        plan = _resolve_gke(_req(30), dynamic_gpu_enabled=False)
        assert plan.source == "informational"
        assert plan.gpu_count == 1

    def test_force_single_gpu_clamps_even_when_only_x8_is_provisioned(self):
        """A deployment that provisions ONLY the 8-GPU pool must still not silently
        lease 8 GPUs for a 1-GPU paper: we request 1 GPU on the bigger node."""
        plan = _resolve_gke(_req(None), provisioned_skus=("gcp_a100_80x8",))
        assert plan.gpu_count == 1, "silently over-provisioned to a full 8-GPU node"

    def test_one_gpu_plan_bills_one_gpu(self):
        plan = _resolve_gke(_req(30))
        assert plan.total_usd_per_hr == pytest.approx(plan.sku_usd_per_hr)
        assert plan.total_usd_per_hr == pytest.approx(3.93)


# ---------------------------------------------------------------------------
# A genuine multi-GPU paper still resolves correctly
# ---------------------------------------------------------------------------

class TestGenuineMultiGpuPaperStillResolves:
    def test_multi_gpu_paper_reaches_the_x8_pool(self):
        """force_single_gpu=False (the documented multi-GPU opt-in) + a big paper
        must reach the 8-GPU pool -- under the DEFAULT $10/GPU-hr cap."""
        plan = _resolve_gke(_req(400), force_single_gpu=False)
        assert plan.short_name == "gcp_a100_80x8"
        assert plan.gpu_count == 8
        assert plan.runpod_id == "a2-ultragpu-8g"

    def test_multi_gpu_paper_is_not_blocked_by_the_per_gpu_cap(self):
        """Regression: the cap ($10/GPU) was compared against the WHOLE-MACHINE rate
        ($31.44), so the x8 pool was excluded and a real multi-GPU paper raised."""
        try:
            _resolve_gke(_req(400), force_single_gpu=False)
        except r.GpuResolutionError as exc:  # pragma: no cover - failure path
            pytest.fail(
                f"a genuine multi-GPU paper must resolve under the default per-GPU "
                f"cap (a2-ultragpu-8g is $3.93/GPU, well under $10): {exc}"
            )

    def test_multi_gpu_plan_bills_the_true_machine_rate(self):
        """8 x $3.93/GPU = $31.44/hr -- the real a2-ultragpu-8g price. The old code
        reported 31.44 x 8 = $251.52/hr, an 8x phantom cost that spuriously tripped
        the run-USD budget cap."""
        plan = _resolve_gke(_req(400), force_single_gpu=False)
        assert plan.sku_usd_per_hr == pytest.approx(3.93)
        assert plan.total_usd_per_hr == pytest.approx(31.44)

    def test_multi_gpu_paper_that_fits_one_gpu_stays_on_one_gpu(self):
        """force_single_gpu=False must not mean 'always take the big node'."""
        plan = _resolve_gke(_req(30), force_single_gpu=False)
        assert plan.gpu_count == 1
        assert plan.short_name == "gcp_a100_80"

    def test_ladder_from_single_pool_reaches_x8_when_multi_gpu_allowed(self):
        plan = _resolve_gke(_req(30), force_single_gpu=False)
        assert "gcp_a100_80x8" in plan.ladder_remaining

    def test_force_single_gpu_ladder_never_escalates_to_multi_gpu(self):
        """force_single_gpu=True means the OOM ladder must not sneak in an 8-GPU node."""
        plan = _resolve_gke(_req(30))
        for name in plan.ladder_remaining:
            assert _sku(name).gpu_count == 1, f"ladder offers multi-GPU {name}"


# ---------------------------------------------------------------------------
# Per-GPU price semantics (GpuPlan schema: sku_usd_per_hr IS the per-GPU rate)
# ---------------------------------------------------------------------------

class TestPerGpuPriceSemantics:
    def test_usd_per_gpu_hr_divides_the_machine_rate(self):
        assert cat.usd_per_gpu_hr(_sku("gcp_a100_80x8")) == pytest.approx(31.44 / 8)
        assert cat.usd_per_gpu_hr(_sku("gcp_a100_80")) == pytest.approx(3.93)

    def test_usd_per_gpu_hr_is_identity_for_every_single_gpu_row(self):
        """This is what keeps the entire RunPod path byte-for-byte identical."""
        for sku in cat.CATALOG:
            if sku.gpu_count == 1:
                assert cat.usd_per_gpu_hr(sku) == pytest.approx(sku.approx_usd_per_hr)

    def test_every_runpod_row_is_single_gpu(self):
        """The premise of the byte-identical claim above."""
        assert all(s.gpu_count == 1 for s in cat.CATALOG if s.provider == "runpod")
