from backend.services.runtime.env_adapters.base import (
    EnvironmentAdapter, EnvSetupResult, SmokeResult, HealthReport, ProvisionCtx, _fail,
)
from backend.agents.rlm import exclusion as X


def test_env_setup_result_env_vars_roundtrip():
    r = EnvSetupResult(env="ALFWorld", ok=True, data_path="/d")
    assert r.as_env_vars() == {"ALFWORLD_DATA": "/d"}
    assert EnvSetupResult(env="X", ok=False).as_env_vars() == {}


def test_fail_builds_verified_exclusion():
    r = _fail("WebShop", "boom", evidence="e")
    assert not r.ok and r.exclusion.verified and r.exclusion.kind == X.KIND_ENV_SETUP_FAILED
    assert r.exclusion.axis == X.AXIS_ENVIRONMENT and r.exclusion.item == "WebShop"


def test_default_smoke_and_health_are_safe():
    class _A(EnvironmentAdapter):
        key = "x"
        def applies(self, env_name): return env_name == "x"
        def provision(self, ctx): return EnvSetupResult(env="x", ok=True)
    a = _A()
    smoke = a.smoke(ProvisionCtx())
    assert isinstance(smoke, SmokeResult) and smoke.ok is True
    assert isinstance(a.health(ProvisionCtx(display_name="x")), HealthReport)
