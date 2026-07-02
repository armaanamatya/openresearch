"""Tests for the ``campaign`` CLI subcommand (Unit 9a, cli.py additions).

Hermetic: parser-level tests never touch services; ``cmd_campaign`` tests
monkeypatch the ingest chain (``_make_services``) and ``build_campaign`` to
fakes, so no sqlite, no network, no subprocess, and nothing written outside
``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import backend.cli as cli
from backend.agents.rlm.reproduction_campaign import CampaignLedgerError

_MONEY_FLAGS = {"--max-llm-usd": "12", "--max-gpu-usd": "34", "--max-gpu-hours": "5"}


@pytest.fixture(autouse=True)
def _clean_campaign_env(monkeypatch):
    for var in (
        "OPENRESEARCH_CAMPAIGN_MAX_ATTEMPTS",
        "OPENRESEARCH_CAMPAIGN_WALL_CLOCK_S",
        "OPENRESEARCH_CAMPAIGN_MODE",
        "OPENRESEARCH_CAMPAIGN_DRIVER",
        "OPENRESEARCH_CAMPAIGN_WIDTH",
        "OPENRESEARCH_CAMPAIGN_PLATEAU_K",
        "OPENRESEARCH_CAMPAIGN_REQUIRE_CPU_TIER",
    ):
        monkeypatch.delenv(var, raising=False)


def _campaign_argv(*extra: str, runs_root: str | None = None, source: str = "2605.15155") -> list[str]:
    argv: list[str] = []
    if runs_root is not None:
        argv += ["--runs-root", runs_root]
    argv += ["campaign", source]
    for flag, value in _MONEY_FLAGS.items():
        argv += [flag, value]
    argv += list(extra)
    return argv


# --------------------------------------------------------------------------- #
# Parser                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("missing", sorted(_MONEY_FLAGS))
def test_campaign_requires_money_meters(missing, capsys):
    argv = ["campaign", "2605.15155"]
    for flag, value in _MONEY_FLAGS.items():
        if flag != missing:
            argv += [flag, value]

    with pytest.raises(SystemExit) as excinfo:
        cli._build_parser().parse_args(argv)

    assert excinfo.value.code == 2
    assert missing in capsys.readouterr().err  # argparse names the missing required flag


@pytest.mark.parametrize("driver", ["unified", "paired"])
def test_campaign_accepts_unified_and_paired_driver(driver):
    args = cli._build_parser().parse_args(_campaign_argv("--campaign-driver", driver))
    assert args.driver == driver


def test_campaign_driver_env_default_unified(monkeypatch):
    # argparse re-parses a string default through type=, so the env default
    # is validated exactly like an explicit flag.
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_DRIVER", "unified")
    args = cli._build_parser().parse_args(_campaign_argv())
    assert args.driver == "unified"


def test_campaign_rejects_garbage_driver_value(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli._build_parser().parse_args(_campaign_argv("--campaign-driver", "bogus"))

    assert excinfo.value.code == 2
    assert "invalid --campaign-driver" in capsys.readouterr().err


def test_campaign_parser_defaults():
    args = cli._build_parser().parse_args(_campaign_argv())

    assert args.cmd == "campaign"
    assert args.func is cli.cmd_campaign
    assert args.max_llm_usd == 12.0
    assert args.max_gpu_usd == 34.0
    assert args.max_gpu_hours == 5.0
    assert args.max_attempts == 6
    assert args.wall_clock_s is None
    assert args.mode == "unattended"
    assert args.driver == "live"
    assert args.width == 1
    assert args.plateau_k == 2
    assert args.sandbox == "local"
    assert args.billing_sandbox is None
    assert args.gpu_usd_per_hr is None
    assert args.est_gpu_hours == 2.0
    assert args.run_spec == "configs/campaign_run_spec.json"
    assert args.paper_class == "generic"
    assert args.require_cpu_tier is False
    assert args.resume is False


def test_campaign_env_defaults(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_MODE", "checkpoint")
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_WIDTH", "2")
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_PLATEAU_K", "4")
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_WALL_CLOCK_S", "7200")
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_REQUIRE_CPU_TIER", "1")

    args = cli._build_parser().parse_args(_campaign_argv())

    assert args.max_attempts == 3
    assert args.mode == "checkpoint"
    assert args.width == 2
    assert args.plateau_k == 4
    assert args.wall_clock_s == 7200.0
    assert args.require_cpu_tier is True


def test_campaign_garbage_int_env_does_not_break_parser(monkeypatch):
    # int-typed env defaults are evaluated at PARSER BUILD time; garbage must
    # fall back rather than break every subcommand's parsing.
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_MAX_ATTEMPTS", "not-a-number")
    args = cli._build_parser().parse_args(_campaign_argv())
    assert args.max_attempts == 6


def test_existing_reproduce_parser_untouched():
    args = cli._build_parser().parse_args(
        ["reproduce", "2605.15155", "--mode", "rlm", "--sandbox", "local",
         "--max-usd", "5", "--run-spec", "spec.json", "--scope-spec", "{}"]
    )

    assert args.cmd == "reproduce"
    assert args.func is cli.cmd_reproduce
    assert args.mode == "rlm"
    assert args.sandbox == "local"
    assert args.max_usd == 5.0
    assert args.run_spec == "spec.json"
    assert args.scope_spec == "{}"
    # Known reproduce flags still present with their defaults.
    assert args.paper_hint is None
    assert args.resume is False
    assert args.project_id is None


# --------------------------------------------------------------------------- #
# cmd_campaign wiring                                                          #
# --------------------------------------------------------------------------- #


class _FakeStore:
    def close(self) -> None:
        pass


class _FakeIntake:
    def __init__(self, project_id: str) -> None:
        self._project_id = project_id
        self.registered: list = []
        self.fetched: list = []

    def register_project(self, cmd, project_id_override=None):
        self.registered.append(cmd)
        return self._project_id

    def fetch_paper(self, cmd):
        self.fetched.append(cmd)
        return True


class _FakeStep:
    def __init__(self) -> None:
        self.calls: list = []

    def _ok(self, cmd):
        self.calls.append(cmd)
        return True

    start_parsing = _ok
    discover = _ok
    start_indexing = _ok
    build_workspace = _ok


def _install_fakes(monkeypatch, *, project_id: str = "prj_fake"):
    intake = _FakeIntake(project_id)
    parser_svc, discovery, indexer, workspace = _FakeStep(), _FakeStep(), _FakeStep(), _FakeStep()

    def _fake_make_services(database_url, runs_root):
        return _FakeStore(), intake, parser_svc, discovery, indexer, workspace

    monkeypatch.setattr(cli, "_make_services", _fake_make_services)
    return intake, parser_svc, discovery, indexer, workspace


def _install_fake_campaign(monkeypatch, outcome=None, *, raise_ledger_error: bool = False):
    captured: dict = {}

    class _FakeCampaign:
        def run(self):
            if raise_ledger_error:
                raise CampaignLedgerError("disk full")
            return outcome

    def _fake_build_campaign(project_id, opts):
        captured["project_id"] = project_id
        captured["opts"] = opts
        return _FakeCampaign()

    monkeypatch.setattr(cli, "build_campaign", _fake_build_campaign)
    return captured


def test_cmd_campaign_wires_options(tmp_path, monkeypatch):
    intake, parser_svc, discovery, indexer, workspace = _install_fakes(monkeypatch)
    outcome = {"kind": "EXHAUSTED", "rule": "max_attempts", "stop_reason": "max_attempts",
               "champion_attempt_n": 2, "spent": {"llm_usd": 1.0}}
    captured = _install_fake_campaign(monkeypatch, outcome)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")

    args = cli._build_parser().parse_args(
        _campaign_argv(
            "--mode", "checkpoint", "--width", "2", "--plateau-k", "3",
            "--sandbox", "local", "--billing-sandbox", "gcp",
            "--gpu-usd-per-hr", "2.5", "--est-gpu-hours", "1.5",
            "--run-spec", str(spec_path), "--paper-class", "vision",
            "--wall-clock-s", "3600", "--max-attempts", "4",
            runs_root=str(tmp_path / "runs"),
        )
    )
    rc = cli.cmd_campaign(args)

    assert rc == 0
    assert captured["project_id"] == "prj_fake"
    opts = captured["opts"]
    assert opts.paper_ref == "2605.15155"
    assert opts.arxiv_id == "2605.15155"
    assert opts.runs_root == Path(tmp_path / "runs")
    assert (opts.max_llm_usd, opts.max_gpu_usd, opts.max_gpu_hours) == (12.0, 34.0, 5.0)
    assert opts.max_attempts == 4
    assert opts.wall_clock_s == 3600.0
    assert opts.mode == "checkpoint"
    assert opts.driver == "live"
    assert (opts.width, opts.plateau_k) == (2, 3)
    assert (opts.sandbox, opts.billing_sandbox) == ("local", "gcp")
    assert opts.gpu_usd_per_hr == 2.5
    assert opts.est_gpu_hours == 1.5
    assert opts.run_spec_path == str(spec_path)
    assert opts.scope_spec is None
    assert opts.scope_ladder == ("full",)  # no scope-spec, no ladder --> single full rung
    assert opts.paper_class == "vision"
    assert opts.require_cpu_tier is False
    assert opts.resume is False
    # Full ingest chain ran, in order, on the fakes.
    assert len(intake.registered) == 1 and len(intake.fetched) == 1
    for step in (parser_svc, discovery, indexer, workspace):
        assert len(step.calls) == 1


def test_cmd_campaign_scope_ladder_resolution(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    captured = _install_fake_campaign(
        monkeypatch, {"kind": "EXHAUSTED", "rule": "r", "stop_reason": None,
                      "champion_attempt_n": None, "spent": {}}
    )

    args = cli._build_parser().parse_args(
        _campaign_argv("--scope-spec", '{"models":["a"]}', runs_root=str(tmp_path / "runs"))
    )
    assert cli.cmd_campaign(args) == 0
    assert captured["opts"].scope_ladder == ('{"models":["a"]}',)  # scope-spec as single rung

    args2 = cli._build_parser().parse_args(
        _campaign_argv("--scope-ladder", "rung0, rung1 ,rung2", runs_root=str(tmp_path / "runs"))
    )
    assert cli.cmd_campaign(args2) == 0
    assert captured["opts"].scope_ladder == ("rung0", "rung1", "rung2")


def test_cmd_campaign_paused_exits_2(tmp_path, monkeypatch, capsys):
    _install_fakes(monkeypatch)
    _install_fake_campaign(
        monkeypatch, {"kind": "PAUSED", "pending_approval": {"reason": "unenforceable:x"}}
    )

    args = cli._build_parser().parse_args(_campaign_argv(runs_root=str(tmp_path / "runs")))
    rc = cli.cmd_campaign(args)

    assert rc == 2
    assert "PAUSED" in capsys.readouterr().err


def test_cmd_campaign_ledger_error_exits_3_with_money_halt(tmp_path, monkeypatch, capsys):
    _install_fakes(monkeypatch)
    _install_fake_campaign(monkeypatch, raise_ledger_error=True)

    args = cli._build_parser().parse_args(_campaign_argv(runs_root=str(tmp_path / "runs")))
    rc = cli.cmd_campaign(args)

    assert rc == 3
    assert "MONEY-HALT" in capsys.readouterr().err


def test_cmd_campaign_resume_skips_ingest_when_project_dir_exists(tmp_path, monkeypatch):
    intake, parser_svc, discovery, indexer, workspace = _install_fakes(monkeypatch)
    _install_fake_campaign(
        monkeypatch, {"kind": "EXHAUSTED", "rule": "r", "stop_reason": None,
                      "champion_attempt_n": None, "spent": {}}
    )
    runs_root = tmp_path / "runs"
    (runs_root / "prj_fake").mkdir(parents=True)

    args = cli._build_parser().parse_args(_campaign_argv("--resume", runs_root=str(runs_root)))
    rc = cli.cmd_campaign(args)

    assert rc == 0
    # register_project still runs (idempotent id resolution); the rest skips.
    assert len(intake.registered) == 1
    assert not intake.fetched
    for step in (parser_svc, discovery, indexer, workspace):
        assert not step.calls


def test_cmd_campaign_ingest_failure_exits_1(tmp_path, monkeypatch):
    intake, *_ = _install_fakes(monkeypatch)
    monkeypatch.setattr(intake, "fetch_paper", lambda cmd: False)
    built: list = []
    monkeypatch.setattr(cli, "build_campaign", lambda pid, opts: built.append(pid))

    args = cli._build_parser().parse_args(_campaign_argv(runs_root=str(tmp_path / "runs")))
    rc = cli.cmd_campaign(args)

    assert rc == 1
    assert not built  # no campaign is ever built on a failed ingest


def test_campaign_help_does_not_leak_into_reproduce(tmp_path):
    # The campaign subparser is additive: reproduce still parses a campaign
    # flag name as an error, not as an inherited option.
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["reproduce", "x.pdf", "--max-llm-usd", "5"])
