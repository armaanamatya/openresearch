"""Tests for --dynamic-gpu / --force-single-gpu / --max-gpu-usd-per-hour /
--max-run-gpu-usd / --dynamic-gpu-headroom / --vram-gb CLI flags."""

from __future__ import annotations

from backend.cli import _build_parser


def test_reproduce_parser_accepts_all_dynamic_gpu_flags():
    parser = _build_parser()
    args = parser.parse_args([
        "reproduce", "paper.pdf",
        "--dynamic-gpu",
        "--no-force-single-gpu",
        "--max-gpu-usd-per-hour", "5.0",
        "--max-run-gpu-usd", "8.0",
        "--dynamic-gpu-headroom", "1.5",
        "--vram-gb", "80",
    ])
    assert args.dynamic_gpu is True
    assert args.force_single_gpu is False
    assert args.max_gpu_usd_per_hour == 5.0
    assert args.max_run_gpu_usd == 8.0
    assert args.dynamic_gpu_headroom == 1.5
    assert args.vram_gb == 80


def test_no_dynamic_gpu_flag_sets_false():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "paper.pdf", "--no-dynamic-gpu"])
    assert args.dynamic_gpu is False


def test_force_single_gpu_flag_sets_true():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "paper.pdf", "--force-single-gpu"])
    assert args.force_single_gpu is True


def test_dynamic_gpu_flags_default_to_none():
    """All six flags default to None when not supplied — no silent override of env vars."""
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "paper.pdf"])
    assert args.dynamic_gpu is None
    assert args.force_single_gpu is None
    assert args.max_gpu_usd_per_hour is None
    assert args.max_run_gpu_usd is None
    assert args.dynamic_gpu_headroom is None
    assert args.vram_gb is None


def test_max_run_gpu_usd_is_float():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "paper.pdf", "--max-run-gpu-usd", "10.0"])
    assert args.max_run_gpu_usd == 10.0
    assert isinstance(args.max_run_gpu_usd, float)


def test_vram_gb_is_int():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "paper.pdf", "--vram-gb", "48"])
    assert args.vram_gb == 48
    assert isinstance(args.vram_gb, int)
