"""Estimate cache keyed on (paper_sha8, recipe_mode, catalog+calibration version).

Files land under `runs_root/_estimates/{sha8}_{recipe_mode}_{cat_v}_{cal_v}.json`.
Schema-version mismatch on either catalog or calibration → cache miss (invariant 8).

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §Cache
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from backend.services.pricing.catalog import CATALOG_SCHEMA_VERSION

logger = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION: int = 1


def _cache_dir(runs_root: Path) -> Path:
    return runs_root / "_estimates"


def _cache_key(sha256: str, recipe_mode: str) -> str:
    """Build the stem for the cache file.

    Encodes both schema versions so either bump invalidates cached estimates
    (invariant 8).
    """
    sha8 = sha256[:8]
    return f"{sha8}_{recipe_mode}_{CATALOG_SCHEMA_VERSION}_{CALIBRATION_SCHEMA_VERSION}"


def get_cached(
    runs_root: Path,
    sha256: str,
    recipe_mode: str,
) -> dict | None:
    """Return a cached estimate dict or None on miss / version mismatch."""
    path = _cache_dir(runs_root) / f"{_cache_key(sha256, recipe_mode)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — corrupt cache → miss
        logger.warning("pricing.cache: unreadable cache file %s — treating as miss", path)
        return None

    stored_cat = data.get("catalog_schema_version")
    stored_cal = data.get("calibration_schema_version")
    if stored_cat != CATALOG_SCHEMA_VERSION or stored_cal != CALIBRATION_SCHEMA_VERSION:
        logger.info(
            "pricing.cache: version mismatch (stored cat=%s cal=%s, current cat=%s cal=%s) "
            "— cache miss for %s",
            stored_cat,
            stored_cal,
            CATALOG_SCHEMA_VERSION,
            CALIBRATION_SCHEMA_VERSION,
            path.name,
        )
        return None
    return data


def set_cached(
    runs_root: Path,
    sha256: str,
    recipe_mode: str,
    estimate: dict,
) -> None:
    """Persist an estimate dict atomically."""
    cache_dir = _cache_dir(runs_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **estimate,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
    }
    path = cache_dir / f"{_cache_key(sha256, recipe_mode)}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
