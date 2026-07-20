"""Phase 1e Unit B — global cross-paper infra memory + ``ExperienceMemory``.

Advisory-only cross-run self-improvement. ``ExperienceMemory`` WRAPS the
existing per-paper memory modules (:mod:`lesson_distiller`,
:mod:`recipe_library`) and adds exactly ONE new storage surface: a
**global, cross-paper "infra" store** (``runs/_memory/infra/<signature>.json``)
for environment / dependency / GPU / network failures whose fix generalises
across ANY paper (a
:class:`~backend.agents.rlm.failure_attribution.FailureAttribution` with
``scope == "infra"``).

Routing invariant (a test enforces this — the whole point of the split):
:meth:`ExperienceMemory.record` dispatches on ``attribution.scope``:

* ``"infra"``  — recorded into the new global store, keyed by ``signature``
  alone (cross-paper: the SAME root cause recorded against two different
  ``arxiv_id`` runs recurs onto the SAME on-disk entry).
* ``"method"`` — DELEGATED to the existing per-paper lessons storage owned
  by :mod:`lesson_distiller` (``runs/_lessons/<arxiv_id>.json``, the exact
  same file + schema ``lesson_distiller.mine_lessons`` maintains). Unit B
  introduces NO new per-paper storage surface — only the global-infra store
  above is new. A method-scoped record must NEVER create/enter the global
  infra store.

Every public method here is advisory: it returns a bool / ``list[str]`` /
``str`` that a caller may inject into an implementer/provisioning prompt as
a HINT. Nothing in this module touches sandbox, execution, or scoring code
paths, and nothing here mutates ``lesson_distiller.py`` /
``failure_attribution.py`` / ``recipe_library.py`` (all imported read-only —
this module calls their public API, plus ``lesson_distiller``'s private
storage helpers to physically share its per-paper file/schema instead of
inventing a second one).

Guardrails (mirrors ``lesson_distiller``): recurrence-gated promotion
(``occurrences>=2``), dedup, caps (<=``MAX_INFRA_HINTS`` hints /
<=``MAX_HINT_LEN`` chars each). Fail-soft throughout: any error degrades to
"no hint", never raises into the caller.

Default OFF (``OPENRESEARCH_EXPERIENCE_MEMORY`` unset/falsey): ``record`` is
a no-op, ``infra_hints() == []``, ``guidance_block() == ""`` — byte-identical
to today.

See ``docs/history/plans/2026-07-01-phase-1e-experience-memory.md``,
Unit B.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend.agents.rlm import lesson_distiller, recipe_library
from backend.agents.rlm.failure_attribution import FailureAttribution

MAX_INFRA_HINTS = 5
MAX_HINT_LEN = 200

_ENV_FLAG = "OPENRESEARCH_EXPERIENCE_MEMORY"
_TRUTHY = ("1", "on", "true", "yes")


def experience_memory_enabled() -> bool:
    """True iff ``OPENRESEARCH_EXPERIENCE_MEMORY`` opts the feature ON.

    Default OFF: any unset/empty/falsey value disables the feature entirely.
    """
    return os.environ.get(_ENV_FLAG, "").strip().lower() in _TRUTHY


def _safe_key(text: str) -> str:
    safe = "".join(c for c in str(text) if c.isalnum() or c in "._-")[:80]
    return safe or "unknown"


def _infra_store_path(runs_root: Path | str, signature: str) -> Path:
    """``runs/_memory/infra/<signature>.json`` — the ONE new storage surface
    Unit B introduces. Keyed by ``signature`` alone (no ``arxiv_id``
    component: this is the cross-paper store)."""
    return Path(runs_root) / "_memory" / "infra" / f"{_safe_key(signature)}.json"


def _read_entry(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _atomic_write(path: Path, payload: Any) -> None:
    """Atomic temp + ``os.replace`` write (mirrors ``lesson_distiller``/
    ``recipe_library``'s own store-write pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class ExperienceMemory:
    """Advisory cross-run memory: wraps the existing per-paper stores and adds
    ONE new global-infra store. See the module docstring for the scope
    routing invariant. Fail-soft — no public method here ever raises.
    """

    def __init__(self, runs_root: Path | str, *, enabled: bool | None = None) -> None:
        self.runs_root = Path(runs_root)
        self.enabled = experience_memory_enabled() if enabled is None else bool(enabled)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        attribution: FailureAttribution,
        *,
        arxiv_id: str | None = None,
        hint: str = "",
    ) -> None:
        """Route ``attribution`` by ``.scope``. No-op when disabled. Never raises.

        ``scope == "infra"`` -> the new global-infra store (cross-paper,
        keyed by ``signature``). Any other scope (``"method"`` today, and
        conservatively any future/unknown scope) -> DELEGATED to the
        existing per-paper lessons store — never the global one.
        """
        if not self.enabled:
            return
        try:
            if attribution.scope == "infra":
                self._record_infra(attribution, hint=hint)
            else:
                self._record_method(attribution, arxiv_id=arxiv_id, hint=hint)
        except Exception:
            return

    def _record_infra(self, attribution: FailureAttribution, *, hint: str) -> None:
        """Global, cross-paper: keyed by ``signature`` only."""
        path = _infra_store_path(self.runs_root, attribution.signature)
        entry = _read_entry(path)
        if not entry:
            entry = {
                "signature": attribution.signature,
                "root_cause": attribution.root_cause,
                "scope": "infra",
                "occurrences": 0,
                "hint": "",
                "staleness": 0,
            }
        entry["occurrences"] = int(entry.get("occurrences", 0)) + 1
        entry["staleness"] = 0
        entry["root_cause"] = attribution.root_cause or entry.get("root_cause", "")
        clean_hint = str(hint or "").strip()[:MAX_HINT_LEN]
        if clean_hint:
            entry["hint"] = clean_hint  # refresh to the latest hint text (never blank a good one)
        _atomic_write(path, entry)

    def _record_method(
        self,
        attribution: FailureAttribution,
        *,
        arxiv_id: str | None,
        hint: str,
    ) -> None:
        """DELEGATE to the existing per-paper store owned by
        :mod:`lesson_distiller` — same file (``runs/_lessons/<arxiv_id>.json``)
        and same entry schema ``mine_lessons`` maintains, so a record written
        here is fully interoperable with ``active_lessons``/
        ``negative_lessons_block``. Never a new storage surface.
        """
        if not arxiv_id:
            return
        path = lesson_distiller._store_path(self.runs_root, arxiv_id)
        store = lesson_distiller._read_store(path)
        fclass = attribution.root_cause or "unknown"
        entry = store.setdefault(
            fclass, {"failure_class": fclass, "occurrences": 0, "staleness": 0}
        )
        entry["occurrences"] = int(entry.get("occurrences", 0)) + 1
        entry["staleness"] = 0
        clean_hint = str(hint or "").strip()[: lesson_distiller.MAX_FIX_LEN]
        if clean_hint:
            entry["suggested_fix"] = clean_hint
        _atomic_write(path, store)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def infra_hints(self, *, env_fingerprint: str = "") -> list[str]:
        """Top-k (<=``MAX_INFRA_HINTS``) recurrence-gated (``occurrences>=2``)
        cross-paper infra hints, deduped. ``env_fingerprint`` is reserved for
        a future environment-scoped filter (Phase 1e does not yet filter by
        it). ``[]`` when disabled, no store, or on any error.
        """
        if not self.enabled:
            return []
        try:
            return self._infra_hints_inner()
        except Exception:
            return []

    def _infra_hints_inner(self) -> list[str]:
        infra_dir = self.runs_root / "_memory" / "infra"
        if not infra_dir.exists():
            return []

        entries: list[dict[str, Any]] = []
        for p in sorted(infra_dir.glob("*.json")):
            entry = _read_entry(p)
            if entry and int(entry.get("occurrences", 0)) >= 2:
                entries.append(entry)
        # Most-recurring first — the most cross-paper-validated hint leads.
        entries.sort(key=lambda e: int(e.get("occurrences", 0)), reverse=True)

        hints: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            hint_text = str(entry.get("hint") or "").strip()
            label = str(entry.get("root_cause") or "infra")
            rendered = f"[{label}] {hint_text}" if hint_text else f"[{label}]"
            rendered = rendered[:MAX_HINT_LEN]
            if rendered in seen:
                continue
            seen.add(rendered)
            hints.append(rendered)
            if len(hints) >= MAX_INFRA_HINTS:
                break
        return hints

    def guidance_block(
        self, *, arxiv_id: str | None = None, env_fingerprint: str = ""
    ) -> str:
        """Compose the existing recipe + per-paper lesson blocks with the new
        cross-paper infra hints into ONE advisory string. ``""`` when
        disabled or nothing to say. Never raises.
        """
        if not self.enabled:
            return ""
        try:
            return self._guidance_block_inner(
                arxiv_id=arxiv_id, env_fingerprint=env_fingerprint
            )
        except Exception:
            return ""

    def _guidance_block_inner(self, *, arxiv_id: str | None, env_fingerprint: str) -> str:
        sections: list[str] = []

        try:
            lesson_block = lesson_distiller.negative_lessons_block(self.runs_root, arxiv_id)
        except Exception:
            lesson_block = ""
        if lesson_block:
            sections.append(lesson_block)

        try:
            paper_class = recipe_library.derive_paper_class(arxiv_id=arxiv_id)
            recipe_block = recipe_library.recipe_guidance_block(self.runs_root, paper_class)
        except Exception:
            recipe_block = ""
        if recipe_block:
            sections.append(recipe_block)

        infra = self.infra_hints(env_fingerprint=env_fingerprint)
        if infra:
            lines = [
                "CROSS-PAPER INFRA MEMORY (environment/dependency failures seen "
                "on other papers — advisory only):"
            ]
            lines.extend(f"- {h}" for h in infra)
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


__all__ = ["ExperienceMemory", "experience_memory_enabled"]
