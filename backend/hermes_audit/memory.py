"""Self-learning memory for the Hermes audit adapter.

Persists per-provider success rates between runs so the adapter can:
  * Skip a provider that has been failing repeatedly (e.g., Nous Hermes
    runtime not installed) instead of paying its import-error cost on
    every audit.
  * Promote a provider that has been succeeding to the front of the
    chain so the next run starts with what worked last time.

Storage: a single JSON file at ``<runs_root>/.hermes_adapter_memory.json``.
Reads are best-effort (file missing → fresh memory); writes are atomic
(temp file + os.replace) so a kill mid-write can never leave a partial
JSON the next process would crash on.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


_MEMORY_FILE = ".hermes_adapter_memory.json"

# Skip a provider after this many consecutive failures. The window resets
# on the first success. Picked low (3) because each retry burns network /
# compute on top of the failure that already happened.
_MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class ProviderStats:
    """Rolling success/failure counters for a single provider."""

    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_outcome: str = "unknown"   # "ok" | "fail" | "unknown"
    last_error: str = ""

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.5  # No data → neutral; lets a new provider get a try.
        return self.successes / self.total

    @property
    def is_quarantined(self) -> bool:
        """True when we've decided to skip this provider for now."""
        return self.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES

    def record_success(self) -> None:
        self.successes += 1
        self.consecutive_failures = 0
        self.last_outcome = "ok"
        self.last_error = ""

    def record_failure(self, error: str = "") -> None:
        self.failures += 1
        self.consecutive_failures += 1
        self.last_outcome = "fail"
        # Truncate so a 500 KB stack trace can't blow up the memory file.
        self.last_error = (error or "")[:500]


@dataclass
class AdapterMemory:
    """Aggregate memory across providers, persisted between runs."""

    providers: dict[str, ProviderStats] = field(default_factory=dict)
    last_successful_provider: str = ""

    def stats_for(self, name: str) -> ProviderStats:
        if name not in self.providers:
            self.providers[name] = ProviderStats()
        return self.providers[name]

    def record_success(self, name: str) -> None:
        self.stats_for(name).record_success()
        self.last_successful_provider = name

    def record_failure(self, name: str, error: str = "") -> None:
        self.stats_for(name).record_failure(error)

    def preferred_order(self, candidates: list[str]) -> list[str]:
        """Return ``candidates`` reordered by learned preference.

        Order:
          1. ``last_successful_provider`` if it's still in candidates and
             not quarantined (recency wins — last good thing first).
          2. Remaining candidates by success rate (descending), with
             quarantined providers pushed to the end (still tried as a
             last resort because providers can recover).
          3. Stable: providers with identical scores keep their input
             order so the caller's preference still matters.
        """

        if not candidates:
            return []

        head: list[str] = []
        if self.last_successful_provider in candidates:
            stats = self.providers.get(self.last_successful_provider)
            if stats is None or not stats.is_quarantined:
                head.append(self.last_successful_provider)

        rest = [c for c in candidates if c not in head]
        rest.sort(
            key=lambda name: (
                self.stats_for(name).is_quarantined,   # False first
                -self.stats_for(name).success_rate,    # higher first
            )
        )
        return head + rest

    def to_dict(self) -> dict:
        return {
            "last_successful_provider": self.last_successful_provider,
            "providers": {
                name: asdict(stats) for name, stats in self.providers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdapterMemory":
        memory = cls()
        memory.last_successful_provider = str(data.get("last_successful_provider", ""))
        for name, stats_dict in (data.get("providers") or {}).items():
            stats = ProviderStats()
            stats.successes = int(stats_dict.get("successes", 0))
            stats.failures = int(stats_dict.get("failures", 0))
            stats.consecutive_failures = int(stats_dict.get("consecutive_failures", 0))
            stats.last_outcome = str(stats_dict.get("last_outcome", "unknown"))
            stats.last_error = str(stats_dict.get("last_error", ""))
            memory.providers[str(name)] = stats
        return memory


def memory_path(runs_root: str | Path) -> Path:
    return Path(runs_root).expanduser() / _MEMORY_FILE


def load_memory(runs_root: str | Path) -> AdapterMemory:
    path = memory_path(runs_root)
    try:
        return AdapterMemory.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return AdapterMemory()
    except (json.JSONDecodeError, ValueError, TypeError):
        # File is corrupt — start fresh rather than crash. The next save
        # will overwrite the bad file with a clean one.
        return AdapterMemory()


def save_memory(runs_root: str | Path, memory: AdapterMemory) -> None:
    path = memory_path(runs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(memory.to_dict(), indent=2, sort_keys=True)
    # Atomic write: temp file in same dir, then os.replace.
    fd, tmp_name = tempfile.mkstemp(prefix=".hermes_mem_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup of the orphan temp file.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


__all__ = [
    "AdapterMemory",
    "ProviderStats",
    "load_memory",
    "memory_path",
    "save_memory",
]
