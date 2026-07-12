"""Unit tests for ``backend.services.runtime.job_fence`` -- pure fencing helpers.

WS3 design (durable cloud-native orchestration), Phase 1: this module is PURE
and does no I/O of its own -- these tests only assert on function *output*,
never on any cluster/GCS call.
"""

from __future__ import annotations

import re

from backend.services.runtime.job_fence import (
    adopt_or_submit,
    fenced_blob_prefix,
    fenced_job_name,
)

# RFC 1123 label: lowercase alphanumeric or '-', must start and end alphanumeric.
_DNS_1123_LABEL_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def _assert_dns_1123_label(name: str) -> None:
    assert len(name) <= 63, f"{name!r} is {len(name)} chars, exceeds the 63 cap"
    assert _DNS_1123_LABEL_RE.match(name), f"{name!r} is not a valid DNS-1123 label"


# ---------------------------------------------------------------------------
# fenced_job_name
# ---------------------------------------------------------------------------

class TestFencedJobNameDeterminism:
    def test_same_inputs_produce_identical_name(self) -> None:
        a = fenced_job_name("run-abc123", "cell-0", 1)
        b = fenced_job_name("run-abc123", "cell-0", 1)
        assert a == b

    def test_different_gen_produces_different_name(self) -> None:
        a = fenced_job_name("run-abc123", "cell-0", 1)
        b = fenced_job_name("run-abc123", "cell-0", 2)
        assert a != b

    def test_different_run_id_produces_different_name(self) -> None:
        a = fenced_job_name("run-abc123", "cell-0", 1)
        b = fenced_job_name("run-xyz789", "cell-0", 1)
        assert a != b

    def test_different_cell_id_produces_different_name(self) -> None:
        a = fenced_job_name("run-abc123", "cell-0", 1)
        b = fenced_job_name("run-abc123", "cell-1", 1)
        assert a != b


class TestFencedJobNameShape:
    def test_short_name_is_dns_1123_valid(self) -> None:
        name = fenced_job_name("run1", "cellA", 3)
        _assert_dns_1123_label(name)

    def test_short_name_carries_the_generation_suffix(self) -> None:
        name = fenced_job_name("run1", "cellA", 3)
        assert name.endswith("-g3")

    def test_short_name_starts_with_the_reprolab_cell_prefix(self) -> None:
        name = fenced_job_name("run1", "cellA", 3)
        assert name.startswith("reprolab-cell-")

    def test_uppercase_and_underscore_inputs_are_normalized(self) -> None:
        name = fenced_job_name("Run_ABC", "Cell_0", 1)
        _assert_dns_1123_label(name)
        assert name == name.lower()


class TestFencedJobNameOverlongCompression:
    _LONG_RUN = "a-very-long-run-identifier-that-goes-on-and-on-and-on-forever"
    _LONG_CELL = "an-equally-long-cell-identifier-for-this-training-matrix-run"

    def test_overlong_inputs_compress_to_the_63_char_cap(self) -> None:
        natural = f"reprolab-cell-{self._LONG_RUN}-{self._LONG_CELL}-g7"
        assert len(natural) > 63, "fixture must actually exceed the cap"

        name = fenced_job_name(self._LONG_RUN, self._LONG_CELL, 7)
        _assert_dns_1123_label(name)

    def test_overlong_compression_is_deterministic(self) -> None:
        a = fenced_job_name(self._LONG_RUN, self._LONG_CELL, 7)
        b = fenced_job_name(self._LONG_RUN, self._LONG_CELL, 7)
        assert a == b

    def test_overlong_compression_still_differs_by_generation(self) -> None:
        a = fenced_job_name(self._LONG_RUN, self._LONG_CELL, 7)
        b = fenced_job_name(self._LONG_RUN, self._LONG_CELL, 8)
        assert a != b
        _assert_dns_1123_label(a)
        _assert_dns_1123_label(b)

    def test_overlong_compression_still_differs_by_run_id(self) -> None:
        a = fenced_job_name(self._LONG_RUN, self._LONG_CELL, 7)
        b = fenced_job_name(self._LONG_RUN + "-x", self._LONG_CELL, 7)
        assert a != b
        _assert_dns_1123_label(a)
        _assert_dns_1123_label(b)


# ---------------------------------------------------------------------------
# fenced_blob_prefix
# ---------------------------------------------------------------------------

class TestFencedBlobPrefix:
    def test_run_level_prefix_without_cell_id(self) -> None:
        assert fenced_blob_prefix("run-abc", 3) == "runs/run-abc/gen-3/"

    def test_cell_level_prefix_with_cell_id(self) -> None:
        assert (
            fenced_blob_prefix("run-abc", 3, cell_id="cell-0")
            == "runs/run-abc/gen-3/cells/cell-0/"
        )

    def test_different_generations_never_share_a_prefix(self) -> None:
        gen1 = fenced_blob_prefix("run-abc", 1, cell_id="cell-0")
        gen2 = fenced_blob_prefix("run-abc", 2, cell_id="cell-0")
        assert gen1 != gen2
        assert not gen2.startswith(gen1)
        assert not gen1.startswith(gen2)

    def test_different_generations_never_share_a_run_level_prefix(self) -> None:
        gen1 = fenced_blob_prefix("run-abc", 1)
        gen2 = fenced_blob_prefix("run-abc", 2)
        assert gen1 != gen2


# ---------------------------------------------------------------------------
# adopt_or_submit
# ---------------------------------------------------------------------------

class TestAdoptOrSubmit:
    def test_already_succeeded_always_skips_even_if_phase_looks_live(self) -> None:
        assert adopt_or_submit("Running", already_succeeded=True) == "skip"

    def test_already_succeeded_skips_with_no_existing_phase(self) -> None:
        assert adopt_or_submit(None, already_succeeded=True) == "skip"

    def test_live_phases_adopt(self) -> None:
        for phase in ("Running", "Pending", "Active"):
            assert adopt_or_submit(phase, already_succeeded=False) == "adopt"

    def test_no_existing_job_submits(self) -> None:
        assert adopt_or_submit(None, already_succeeded=False) == "submit"
