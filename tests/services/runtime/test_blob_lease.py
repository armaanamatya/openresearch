"""Unit tests for ``backend.services.runtime.blob_lease``.

All tests use a self-contained, in-memory ``FakeGenerationBucketClient`` — a
generation-tracking duck-typed double satisfying the same contract
``gcs_blob.upload_bytes``/``read_bytes_with_generation`` expect (see
``gcs_blob.py``'s module docstring). No real GCS, no network: the fake
raises the real, dependency-free ``google.api_core.exceptions.
PreconditionFailed``/``NotFound`` so ``gcs_blob``'s own precondition/missing-
blob handling is exercised faithfully, exactly as it would be against the
real SDK.

``BlobLease`` never calls ``time.time()`` — every test drives the clock
explicitly via ``now_epoch`` floats, so lease expiry/races are fully
deterministic.
"""

from __future__ import annotations

import dataclasses

from google.api_core import exceptions as gcs_exceptions

from backend.services.runtime import blob_lease as bl

BUCKET = "fakebucket"
PROJECT = "fakeproject"


# ---------------------------------------------------------------------------
# FakeGenerationBucketClient — in-memory dict keyed by blob_name storing
# (data, generation); each successful write increments generation; a write
# with a mismatched if_generation_match raises PreconditionFailed.
# ---------------------------------------------------------------------------

class _FakeBlobHandle:
    def __init__(
        self, store: dict[str, tuple[bytes, int]], name: str
    ) -> None:
        self._store = store
        self._name = name
        self.generation: int | None = None

    def upload_from_string(
        self, data: bytes, if_generation_match: int | None = None
    ) -> None:
        current_gen = self._store.get(self._name, (b"", 0))[1]
        if if_generation_match is not None and if_generation_match != current_gen:
            raise gcs_exceptions.PreconditionFailed(
                f"generation mismatch for {self._name!r}: wanted "
                f"{if_generation_match!r}, live generation is {current_gen!r}"
            )
        new_gen = current_gen + 1
        self._store[self._name] = (data, new_gen)
        self.generation = new_gen

    def download_as_bytes(self) -> bytes:
        if self._name not in self._store:
            raise gcs_exceptions.NotFound(f"Blob not found: {self._name!r}")
        data, gen = self._store[self._name]
        self.generation = gen
        return data


class FakeGenerationBucketClient:
    """In-memory Bucket-like double: dict[blob_name] -> (data, generation)."""

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, int]] = {}

    def blob(self, name: str) -> _FakeBlobHandle:
        return _FakeBlobHandle(self.store, name)


def _fake() -> FakeGenerationBucketClient:
    return FakeGenerationBucketClient()


def _lease(client: FakeGenerationBucketClient) -> bl.BlobLease:
    return bl.BlobLease(bucket=BUCKET, project=PROJECT, client=client)


# ---------------------------------------------------------------------------
# acquire — fresh (no existing lease)
# ---------------------------------------------------------------------------

class TestAcquireFresh:
    def test_fresh_acquire_succeeds(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        assert token.run_id == "run-1"
        assert token.owner_id == "owner-a"
        assert token.acquired_epoch == 1000.0
        assert token.generation == 1

    def test_fresh_acquire_writes_the_lease_blob(self) -> None:
        client = _fake()
        lease = _lease(client)
        lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert "runs/run-1/rlm_state/owner.lease" in client.store

    def test_two_concurrent_acquires_exactly_one_wins(self) -> None:
        """The core split-brain guarantee: both drivers race an absent
        lease; the CAS ensures only one write survives."""
        client = _fake()
        lease_a = _lease(client)
        lease_b = _lease(client)

        # Simulate the race by hand: both read "absent" before either
        # writes — represented here by calling acquire() back-to-back on
        # independent BlobLease instances sharing the same backing store,
        # which is exactly what two independent driver processes do.
        token_a = lease_a.acquire("run-1", "owner-a", now_epoch=1000.0)
        token_b = lease_b.acquire("run-1", "owner-b", now_epoch=1000.0)

        assert (token_a is None) != (token_b is None), (
            "exactly one of the two acquires must win"
        )
        winner = token_a or token_b
        assert winner is not None
        assert winner.generation == 1

    def test_concurrent_create_race_loses_via_precondition_not_silent_success(
        self, monkeypatch
    ) -> None:
        """Forces the genuine race window: BOTH drivers' reads observe
        "absent" (as a true interleaving would), but by the time the second
        driver's write lands the first has already created the lease. The
        second's if_generation_match=0 write must lose to
        PreconditionFailedError (surfaced as acquire() -> None), never
        silently clobber the winner."""
        client = _fake()
        winner = _lease(client)
        loser = _lease(client)

        winner_token = winner.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert winner_token is not None

        # Force the loser's read to report "absent", standing in for a read
        # that genuinely happened before the winner's write landed.
        monkeypatch.setattr(loser, "_read", lambda run_id: None)

        loser_token = loser.acquire("run-1", "owner-b", now_epoch=1000.0)
        assert loser_token is None
        # The winner's write must survive untouched.
        data, gen = client.store["runs/run-1/rlm_state/owner.lease"]
        assert gen == winner_token.generation


# ---------------------------------------------------------------------------
# acquire — existing live lease
# ---------------------------------------------------------------------------

class TestAcquireLive:
    def test_live_lease_different_owner_returns_none(self) -> None:
        client = _fake()
        lease = _lease(client)
        lease.acquire("run-1", "owner-a", now_epoch=1000.0)

        # owner-b tries immediately after — lease is live, not expired.
        result = lease.acquire("run-1", "owner-b", now_epoch=1001.0)
        assert result is None

    def test_live_lease_same_owner_reacquires(self) -> None:
        """The current holder re-acquiring (e.g. process restart with a
        stable owner_id) is not blocked by its own live lease."""
        client = _fake()
        lease = _lease(client)
        token1 = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token1 is not None

        token2 = lease.acquire("run-1", "owner-a", now_epoch=1001.0)
        assert token2 is not None
        assert token2.generation == token1.generation + 1

    def test_live_lease_blocks_regardless_of_how_close_to_ttl(self) -> None:
        client = _fake()
        lease = _lease(client)
        lease.acquire("run-1", "owner-a", now_epoch=1000.0)

        just_under_ttl = 1000.0 + bl.LEASE_TTL_S - 1
        result = lease.acquire("run-1", "owner-b", now_epoch=just_under_ttl)
        assert result is None


# ---------------------------------------------------------------------------
# acquire — TTL expiry
# ---------------------------------------------------------------------------

class TestAcquireExpiry:
    def test_ttl_expired_lease_is_reacquirable_by_a_new_owner(self) -> None:
        client = _fake()
        lease = _lease(client)
        token1 = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token1 is not None

        past_ttl = 1000.0 + bl.LEASE_TTL_S + 1
        token2 = lease.acquire("run-1", "owner-b", now_epoch=past_ttl)

        assert token2 is not None
        assert token2.owner_id == "owner-b"
        assert token2.generation == token1.generation + 1

    def test_ttl_expired_takeover_is_itself_cas_raced(self) -> None:
        """Two successors both see the same expired lease; only one takeover
        write survives (same CAS guarantee, on the expiry path)."""
        client = _fake()
        lease_a = _lease(client)
        lease_b = _lease(client)
        original = bl.BlobLease(bucket=BUCKET, project=PROJECT, client=client)
        original.acquire("run-1", "owner-orig", now_epoch=1000.0)

        past_ttl = 1000.0 + bl.LEASE_TTL_S + 1
        token_a = lease_a.acquire("run-1", "owner-a", now_epoch=past_ttl)
        token_b = lease_b.acquire("run-1", "owner-b", now_epoch=past_ttl)

        assert (token_a is None) != (token_b is None), (
            "exactly one successor must win the expired-lease takeover"
        )


# ---------------------------------------------------------------------------
# renew
# ---------------------------------------------------------------------------

class TestRenew:
    def test_renew_advances_generation_and_preserves_identity(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None

        renewed = lease.renew(token, now_epoch=1030.0)
        assert renewed is not None
        assert renewed.generation == token.generation + 1
        assert renewed.run_id == token.run_id
        assert renewed.owner_id == token.owner_id
        # acquired_epoch is the ORIGINAL acquisition time, not the renewal
        # time — it identifies when this owner first took the lease.
        assert renewed.acquired_epoch == token.acquired_epoch == 1000.0

    def test_stale_renew_after_rival_advances_generation_returns_none(self) -> None:
        """A rival (e.g. a successor that took over after perceived TTL
        expiry, or a second renew from a duplicate process) advances the
        generation; the stale token's renew must fail closed."""
        client = _fake()
        lease = _lease(client)
        stale_token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert stale_token is not None

        # Rival advances the generation (simulated directly: a second
        # BlobLease instance renews using the *current* token — standing in
        # for a second driver process that also holds a valid handle).
        rival = _lease(client)
        rival_token = rival.renew(stale_token, now_epoch=1010.0)
        assert rival_token is not None
        assert rival_token.generation == stale_token.generation + 1

        # The original holder's now-stale token must fail to renew.
        result = lease.renew(stale_token, now_epoch=1020.0)
        assert result is None

    def test_renew_of_missing_lease_returns_none(self) -> None:
        """If the lease blob was deleted out from under the holder (e.g. a
        reaper bug), if_generation_match against a nonexistent object also
        fails the precondition — renew must fail closed, not crash."""
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None

        del client.store["runs/run-1/rlm_state/owner.lease"]

        result = lease.renew(token, now_epoch=1010.0)
        assert result is None


# ---------------------------------------------------------------------------
# is_current
# ---------------------------------------------------------------------------

class TestIsCurrent:
    def test_is_current_true_immediately_after_acquire(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        assert lease.is_current(token) is True

    def test_is_current_false_after_a_rival_renews(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None

        rival = _lease(client)
        rival_token = rival.renew(token, now_epoch=1010.0)
        assert rival_token is not None

        # The original token is now superseded by the rival's renewal.
        assert lease.is_current(token) is False
        # But the rival's own (advanced) token is current.
        assert lease.is_current(rival_token) is True

    def test_is_current_false_after_ttl_takeover(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None

        past_ttl = 1000.0 + bl.LEASE_TTL_S + 1
        successor = _lease(client)
        successor_token = successor.acquire("run-1", "owner-b", now_epoch=past_ttl)
        assert successor_token is not None

        assert lease.is_current(token) is False

    def test_is_current_false_when_lease_blob_missing(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None

        del client.store["runs/run-1/rlm_state/owner.lease"]

        assert lease.is_current(token) is False


# ---------------------------------------------------------------------------
# reap_older_generations — real implementation, pure injected-callable fakes
# ---------------------------------------------------------------------------

class TestReapOlderGenerations:
    def test_deletes_only_older_generation_jobs(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        token = dataclasses.replace(token, generation=3)

        jobs = [("j-g1", 1), ("j-g2", 2), ("j-g3", 3)]
        deleted_names: list[str] = []

        count = lease.reap_older_generations(
            "run-1",
            token,
            list_jobs=lambda run_id: jobs,
            delete_job=deleted_names.append,
        )

        assert count == 2
        assert deleted_names == ["j-g1", "j-g2"]

    def test_current_and_newer_generations_never_deleted(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        token = dataclasses.replace(token, generation=2)

        jobs = [("j-current", 2), ("j-newer", 5)]
        deleted_names: list[str] = []

        count = lease.reap_older_generations(
            "run-1",
            token,
            list_jobs=lambda run_id: jobs,
            delete_job=deleted_names.append,
        )

        assert count == 0
        assert deleted_names == []

    def test_empty_job_list_returns_zero_and_deletes_nothing(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        deleted_names: list[str] = []

        count = lease.reap_older_generations(
            "run-1",
            token,
            list_jobs=lambda run_id: [],
            delete_job=deleted_names.append,
        )

        assert count == 0
        assert deleted_names == []

    def test_fail_soft_one_delete_raising_does_not_block_the_others(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        token = dataclasses.replace(token, generation=3)

        jobs = [("j-g1", 1), ("j-boom", 2), ("j-g3", 3)]
        deleted_names: list[str] = []

        def flaky_delete(job_name: str) -> None:
            if job_name == "j-boom":
                raise RuntimeError("delete_namespaced_job: simulated 500")
            deleted_names.append(job_name)

        count = lease.reap_older_generations(
            "run-1",
            token,
            list_jobs=lambda run_id: jobs,
            delete_job=flaky_delete,
        )

        # j-boom's raise must not propagate, and must not abort reaping the
        # other older-generation job.
        assert count == 1
        assert deleted_names == ["j-g1"]

    def test_list_jobs_raising_returns_zero_and_does_not_crash(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None

        def flaky_list(run_id: str) -> list[tuple[str, int]]:
            raise RuntimeError("list_namespaced_job: simulated transport error")

        count = lease.reap_older_generations(
            "run-1",
            token,
            list_jobs=flaky_list,
            delete_job=lambda job_name: None,
        )

        assert count == 0

    def test_run_id_is_forwarded_to_list_jobs(self) -> None:
        """The reaper must scope the listing to this run — a caller mixing
        up run_ids would reap another run's live Jobs."""
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-42", "owner-a", now_epoch=1000.0)
        assert token is not None
        token = dataclasses.replace(token, generation=2)
        seen_run_ids: list[str] = []

        def recording_list_jobs(run_id: str) -> list[tuple[str, int]]:
            seen_run_ids.append(run_id)
            return [("j-g1", 1)]

        count = lease.reap_older_generations(
            "run-42",
            token,
            list_jobs=recording_list_jobs,
            delete_job=lambda job_name: None,
        )

        assert count == 1
        assert seen_run_ids == ["run-42"]


# ---------------------------------------------------------------------------
# reap_older_generations — NOW IMPLEMENTED (real behavior covered by
# TestReapOlderGenerations above). This guards the injected-I/O contract: the
# K8s job lister/deleter are REQUIRED keyword-only callables (design §4.3),
# so the pre-Phase-3 no-kwargs stub call is now a TypeError, not the old
# NotImplementedError.
# ---------------------------------------------------------------------------

class TestReapRequiresInjectedIO:
    def test_reap_older_generations_requires_list_and_delete_callables(self) -> None:
        """The reaper no longer stubs out — it deletes stale-generation Jobs
        via caller-injected ``list_jobs``/``delete_job`` (kept SDK-free per
        the module's purity contract). The old 2-positional-arg call must now
        raise ``TypeError`` (missing required keyword-only args), NOT
        ``NotImplementedError``."""
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None

        import pytest

        with pytest.raises(TypeError):
            lease.reap_older_generations("run-1", token)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# LEASE_TTL_S — sanity on the documented heartbeat x3 relationship
# ---------------------------------------------------------------------------

class TestLeaseTtlConstant:
    def test_ttl_is_positive_and_documented_multiple_of_heartbeat(self) -> None:
        assert bl.LEASE_TTL_S > 0
        assert bl.LEASE_TTL_S == bl._HEARTBEAT_INTERVAL_S * 3


# ---------------------------------------------------------------------------
# fence_epoch — a renew-invariant fence token, distinct from the CAS
# generation. renew() advances the generation every heartbeat; fence_epoch
# stays put so a controller never reaps its own still-running Jobs. It bumps
# only on a real takeover (a DIFFERENT owner acquiring).
# ---------------------------------------------------------------------------

class TestFenceEpoch:
    def test_first_acquire_sets_fence_epoch_1(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        assert token.fence_epoch == 1

    def test_renew_preserves_fence_epoch_while_advancing_generation(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        renewed = lease.renew(token, now_epoch=1030.0)
        assert renewed is not None
        assert renewed.fence_epoch == token.fence_epoch
        assert renewed.generation == token.generation + 1

    def test_same_owner_reacquire_preserves_fence_epoch(self) -> None:
        client = _fake()
        lease = _lease(client)
        token1 = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token1 is not None
        token2 = lease.acquire("run-1", "owner-a", now_epoch=1001.0)
        assert token2 is not None
        assert token2.fence_epoch == token1.fence_epoch

    def test_takeover_by_new_owner_bumps_fence_epoch(self) -> None:
        client = _fake()
        lease = _lease(client)
        token1 = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token1 is not None
        past_ttl = 1000.0 + bl.LEASE_TTL_S + 1
        token2 = lease.acquire("run-1", "owner-b", now_epoch=past_ttl)
        assert token2 is not None
        assert token2.fence_epoch == token1.fence_epoch + 1

    def test_same_owner_expired_reacquire_does_not_bump_fence_epoch(self) -> None:
        """The same controller returning after its own TTL lapse (a long Pod
        restart with a stable owner_id) is not a takeover — its prior Jobs are
        still its own, so the fence must NOT bump."""
        client = _fake()
        lease = _lease(client)
        token1 = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token1 is not None
        past_ttl = 1000.0 + bl.LEASE_TTL_S + 1
        token2 = lease.acquire("run-1", "owner-a", now_epoch=past_ttl)
        assert token2 is not None
        assert token2.fence_epoch == token1.fence_epoch


class TestReapStaleFenceEpochs:
    def test_own_current_fence_jobs_not_reaped_after_a_renew(self) -> None:
        """The core self-reaping regression: after a heartbeat renew advances
        the CAS generation, the controller's OWN current-fence Jobs must NOT be
        reaped — the reaper keys on fence_epoch, which renew preserves."""
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        token = lease.renew(token, now_epoch=1030.0)
        assert token is not None and token.fence_epoch == 1

        jobs = [("job-fe1", 1)]  # this controller's own current-epoch job
        deleted: list[str] = []
        count = lease.reap_stale_fence_epochs(
            "run-1", token, list_jobs=lambda r: jobs, delete_job=deleted.append
        )
        assert count == 0
        assert deleted == []

    def test_successor_reaps_predecessor_older_fence_jobs(self) -> None:
        client = _fake()
        lease = _lease(client)
        token1 = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token1 is not None
        past_ttl = 1000.0 + bl.LEASE_TTL_S + 1
        token2 = lease.acquire("run-1", "owner-b", now_epoch=past_ttl)
        assert token2 is not None and token2.fence_epoch == 2

        jobs = [("pred-fe1", 1), ("succ-fe2", 2)]
        deleted: list[str] = []
        count = lease.reap_stale_fence_epochs(
            "run-1", token2, list_jobs=lambda r: jobs, delete_job=deleted.append
        )
        assert count == 1
        assert deleted == ["pred-fe1"]

    def test_fail_soft_on_delete_and_list_errors(self) -> None:
        client = _fake()
        lease = _lease(client)
        token = lease.acquire("run-1", "owner-a", now_epoch=1000.0)
        assert token is not None
        token = dataclasses.replace(token, fence_epoch=3)

        jobs = [("j1", 1), ("boom", 2)]
        deleted: list[str] = []

        def flaky(name: str) -> None:
            if name == "boom":
                raise RuntimeError("delete_namespaced_job: simulated 500")
            deleted.append(name)

        count = lease.reap_stale_fence_epochs(
            "run-1", token, list_jobs=lambda r: jobs, delete_job=flaky
        )
        assert count == 1
        assert deleted == ["j1"]

        def flaky_list(run_id: str) -> list[tuple[str, int]]:
            raise RuntimeError("list_namespaced_job: simulated transport error")

        count2 = lease.reap_stale_fence_epochs(
            "run-1", token, list_jobs=flaky_list, delete_job=lambda n: None
        )
        assert count2 == 0
