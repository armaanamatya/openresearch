"""Runtime application service.

This service is the IO boundary for sandbox lifecycle operations. It can be
used with or without an EventStore; when a store is provided it records the
runtime events emitted by the pure SandboxAggregate.
"""

from __future__ import annotations

from pydantic import ConfigDict

from backend.eventstore.interface import EventStore
from backend.messaging.command import Command
from backend.messaging.envelope import (
    AggregateId,
    CorrelationId,
    EventEnvelope,
    make_envelope,
    new_correlation_id,
)
from backend.messaging.event import DomainEvent, resolve_event_class
from backend.services.runtime.aggregate import SandboxAggregate
from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


class CreateSandbox(Command):
    model_config = ConfigDict(frozen=True)
    config: SandboxConfig


class ExecuteCommand(Command):
    model_config = ConfigDict(frozen=True)
    sandbox: Sandbox
    command: str
    timeout: int


class DestroySandbox(Command):
    model_config = ConfigDict(frozen=True)
    sandbox: Sandbox


class RuntimeAppService:
    def __init__(
        self,
        backend: RuntimeBackend,
        *,
        store: EventStore | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        self._memory: dict[tuple[str, str], SandboxAggregate] = {}

    async def create_sandbox(
        self,
        cmd: CreateSandbox,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> Sandbox:
        cid = correlation_id or new_correlation_id()
        agg = self._load_aggregate(cmd.config.project_id, cmd.config.run_id)
        self._append(agg, agg.handle_request(cmd.config), cid)
        try:
            sandbox = await self._backend.create_sandbox(cmd.config)
        except SandboxRuntimeError as exc:
            self._append(agg, agg.handle_failed(exc), cid)
            raise
        self._append(agg, agg.handle_created(sandbox.sandbox_id, sandbox.image), cid)
        return sandbox

    async def execute(
        self,
        cmd: ExecuteCommand,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> ExecResult:
        cid = correlation_id or new_correlation_id()
        result = await self._backend.exec(cmd.sandbox, cmd.command, cmd.timeout)
        agg = self._load_aggregate(cmd.sandbox.config.project_id, cmd.sandbox.config.run_id)
        self._append(agg, agg.handle_command_result(result), cid)
        return result

    async def destroy(
        self,
        cmd: DestroySandbox,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        cid = correlation_id or new_correlation_id()
        try:
            await self._backend.destroy(cmd.sandbox)
        finally:
            agg = self._load_aggregate(
                cmd.sandbox.config.project_id, cmd.sandbox.config.run_id
            )
            self._append(agg, agg.handle_destroyed(), cid)

    def _load_aggregate(self, project_id: str, run_id: str) -> SandboxAggregate:
        agg = SandboxAggregate.empty(project_id, run_id)
        if self._store is None:
            return self._memory.setdefault((project_id, run_id), agg)
        for stored in self._store.load(_runtime_aggregate_id(project_id, run_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _append(
        self,
        agg: SandboxAggregate,
        events: list[DomainEvent] | tuple[DomainEvent, ...],
        correlation_id: CorrelationId,
    ) -> None:
        if not events:
            return
        if self._store is not None:
            envelopes: list[EventEnvelope] = [
                make_envelope(
                    source="runtime.service",
                    correlation_id=correlation_id,
                )
                for _ in events
            ]
            self._store.append(
                aggregate_id=_runtime_aggregate_id(agg.project_id, agg.run_id),
                aggregate_type="runtime",
                events=list(events),
                expected_version=agg.version,
                envelopes=envelopes,
            )
        agg.apply_all(events)


def _runtime_aggregate_id(project_id: str, run_id: str) -> AggregateId:
    return AggregateId(f"{project_id}:runtime:{run_id}")


__all__ = [
    "CreateSandbox",
    "DestroySandbox",
    "ExecuteCommand",
    "RuntimeAppService",
]
