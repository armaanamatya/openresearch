"""Local process runtime backend.

This backend executes commands on the host machine using the same artifact
directory contract as Docker. It is useful for fast iteration and environments
where Docker is unavailable, but it is not an isolation boundary.

Execution-reliability redesign (2026-06-08, local-scoped): ``exec`` STREAMS both
pipes to a live log + heartbeat sidecar (so a long run is observable while it
runs) and enforces a **stall** window (no liveness for ``REPROLAB_EXPERIMENT_STALL_S``
→ ``exec_stalled``) independently from the **hard** budget cap (the ``timeout``
param → ``exec_timeout``). Liveness = ANY of {new stdout/stderr line, checkpoint
mtime bump, GPU-util on the pinned physical ids, process-tree CPU-util} — a real
hang sits at ~0% GPU+CPU; a quiet-but-computing phase does not. Every augmentation
is fail-soft: a streaming/stall/util-poll bug degrades to "no augmentation", never
an exec failure. runpod/docker exec paths are byte-for-byte unaffected (this file is
only the LOCAL backend).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


def _venv_cuda_lib_dirs(env: dict[str, str]) -> list[str]:
    """CUDA shared-library dirs bundled INSIDE the experiment venv, for LD_LIBRARY_PATH.

    torch ships its CUDA runtime libs (``libcupti.so.*``, ``libcudart``, ``libnvrtc``,
    …) under ``site-packages/{torch/lib, nvidia/*/lib}``. An inherited
    ``LD_LIBRARY_PATH`` (e.g. ``/usr/local/cuda/lib64``) can SHADOW them and break
    ``import torch`` with "libcupti.so.12: cannot open shared object file" — the
    2026-06-07 All-Conv-Net failure. Returning these dirs (the caller PREPENDS them)
    makes the venv's own libs win.

    The batch per-run venv (``scripts/batch_reproduce.py``) is created
    ``--system-site-packages`` and ships a ``_reprolab_base_inherit.pth`` pointing at the
    repo ``.venv``'s site-packages — where the shared, coherent cu121 torch + nvidia
    wheels PHYSICALLY live. After ``env_pin`` strips the agent's torch re-pin, the per-run
    venv has no torch of its OWN, so we must follow that ``.pth`` to find the active
    torch's libs — globbing only the per-run venv would miss them entirely (2026-06-07
    Codex-review Q1). Own-venv dirs rank first (they shadow the base on ``sys.path``).

    Fail-soft: no venv / glob error → ``[]``. Only ever returns torch/nvidia lib dirs
    found inside a venv site-packages (the per-run venv or a ``.pth``-referenced base
    venv) — never a bare system CUDA path — so it cannot break an already-working torch.
    """
    venv = (env.get("REPROLAB_EXPERIMENT_VENV") or env.get("VIRTUAL_ENV") or "").strip()
    if not venv:
        return []
    try:
        # Site dirs to scan, in sys.path precedence order: the per-run venv's OWN
        # site-packages first, then any dir a ``.pth`` adds to the import path (the
        # base ``.venv`` for batch runs). Skip comment + executable (``import …``)
        # ``.pth`` lines — only filesystem path lines name a site dir.
        site_dirs: list[Path] = []
        for site in Path(venv).glob("lib/python*/site-packages"):
            site_dirs.append(site)
            for pth in sorted(site.glob("*.pth")):
                try:
                    for raw in pth.read_text(encoding="utf-8").splitlines():
                        line = raw.strip()
                        if not line or line.startswith(("#", "import ")):
                            continue
                        added = Path(line)
                        if added.is_dir():
                            site_dirs.append(added)
                except OSError:
                    continue
        dirs: list[str] = []
        for site in site_dirs:
            torch_lib = site / "torch" / "lib"
            if torch_lib.is_dir():
                dirs.append(str(torch_lib))
            nvidia = site / "nvidia"
            if nvidia.is_dir():
                for libdir in sorted(nvidia.glob("*/lib")):
                    if libdir.is_dir():
                        dirs.append(str(libdir))
        seen: set[str] = set()
        return [d for d in dirs if not (d in seen or seen.add(d))]
    except Exception:  # noqa: BLE001 — env augmentation must never break exec
        return []


# ---------------------------------------------------------------------------
# Streaming + stall instrumentation (2026-06-08; all fail-soft, local-scoped)
# ---------------------------------------------------------------------------

_LIVE_LOG_NAME = ".exec_live.log"
_HEARTBEAT_NAME = ".exec_heartbeat.json"
_STALL_POLL_S = 30.0           # how often the monitor checks liveness
_GPU_UTIL_THRESHOLD = 5.0      # >5% util on a pinned card counts as "computing"
_DEFAULT_STALL_S = 3600.0      # generous default (60 min); 0 disables
_RETAIN_HEAD = 128 * 1024      # per-stream retained-text cap: 128 KB head …
_RETAIN_TAIL = 128 * 1024      # … + 128 KB tail (live log keeps everything)
_LINE_SPLIT_RE = re.compile(rb"\r\n|\r|\n")
# progress-bearing files whose mtime bump proves the run is alive even with no stdout
_PROGRESS_GLOBS = ("*.pt", "*.ckpt", "*.safetensors", "metrics.json")


def _stall_window_s(env: dict[str, str]) -> float:
    """Resolve the stall window from the merged env (``0`` disables). Fail-soft."""
    raw = (env.get("REPROLAB_EXPERIMENT_STALL_S", "") or "").strip()
    if not raw:
        return _DEFAULT_STALL_S
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_STALL_S
    return val if val >= 0 else _DEFAULT_STALL_S


def _gpu_liveness_enabled(env: dict[str, str]) -> bool:
    return (env.get("REPROLAB_EXPERIMENT_GPU_LIVENESS", "1") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


class _CappedText:
    """Retain head + tail bytes ≤ (head_cap + tail_cap); the middle is summarised.

    The full stream still goes to the live log; this only bounds the text returned
    in the ``ExecResult`` so a chatty multi-hour run can't OOM the harness.
    """

    __slots__ = ("_head", "_tail", "_head_cap", "_tail_cap", "_total")

    def __init__(self, head_cap: int = _RETAIN_HEAD, tail_cap: int = _RETAIN_TAIL) -> None:
        self._head = bytearray()
        self._tail = bytearray()
        self._head_cap = head_cap
        self._tail_cap = tail_cap
        self._total = 0

    def add(self, chunk: bytes) -> None:
        self._total += len(chunk)
        if len(self._head) < self._head_cap:
            take = self._head_cap - len(self._head)
            self._head += chunk[:take]
            chunk = chunk[take:]
        if not chunk:
            return
        self._tail += chunk
        if len(self._tail) > self._tail_cap:
            del self._tail[: len(self._tail) - self._tail_cap]

    def text(self) -> str:
        retained = len(self._head) + len(self._tail)
        if self._total <= retained:
            return (bytes(self._head) + bytes(self._tail)).decode("utf-8", errors="replace")
        dropped = self._total - retained
        mid = f"\n...[{dropped} bytes truncated; full output in {_LIVE_LOG_NAME}]...\n".encode()
        return (bytes(self._head) + mid + bytes(self._tail)).decode("utf-8", errors="replace")


class _ExecState:
    """Mutable liveness state shared between the drain tasks and the stall monitor."""

    __slots__ = ("pid", "command", "lines", "last_line", "last_output_iso", "output_count")

    def __init__(self, pid: int | None, command: str) -> None:
        self.pid = pid
        self.command = command
        self.lines = 0
        self.last_line = ""
        self.last_output_iso = ""
        self.output_count = 0  # monotonically increasing; the monitor diffs it for liveness


def _gpu_busy_blocking(device_ids: tuple[str, ...], threshold: float = _GPU_UTIL_THRESHOLD) -> bool:
    """True if any pinned PHYSICAL GPU is > ``threshold`` % util. Fail-soft → False.

    Queries the physical device ids the run was pinned to (``sandbox.config.gpu_device_ids``),
    NOT the remapped ``cuda:0..N`` — concurrent batch runs lease disjoint cards, so a
    remapped index would read a neighbour's idle card and false-kill. ``nvidia-smi`` ignores
    ``CUDA_VISIBLE_DEVICES`` and enumerates physically, so passing the physical ids is correct.
    """
    if not device_ids:
        return False
    import subprocess  # local import: only when GPU liveness is polled

    try:
        ids = ",".join(str(d) for d in device_ids)
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                ids,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return False
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                if float(line) > threshold:
                    return True
            except ValueError:
                continue
        return False
    except Exception:  # noqa: BLE001 — util poll must never break exec
        return False


def _proc_tree_cpu_seconds(pid: int | None) -> float | None:
    """Process-tree CPU time (user+system, seconds). ``None`` if unmeasurable (fail-soft).

    Covers CPU-bound phases (dataset download / tokenize / preprocess / CPU eval) that sit
    at 0% GPU but are legitimately working. ``psutil`` if importable (clean child tree),
    else a stdlib ``/proc`` tree walk (Linux), else ``None``.
    """
    if not pid:
        return None
    try:
        import psutil  # convenience: recursive child tree in one call

        try:
            proc = psutil.Process(pid)
            procs = [proc, *proc.children(recursive=True)]
            total = 0.0
            for pr in procs:
                try:
                    ct = pr.cpu_times()
                    total += float(ct.user) + float(ct.system)
                except Exception:  # noqa: BLE001 — a vanished child is fine
                    continue
            return total
        except Exception:  # noqa: BLE001 — process gone / access → fall through to /proc
            pass
    except Exception:  # noqa: BLE001 — psutil absent → /proc fallback
        pass
    # stdlib /proc fallback: build a ppid map, sum utime+stime over pid + descendants.
    try:
        clk = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        clk = 100
    try:
        jiffies = 0
        # gather (pid -> (cpu_jiffies, ppid)) for all processes once
        info: dict[int, tuple[int, int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", encoding="utf-8") as fh:
                    data = fh.read()
                rparen = data.rfind(")")
                fields = data[rparen + 2 :].split()
                # after pid(1) + (comm)(2): fields[1]=ppid, fields[11]=utime, fields[12]=stime
                ppid = int(fields[1])
                cpu = int(fields[11]) + int(fields[12])
                info[int(entry)] = (cpu, ppid)
            except (OSError, ValueError, IndexError):
                continue
        if pid not in info:
            return None
        # descend the tree
        stack = [pid]
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            cpu_ppid = info.get(cur)
            if cpu_ppid is None:
                continue
            jiffies += cpu_ppid[0]
            for child, (_c, pp) in info.items():
                if pp == cur and child not in seen:
                    stack.append(child)
        return jiffies / float(clk or 100)
    except Exception:  # noqa: BLE001
        return None


def _newest_progress_mtime(root: Path) -> float:
    """Newest mtime of any checkpoint / metrics file under ``root`` (0.0 if none). Fail-soft."""
    newest = 0.0
    try:
        for pat in _PROGRESS_GLOBS:
            for p in root.rglob(pat):
                try:
                    mt = p.stat().st_mtime
                    if mt > newest:
                        newest = mt
                except OSError:
                    continue
    except Exception:  # noqa: BLE001
        return newest
    return newest


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the whole process group so child processes (dataloader workers,
    distributed ranks, the reparented training process) die too — not just the shell.

    ``exec`` starts the shell with ``start_new_session=True`` so ``process.pid`` is a
    process-group leader; killing the group reaps the entire tree and closes the pipes
    immediately (otherwise an orphaned child holds the stdout pipe open and the drain
    tasks block until it exits — and the orphan keeps burning the GPU). Fail-soft, with
    a plain ``process.kill()`` fallback.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        return
    except Exception:  # noqa: BLE001 — process gone / no pgid → fall back
        pass
    try:
        process.kill()
    except Exception:  # noqa: BLE001
        pass


def _write_heartbeat(path: Path, state: _ExecState) -> None:
    """Atomically write the heartbeat sidecar the SSE tailer reads. Fail-soft."""
    try:
        payload = json.dumps(
            {
                "last_output_at": state.last_output_iso,
                "last_line": state.last_line[:512],
                "lines": state.lines,
                "pid": state.pid,
                "command": state.command[:512],
            }
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — instrumentation must never break exec
        pass


async def _drain(reader: asyncio.StreamReader | None, state: _ExecState, capped: _CappedText, live_fh) -> None:
    """Drain one pipe: stream raw bytes to the live log, parse lines for liveness, cap retained text.

    Raw chunks (with ``\\r`` preserved) go to the live log so a ``tail -f`` renders tqdm bars
    in place; lines are also parsed to update the heartbeat state. Fully fail-soft.
    """
    if reader is None:
        return
    buf = b""
    while True:
        try:
            chunk = await reader.read(4096)
        except Exception:  # noqa: BLE001 — pipe error → stop draining this stream
            break
        if not chunk:
            break
        capped.add(chunk)
        if live_fh is not None:
            try:
                live_fh.write(chunk.decode("utf-8", errors="replace"))
                live_fh.flush()
            except Exception:  # noqa: BLE001
                pass
        buf += chunk
        parts = _LINE_SPLIT_RE.split(buf)
        buf = parts.pop()  # trailing partial line stays buffered
        for raw in parts:
            if not raw:
                continue
            state.lines += 1
            state.output_count += 1
            state.last_line = raw.decode("utf-8", errors="replace")
            state.last_output_iso = datetime.now(timezone.utc).isoformat()
    if buf:
        state.lines += 1
        state.output_count += 1
        state.last_line = buf.decode("utf-8", errors="replace")
        state.last_output_iso = datetime.now(timezone.utc).isoformat()


async def _monitor(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task,
    state: _ExecState,
    *,
    timeout: float | None,
    stall_window: float,
    project_root: Path,
    heartbeat_path: Path,
    device_ids: tuple[str, ...],
    gpu_liveness_on: bool,
) -> str | None:
    """Poll liveness until the process exits or a limit trips.

    Returns ``None`` (process exited on its own), ``"exec_timeout"`` (hard budget cap),
    or ``"exec_stalled"`` (no liveness for ``stall_window``). Liveness = ANY of new output,
    checkpoint mtime bump, GPU-util on the pinned cards, process-tree CPU-util advance.
    """
    loop = asyncio.get_event_loop()
    start = loop.time()
    last_liveness = start
    last_output_count = state.output_count
    last_ckpt = _newest_progress_mtime(project_root)
    last_cpu = await asyncio.to_thread(_proc_tree_cpu_seconds, state.pid)
    # Poll often enough that a small stall window / hard cap is actually observed in
    # time (the 30 s default would never catch a 2 s window). Scale to half the
    # tightest bound, floored at 1 s; for production (3600 s window) this stays 30 s.
    poll = _STALL_POLL_S
    for _bound in (stall_window, timeout):
        if _bound and _bound > 0:
            poll = min(poll, _bound / 2.0)
    poll = max(1.0, poll)
    while True:
        done, _ = await asyncio.wait({wait_task}, timeout=poll)
        if wait_task in done:
            return None
        now_t = loop.time()
        # always refresh the heartbeat sidecar so the SSE tailer sees a recent timestamp
        _write_heartbeat(heartbeat_path, state)
        if timeout and (now_t - start) >= timeout:
            return "exec_timeout"
        alive = False
        if state.output_count > last_output_count:
            alive = True
        last_output_count = state.output_count
        ckpt = _newest_progress_mtime(project_root)
        if ckpt > last_ckpt:
            alive = True
        last_ckpt = ckpt
        cpu = await asyncio.to_thread(_proc_tree_cpu_seconds, state.pid)
        if cpu is not None and last_cpu is not None and cpu > last_cpu:
            alive = True
        if cpu is not None:
            last_cpu = cpu
        if not alive and gpu_liveness_on:
            if await asyncio.to_thread(_gpu_busy_blocking, device_ids):
                alive = True
        if alive:
            last_liveness = now_t
        if stall_window and (now_t - last_liveness) >= stall_window:
            return "exec_stalled"


class LocalProcessBackend(RuntimeBackend):
    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        project_root = config.project_root.resolve()
        artifact_root = config.resolved_artifact_root().resolve()
        if not project_root.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Local project root does not exist: {project_root}",
            )
        artifact_root.mkdir(parents=True, exist_ok=True)
        return Sandbox(
            sandbox_id=f"local-{_safe_name(config.project_id)}-{_safe_name(config.run_id)}",
            name=f"local-{_safe_name(config.project_id)}-{_safe_name(config.run_id)}",
            image="local-process",
            config=config,
        )

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        started_at = datetime.now(timezone.utc)
        env = {
            **os.environ,
            **sandbox.config.environment,
        }
        if sandbox.config.gpu_device_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(sandbox.config.gpu_device_ids)
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        # Make the experiment venv's bundled CUDA libs loadable: PREPEND them to
        # LD_LIBRARY_PATH so an inherited system CUDA path can't shadow libcupti.so.*
        # (the 2026-06-07 "cannot open shared object file" import death). Fail-soft —
        # no venv / no dirs → unchanged.
        _cuda_dirs = _venv_cuda_lib_dirs(env)
        if _cuda_dirs:
            _prev_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                os.pathsep.join([*_cuda_dirs, _prev_ld]) if _prev_ld
                else os.pathsep.join(_cuda_dirs)
            )

        project_root = sandbox.config.project_root
        live_path = Path(project_root) / _LIVE_LOG_NAME
        heartbeat_path = Path(project_root) / _HEARTBEAT_NAME
        stall_window = _stall_window_s(env)
        gpu_liveness_on = _gpu_liveness_enabled(env)
        device_ids = tuple(sandbox.config.gpu_device_ids or ())

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(sandbox.config.project_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # own process group → kill the whole tree on stall/timeout
            )
        except Exception as exc:  # pragma: no cover - subprocess platform edge
            raise SandboxRuntimeError(
                RuntimeCauseKind.command_failed,
                str(exc),
                retryable=False,
            ) from exc

        state = _ExecState(pid=process.pid, command=command)
        # C2: register this exec's process group so binding's per-primitive timeout
        # can SIGKILL it if the OUTER timeout abandons this coroutine before our own
        # stall/timeout fires. Soft + lazy (avoids a services->agents import cycle) +
        # flag-gated (no-op kill unless REPROLAB_ORPHAN_GUARD; byte-for-byte today).
        try:
            from backend.agents.rlm import orphan_guard as _orphan_guard
            _orphan_guard.register(process.pid)
        except Exception:  # noqa: BLE001 — orphan registration must never break exec
            pass
        stdout_cap = _CappedText()
        stderr_cap = _CappedText()
        live_fh = None
        try:  # open the live log once (append); fail-soft if the dir is unwritable
            live_fh = open(live_path, "a", encoding="utf-8")
            live_fh.write(f"=== exec {started_at.isoformat()} :: {command} ===\n")
            live_fh.flush()
        except Exception:  # noqa: BLE001
            live_fh = None
        _write_heartbeat(heartbeat_path, state)

        stdout_task = asyncio.ensure_future(_drain(process.stdout, state, stdout_cap, live_fh))
        stderr_task = asyncio.ensure_future(_drain(process.stderr, state, stderr_cap, live_fh))
        wait_task = asyncio.ensure_future(process.wait())

        kill_reason: str | None = None
        try:
            kill_reason = await _monitor(
                process,
                wait_task,
                state,
                timeout=float(timeout) if timeout else None,
                stall_window=stall_window,
                project_root=Path(project_root),
                heartbeat_path=heartbeat_path,
                device_ids=device_ids,
                gpu_liveness_on=gpu_liveness_on,
            )
        except Exception:  # noqa: BLE001 — a monitor bug must not abort exec; just stop monitoring
            kill_reason = None

        if kill_reason is not None:
            _kill_process_group(process)

        # Drain whatever's buffered (the pipes close once the process dies), then reap.
        try:
            await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task, return_exceptions=True),
                timeout=15,
            )
        except Exception:  # noqa: BLE001
            for t in (stdout_task, stderr_task):
                if not t.done():
                    t.cancel()
        try:
            await asyncio.wait_for(wait_task, timeout=15)
        except Exception:  # noqa: BLE001
            if not wait_task.done():
                wait_task.cancel()
        if live_fh is not None:
            try:
                live_fh.flush()
                live_fh.close()
            except Exception:  # noqa: BLE001
                pass
        _write_heartbeat(heartbeat_path, state)

        # C2: exec reaped (all completion paths reach here) → drop its process group
        # from the orphan registry so a later primitive timeout can't target a
        # since-recycled PID.
        try:
            from backend.agents.rlm import orphan_guard as _orphan_guard
            _orphan_guard.deregister(process.pid)
        except Exception:  # noqa: BLE001
            pass

        finished_at = datetime.now(timezone.utc)
        duration = (finished_at - started_at).total_seconds()
        out_text = stdout_cap.text()
        err_text = stderr_cap.text()

        if kill_reason == "exec_timeout":
            return ExecResult(
                command=command,
                exit_code=None,
                stdout=out_text,
                stderr=(err_text + f"\nCommand timed out after {timeout} seconds.").strip(),
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                timed_out=True,
                cause_kind=RuntimeCauseKind.exec_timeout,
            )
        if kill_reason == "exec_stalled":
            return ExecResult(
                command=command,
                exit_code=None,
                stdout=out_text,
                stderr=(
                    err_text
                    + f"\nCommand stalled: no output / checkpoint / GPU / CPU activity for "
                    f"{stall_window:.0f} s."
                ).strip(),
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                timed_out=True,
                cause_kind=RuntimeCauseKind.exec_stalled,
            )

        exit_code = process.returncode
        return ExecResult(
            command=command,
            exit_code=exit_code,
            stdout=out_text,
            stderr=err_text,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            cause_kind=None if exit_code == 0 else RuntimeCauseKind.command_failed,
        )

    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes:
        host_path = _map_containerish_path(sandbox.config, path)
        try:
            return await asyncio.to_thread(host_path.read_bytes)
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not copy local file {host_path}: {exc}",
            ) from exc

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        host_path = _map_containerish_path(sandbox.config, path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(host_path.write_bytes, data)
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not write local file {host_path}: {exc}",
            ) from exc

    async def destroy(self, sandbox: Sandbox) -> None:
        return None


def _map_containerish_path(config: SandboxConfig, path: str) -> Path:
    posix = PurePosixPath(path)
    if not posix.is_absolute():
        return (config.project_root / path).resolve()

    workdir = PurePosixPath(config.workdir)
    artifacts_dir = PurePosixPath(config.artifacts_dir)
    if posix == workdir or workdir in posix.parents:
        return (config.project_root / posix.relative_to(workdir).as_posix()).resolve()
    if posix == artifacts_dir or artifacts_dir in posix.parents:
        return (
            config.resolved_artifact_root() / posix.relative_to(artifacts_dir).as_posix()
        ).resolve()
    raise SandboxRuntimeError(
        RuntimeCauseKind.copy_failed,
        f"Path {path!r} is outside local runtime mounts.",
    )


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:48]


__all__ = ["LocalProcessBackend"]
