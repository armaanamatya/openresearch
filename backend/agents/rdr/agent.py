"""Phase-3 Reproduction Agent — runs one scoped Claude coding agent per WorkCluster.

Public API::

    from backend.agents.rdr.agent import reproduce
    artifacts = await reproduce(agent_context, ctx=run_ctx)

See ``docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`` §7.

**v1 tool surface — design §4.4 deferred to v2.**  v1 reuses the
``baseline-implementation`` SDK agent (tools: Read / Write / Edit / Bash).
The design's "9 primitives as SDK tools" plus a custom ``paper_search`` tool
are deferred to v2.  The escape hatch for v1 is the agent's Bash access to
``../paper_full.md`` (relative to the code working directory), which covers
the paper-lookup use-case without a dedicated tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from backend.agents.rdr.models import AgentContext, Artifacts

if TYPE_CHECKING:
    from backend.agents.rlm.context import RunContext

logger = logging.getLogger(__name__)

# Maximum byte size for a single file snapshot in Artifacts.files.
_MAX_FILE_BYTES = 100 * 1024  # 100 KB

# Hard cap for the agent when no deadline is set (90 minutes).
_DEFAULT_AGENT_TIMEOUT_S = 5_400.0


def _render_prompt(agent_context: AgentContext, code_dir: Path) -> str:
    """Build the full reproduction prompt from an AgentContext.

    The prompt is the agent's entire task specification:
      - the gradable leaf contract (what the scorer will judge)
      - cited paper excerpts
      - dependency artifacts already on disk
      - optional repair feedback
      - working summary of prior clusters
    """
    ac = agent_context
    cluster = ac.cluster
    parts: list[str] = []

    parts.append(
        f"# Reproduction Task: {cluster.title}\n\n"
        "You are a scientific reproduction coding agent. Your job is to write "
        "**real, runnable reproduction code** for the following research cluster.\n\n"
        "You will be judged by an automated reproducibility scorer against the exact "
        "requirements below — satisfy every one of them.\n"
    )

    # Leaf contract
    parts.append("## Gradable Requirements (leaf contract)\n")
    parts.append(ac.leaf_contract)
    parts.append("")

    # Paper sections
    if ac.paper_sections:
        parts.append("## Relevant Paper Excerpts\n")
        for cs in ac.paper_sections:
            heading = cs.heading or cs.citation
            parts.append(f"### {heading}\n")
            parts.append(cs.text)
            parts.append("")
        parts.append(
            "_If the excerpts above are insufficient, the full paper is available at "
            "`../paper_full.md` — read or grep it as needed._\n"
        )

    # Dependency artifacts
    if ac.dependency_artifacts:
        parts.append("## Existing Files from Prior Clusters\n")
        parts.append(
            "The following files already exist in the working directory. "
            "Build on them — **do not rewrite or delete them**.\n"
        )
        for path, content in ac.dependency_artifacts.items():
            parts.append(f"**`{path}`**")
            parts.append("```")
            parts.append(content)
            parts.append("```")
            parts.append("")

    # Working summary
    if ac.working_summary:
        parts.append("## Project Structure So Far\n")
        parts.append(ac.working_summary)
        parts.append("")

    # Prior feedback (repair pass)
    if ac.prior_feedback is not None:
        parts.append("## Repair Instructions\n")
        parts.append(
            "This is a **repair pass**. The following weaknesses were identified "
            "by the leaf scorer in the previous attempt. Fix them:\n"
        )
        parts.append(ac.prior_feedback)
        parts.append("")

    # Output contract
    parts.append("## Output Contract\n")
    parts.append(
        f"Write all files into the working directory (`{code_dir}`).\n\n"
        "You MUST:\n"
        "1. Write or update `commands.json` — a JSON array of shell command strings "
        "   that, when run in order, fully execute this cluster's experiment.\n"
        "2. Ensure that running those commands produces `metrics.json` in the code "
        "   root — a flat JSON object mapping metric name to numeric value.\n"
        "3. Satisfy every leaf requirement listed in the Gradable Requirements section.\n"
    )

    return "\n".join(parts)


def _snapshot_code_dir(code_dir: Path) -> dict[str, str]:
    """Snapshot all text files under *code_dir*, skipping binary / oversized files.

    Returns a dict of ``{repo_relative_path: content}`` for all readable text
    files up to ``_MAX_FILE_BYTES`` bytes each.
    """
    result: dict[str, str] = {}
    if not code_dir.is_dir():
        return result

    for path in sorted(code_dir.rglob("*")):
        if not path.is_file():
            continue
        # Skip files that are too large.
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_FILE_BYTES:
            logger.debug("rdr/agent: skipping large file %s (%d bytes)", path, size)
            continue
        # Try to read as UTF-8 text; skip binary files.
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            logger.debug("rdr/agent: skipping non-text file %s", path)
            continue
        rel = str(path.relative_to(code_dir))
        result[rel] = content

    return result


def _parse_commands_json(code_dir: Path) -> list[str]:
    """Parse ``commands.json`` written by the agent; return [] on any failure."""
    commands_path = code_dir / "commands.json"
    if not commands_path.exists():
        return []
    try:
        data = json.loads(commands_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(cmd) for cmd in data]
        logger.warning("rdr/agent: commands.json is not a list, ignoring")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("rdr/agent: could not parse commands.json: %s", exc)
    return []


async def reproduce(agent_context: AgentContext, *, ctx: "RunContext") -> Artifacts:
    """Run one scoped Claude coding agent to reproduce one rubric cluster's work.

    Writes real code into ``ctx.project_dir / "code"`` (shared across all
    clusters for a run), collects the resulting file snapshot and parsed
    ``commands.json``, and returns an ``Artifacts``.

    Fail-soft: any exception is caught and returned as ``Artifacts(failed=True)``.
    """
    cluster_id = agent_context.cluster.id

    try:
        return await _reproduce_inner(agent_context, ctx=ctx)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "rdr/agent: cluster %r failed — %s\n%s",
            cluster_id,
            error_msg,
            traceback.format_exc(),
        )
        return Artifacts(
            cluster_id=cluster_id,
            files={},
            commands=[],
            notes="",
            failed=True,
            error=error_msg,
        )


async def _reproduce_inner(
    agent_context: AgentContext,
    *,
    ctx: "RunContext",
) -> Artifacts:
    """Inner (non-fail-soft) implementation; callers must wrap in try/except."""
    from backend.agents.runtime.invoke import collect_agent_text

    cluster_id = agent_context.cluster.id

    # 1. Ensure the shared code directory exists.
    code_dir: Path = ctx.project_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # 2. Render the agent prompt.
    prompt = _render_prompt(agent_context, code_dir)

    # 3. Resolve provider, model, and runtime dynamically from ctx.
    #    agent_model takes precedence over model (same pattern as run_with_sdk).
    model: str | None = ctx.agent_model or ctx.model or None
    provider: str | None = ctx.provider or None
    runtime = ctx.runtime  # may be None → make_runtime(provider) inside collect_agent_text

    # 4. Deadline: bound the agent by the run's remaining wall-clock budget.
    remaining = ctx.remaining_s()
    timeout_s: float = (
        min(_DEFAULT_AGENT_TIMEOUT_S, remaining)
        if remaining is not None
        else _DEFAULT_AGENT_TIMEOUT_S
    )
    # max_turns limits runaway turns; None relies on the provider's own caps.
    # The wall-clock bound is enforced below via asyncio.wait_for(timeout_s).
    max_turns: int | None = None

    logger.info(
        "rdr/agent: running cluster %r  provider=%s model=%s timeout_s=%.0f",
        cluster_id,
        provider,
        model,
        timeout_s,
    )

    # 5. Run the SDK coding agent (same infrastructure as run_with_sdk),
    #    bounded by the run's remaining wall-clock budget.
    agent_text = await asyncio.wait_for(
        collect_agent_text(
            "baseline-implementation",
            prompt,
            project_dir=code_dir,
            model=model,
            provider=provider,
            runtime=runtime,
            max_turns=max_turns,
        ),
        timeout=timeout_s,
    )

    # 6. Snapshot files written by the agent.
    files = _snapshot_code_dir(code_dir)

    # 7. Parse commands.json.
    commands = _parse_commands_json(code_dir)

    logger.info(
        "rdr/agent: cluster %r done — %d files, %d commands",
        cluster_id,
        len(files),
        len(commands),
    )

    return Artifacts(
        cluster_id=cluster_id,
        files=files,
        commands=commands,
        notes=agent_text.strip(),
        failed=False,
        error="",
    )


__all__ = ["reproduce"]
