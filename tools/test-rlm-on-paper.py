"""End-to-end smoke test: run the recursive RlmQueryTool against the
actual RLM paper (arXiv:2512.24601).

Usage:
    .venv/bin/python tools/test-rlm-on-paper.py /tmp/rlm-paper.pdf

Two runs:
    1. Stub LLM — verify the recursion shape (counts, depths, chunks)
       without burning API tokens.
    2. OpenAI LLM (if OPENAI_API_KEY set) — actual answers.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import fitz  # pymupdf

# Local import — running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.tools.rlm_query import RlmQueryTool


# --- minimal in-memory workspace view ----------------------------------------


class _PaperView:
    def __init__(self, value: str, citations: tuple[Citation, ...]) -> None:
        self._cited = Cited(value=value, citations=citations)

    def get(self, name: str):
        return self._cited if name == "paper_text" else None

    def variable_names(self) -> set[str]:
        return {"paper_text"}


# --- LLM clients --------------------------------------------------------------


@dataclass
class StubLlm:
    """Deterministic stub. Records every (system, user) so we can pin the
    recursion shape without an API call."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        # Selection prompts: pick the first 2 chunks (paper RLM-style).
        if "routing assistant" in system:
            return '{"selected": [0, 1]}'
        # Synthesis prompts: claim a fixed answer.
        if "synthesis assistant" in system:
            return "[stub-synthesized]"
        # Leaf prompts: short canned answer.
        return f"[leaf-{len(self.calls)}]"


class OpenAILlmClient:
    """LlmClient backed by OpenAI's Chat Completions. Used for the real-
    answer pass. Cheap model — we're testing the LOOP, not benchmarking."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self._calls += 1
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=600,
        )
        return resp.choices[0].message.content or ""


# --- main -------------------------------------------------------------------


def extract_paper_text(path: str) -> str:
    doc = fitz.open(path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(parts)


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB"]
    v = float(n)
    for unit in units:
        if v < 1024:
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} GB"


def main() -> int:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rlm-paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"missing PDF: {pdf_path}")
        return 2

    print(f"=== Paper: {pdf_path} ===")
    print(f"    file size: {fmt_bytes(os.path.getsize(pdf_path))}")
    t0 = time.time()
    text = extract_paper_text(pdf_path)
    print(f"    extracted: {len(text):,} chars in {time.time() - t0:.2f}s")
    print(f"    ≈ tokens:   {len(text) // 4:,} (4-char-per-token rule of thumb)")
    print()

    citation = Citation(
        source_id="arxiv:2512.24601",
        chunk_id="full",
        locator="arxiv:2512.24601",
        quote=text[:120].replace("\n", " "),
    )
    view = _PaperView(text, citations=(citation,))

    question = "What is the main technical contribution of this paper, and what specific algorithm does it propose?"

    # ── Run 1: stub LLM ────────────────────────────────────────────────────
    print("=== Run 1: stub LLM — verifying the recursion mechanics ===\n")
    stub = StubLlm()
    tool = RlmQueryTool(
        view_provider=lambda _wsid: view,
        llm_client=stub,
        leaf_budget=8_000,
        chunk_size=8_000,
        selection_top_k=4,
        selection_enabled=True,
        max_depth=3,
        max_llm_calls=30,
    )

    t0 = time.time()
    result = tool.call(
        workspace_id="ws_test", question=question, variable_name="paper_text"
    )
    dt = time.time() - t0

    print(f"  context_chars      : {result.value['context_chars']:,}")
    print(f"  leaf_budget        : {result.value['leaf_budget']:,}")
    print(f"  chunk_size         : {result.value['chunk_size']:,}")
    print(f"  max_depth          : {result.value['max_depth']}")
    print(f"  depth_reached      : {result.value['depth_reached']}")
    print(f"  llm_calls          : {result.value['llm_calls']}")
    print(f"  chunks_examined    : {result.value['chunks_examined']}")
    print(f"  truncated_at_max   : {result.value['truncated_at_max_depth']}")
    print(f"  wall time          : {dt:.2f}s")
    print(f"  citations          : {len(result.citations)} ({result.citations[0].source_id})")
    print()
    print("  Selection path (chunks the routing LLM picked at each level):")
    for entry in result.value["selection_path"]:
        print(
            f"    depth={entry['depth']}  total_chunks={entry['total_chunks']}  "
            f"selected={entry['selected']}"
        )
    print()
    print(f"  (stub answer — placeholder text — was: {result.value['answer'][:80]!r})")
    print()

    # ── Run 2: OpenAI LLM ──────────────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print("=== Run 2 skipped: OPENAI_API_KEY not set ===")
        return 0

    print("=== Run 2: OpenAI gpt-4o-mini — real answers ===\n")
    real = OpenAILlmClient(model="gpt-4o-mini")
    tool_real = RlmQueryTool(
        view_provider=lambda _wsid: view,
        llm_client=real,
        leaf_budget=8_000,
        chunk_size=8_000,
        selection_top_k=4,
        selection_enabled=True,
        max_depth=3,
        max_llm_calls=30,
    )

    t0 = time.time()
    result_real = tool_real.call(
        workspace_id="ws_test", question=question, variable_name="paper_text"
    )
    dt = time.time() - t0

    print(f"  llm_calls       : {result_real.value['llm_calls']}")
    print(f"  depth_reached   : {result_real.value['depth_reached']}")
    print(f"  chunks_examined : {result_real.value['chunks_examined']}")
    print(f"  wall time       : {dt:.2f}s")
    print()
    print("  Selection path:")
    for entry in result_real.value["selection_path"]:
        print(
            f"    depth={entry['depth']}  total_chunks={entry['total_chunks']}  "
            f"selected={entry['selected']}"
        )
    print()
    print("  ── ANSWER ──")
    print(result_real.value["answer"])
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
