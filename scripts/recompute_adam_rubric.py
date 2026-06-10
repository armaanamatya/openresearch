"""Recompute Adam's rubric from on-disk metrics.json after the PR-π.2 SSH fix.

The Adam run on prj_6d41d2f09c026403 completed all 4 paper experiments and
wrote 73 measured metrics to code/outputs/<id>/metrics.json — locally rsync'd
before the SSH session dropped — but run_experiment marked the call failed
(PR-π.2 fixes this prospectively). This script grades Adam's actual work.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from backend.agents.rlm.models import resolve_root_model  # noqa: E402
from backend.agents.rlm.run import _build_llm_client  # noqa: E402
from backend.evals.paperbench.leaf_scorer import score_reproduction  # noqa: E402


def main() -> int:
    project_id = "prj_6d41d2f09c026403"
    run_dir = ROOT / "runs" / project_id

    rubric_path = run_dir / "generated_rubric.json"
    if not rubric_path.exists():
        print(f"ERROR: {rubric_path} missing", file=sys.stderr)
        return 1
    rubric = json.loads(rubric_path.read_text())

    metrics_local = run_dir / "code" / "metrics.json"
    if not metrics_local.exists():
        print(f"ERROR: {metrics_local} missing — copy from code/outputs/<id>/metrics.json first", file=sys.stderr)
        return 1
    print(f"[recompute] metrics.json present at {metrics_local} ({metrics_local.stat().st_size} bytes)")

    root_model = resolve_root_model("claude-oauth")
    llm_client, model_label = _build_llm_client("anthropic", root_model)
    print(f"[recompute] llm_client built: {type(llm_client).__name__} model={model_label}")

    print(f"[recompute] calling score_reproduction on {run_dir}")
    scored = score_reproduction(
        rubric_tree=rubric,
        run_dir=run_dir,
        llm_client=llm_client,
        rubric_source=str(rubric.get("source") or "generated"),
        degraded=False,
    )
    overall = scored.get("overall_score")
    print(f"[recompute] overall_score={overall}")
    print(f"[recompute] leaves: total={scored.get('leaf_count')} graded={scored.get('graded')} coverage={scored.get('coverage_pct')}")

    out_path = run_dir / "rubric_evaluation.recomputed.json"
    out_path.write_text(json.dumps(scored, indent=2, default=str))
    print(f"[recompute] wrote {out_path}")

    fr_path = run_dir / "final_report.json"
    if fr_path.exists():
        fr = json.loads(fr_path.read_text())
        fr.setdefault("rubric", {})
        fr["rubric"] = scored
        fr["baseline_metrics"] = json.loads(metrics_local.read_text())
        fr["verdict_pre_recompute"] = fr.get("verdict")
        if overall is not None and overall >= 0.5:
            fr["verdict"] = "complete" if overall >= 0.85 else "partial"
        fr_bak = fr_path.with_suffix(".pre_recompute.json")
        fr_bak.write_text(json.dumps(json.loads(fr_path.read_text()), indent=2))
        fr_path.write_text(json.dumps(fr, indent=2, default=str))
        print(f"[recompute] backup saved to {fr_bak}")
        print(f"[recompute] updated {fr_path}: rubric={overall} verdict={fr['verdict']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
