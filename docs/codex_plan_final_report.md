# Plan - Final Report, Scorecard, and Research Summary

## What this is

This is the adversarially reviewed version of the final-report plan.
It keeps the goal from the first draft, but removes the parts that would
create duplicate scoring logic, break existing schemas, or require data the
repo does not currently persist.

This document does not replace `docs/reprolab-agent-prd.md`.
The PRD remains the product north star: recursive context exploration,
progressive workspace enrichment, verifier-led gates, research maps, and
eventual graph/blackboard evolution are still valid product direction.
This plan is narrower: it defines how to add the final-report artifact to the
codebase that exists today without forcing a broad schema migration or
pretending Phase 2/Phase 3 systems are already present.

The target remains the same:

- produce a deterministic `final_report.json` per run
- surface a readable report in `/lab`
- keep reports available for both successful and failed runs

The implementation must fit the repo that exists today:

- `backend/evals/*` already defines reproduction and innovation scoring
- `backend.agents.schemas.ResearchMap` already exists and is used widely
- `PipelineState` already persists the core run outputs
- the lab UI already hydrates from `loadDemoRun()` and `pipeline_state.json`

## Critical findings from review

### 1. The original draft created a second scoring system

This repo already has:

- [backend/evals/reproduction.py](/home/abheekp/openresearch/backend/evals/reproduction.py)
- [backend/evals/innovation.py](/home/abheekp/openresearch/backend/evals/innovation.py)
- [backend/evals/schemas.py](/home/abheekp/openresearch/backend/evals/schemas.py)
- [backend/evals/runner.py](/home/abheekp/openresearch/backend/evals/runner.py)

Adding a new `backend/agents/final_report/score.py` with new component
weights would create drift. The final report should reuse the existing eval
models and only add presentation-level summarization.

### 2. The original `ResearchMap` proposal was not backward compatible

The repo already uses `backend.agents.schemas.ResearchMap` in:

- [backend/agents/schemas.py](/home/abheekp/openresearch/backend/agents/schemas.py)
- [backend/agents/orchestrator.py](/home/abheekp/openresearch/backend/agents/orchestrator.py)
- [backend/evals/innovation.py](/home/abheekp/openresearch/backend/evals/innovation.py)
- [frontend/src/lib/demo/pipeline-dashboard.ts](/home/abheekp/openresearch/frontend/src/lib/demo/pipeline-dashboard.ts)

Replacing it with a richer `ResearchPath` tree would force a broad migration.
That is not elegant. The report should keep the current `ResearchMap` as its
input contract and derive richer path summaries only inside the final report.

### 3. Mandatory `Citation` objects are not supportable yet for all report sections

The repo does have a strong citation primitive in
[backend/schemas/citations.py](/home/abheekp/openresearch/backend/schemas/citations.py),
but current run artifacts mostly expose path-based evidence:

- `VerifierScore.evidence_refs`
- `commands.log`
- `provenance.json`
- `run.log`
- `agent_telemetry.jsonl`

The first version of the final report should use a report-local `EvidenceRef`
model over concrete artifact locations. Forcing `Citation` everywhere now
would either fail composition or produce fake source pointers.

### 4. The original score inputs referenced fields that do not exist

The first draft depended on values like:

- `verification.code_quality_score`
- `verification.metric_proximity`
- `provenance_manifest.json`

Those are not current repo contracts. The report must only read data that
actually exists today.

### 5. PaperBench should not be folded into the core reproducibility grade

PaperBench is useful, but it is an external judge track, not the same thing as
the pipeline's baseline reproduction outcome. The report should include
PaperBench as an appendix or secondary signal when available, not blend it
into the primary reproduction score by default.

### 6. This is a reporting layer, not an agent

The proposed package path `backend/agents/final_report/` is the wrong home.
This code is deterministic post-processing, not LLM orchestration. It should
live under a reporting or services namespace.

## Revised architecture

### Backend

Use a small deterministic reporting package:

```text
backend/reporting/final_report/
├── __init__.py
├── compose.py       # build FinalReport from PipelineState + disk artifacts
├── loaders.py       # telemetry, paperbench status, provenance readers
├── persistence.py   # atomic temp-file write + os.replace
└── schemas.py       # FinalReport and helper models
```

No `score.py`.
No `research_map.py`.
No `weights.yaml` in v1.

### Frontend

Do not add a new dedicated final-report API route for the lab MVP.
Reuse the existing demo run loader path:

```text
frontend/src/lib/demo/node-runner.ts
frontend/src/lib/demo/demo-run-types.ts
frontend/src/components/lab/final-report.tsx
frontend/src/components/lab/live-demo-client.tsx
```

`loadDemoRun()` should read `runs/<project>/final_report.json` when present and
attach it to the existing payload. The lab already polls this payload, so the
report comes along naturally.

If a public standalone report endpoint is needed later, build it on top of the
same loader after the report format settles.

## Data model

### Reuse existing score models

The report should embed existing eval outputs directly:

```python
from backend.evals.schemas import ReproductionScore, InnovationScore
from backend.agents.schemas import ResearchMap, GateDecision
```

The report should not invent parallel equivalents.

### New report-local models

```python
class EvidenceRef(BaseModel):
    kind: Literal["artifact_path", "state_field", "telemetry", "paperbench_status"]
    locator: str
    summary: str = ""

NonEmptyEvidenceRefs = Annotated[tuple[EvidenceRef, ...], Field(min_length=1)]

class ReportPathSummary(BaseModel):
    path_id: str
    hypothesis: str
    status: Literal["promising", "verified", "rejected", "inconclusive"]
    success: bool
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: NonEmptyEvidenceRefs

class ReportScorecard(BaseModel):
    reproduction_score: float | None = None
    reproduction_grade: Literal["A", "B", "C", "D", "F"] | None = None
    innovation_score: float | None = None
    innovation_grade: Literal["A", "B", "C", "D", "F"] | None = None
    paperbench_mean: float | None = None
    paperbench_standard_error: float | None = None
    notes: list[str] = Field(default_factory=list)

class AssumptionLedgerSummary(BaseModel):
    total: int = 0
    high_risk: int = 0
    critical_risk: int = 0
    unresolved_ids: list[str] = Field(default_factory=list)

class TelemetrySummary(BaseModel):
    agent_invocations: int = 0
    failed_invocations: int = 0
    total_duration_seconds: float = 0.0
    providers: list[str] = Field(default_factory=list)

class PaperBenchSummary(BaseModel):
    run_group_id: str
    mean_score: float | None = None
    standard_error: float | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: NonEmptyEvidenceRefs

class FinalReport(BaseModel):
    project_id: str
    generated_at: str
    stage: str
    report_status: Literal["partial", "complete"]
    paper_title: str | None = None
    scorecard: ReportScorecard
    gate_1: GateDecision | None = None
    gate_2: GateDecision | None = None
    gate_3: GateDecision | None = None
    reproduction_eval: ReproductionScore | None = None
    innovation_eval: InnovationScore | None = None
    research_map: ResearchMap | None = None
    path_summaries: list[ReportPathSummary] = Field(default_factory=list)
    assumption_summary: AssumptionLedgerSummary
    telemetry: TelemetrySummary
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    paperbench: PaperBenchSummary | None = None
    missing_sections: list[str] = Field(default_factory=list)
```

## Scoring strategy

### Core rule

The final report does not define a new scoring formula.

It reuses:

- `ReproductionScore.composite_score()` as the primary reproducibility number
- `InnovationScore.research_map_score.composite_score()` as the research-summary number
- PaperBench `mean_score` as an optional external appendix when present

### Why this is the correct split

Reproducibility and research quality are different axes.
Do not collapse them into one number.

The report can still show one compact scorecard:

- `Reproduction`: baseline execution + metric match + fidelity
- `Research`: path quality + research map quality
- `PaperBench`: optional external judge result

### Grade buckets

If the UI wants letter grades, derive them from the existing composite values:

- `A >= 0.85`
- `B >= 0.70`
- `C >= 0.55`
- `D >= 0.40`
- otherwise `F`

This is a presentation rule, not a second weighting system.

## Research-map handling

Keep `backend.agents.schemas.ResearchMap` unchanged in v1.

The final report derives `ReportPathSummary.status` by combining:

- explicit mentions in `research_map.promising_directions`
- explicit mentions in `research_map.dead_ends`
- explicit mentions in `research_map.inconclusive`
- fallback heuristics from `PathResult.success` and `metrics["improvement"]`

This preserves existing agent contracts while still giving the final report a
structured path table.

## Evidence strategy

### Rule

Every rendered report section must have at least one `EvidenceRef`.

### Allowed evidence sources in v1

- `VerifierScore.evidence_refs`
- artifact file paths under `baseline/`
- `research_map.json`
- `assumption_ledger.json`
- `decision_log.json`
- `agent_telemetry.jsonl`
- `runs/paperbench/<run_group_id>/status.json`

### Explicit non-goal

Do not force all evidence into `Citation` yet.
That migration only makes sense after run artifacts are first-class sources in
the citation graph.

## Report composition behavior

### Partial reports are first-class

The composer must support partial output.

Examples:

- If the run failed after `paper_understood`, write a partial report with
  gate statuses, missing sections, and available artifacts.
- If Gate 2 failed, `reproduction_eval` may still exist, but
  `innovation_eval` and `research_map` can be absent.
- If the run completed, all sections should be filled.

### Failure policy

Report generation must never mask the real pipeline error.

Implementation rule:

- wrap composition in a nested `try/except` inside the orchestrator's finalizer
- log report-composition errors
- do not replace the original pipeline exception with a report exception

### Atomic writes

Keep the original draft's requirement:

- write `final_report.json.tmp`
- `os.replace()` into `final_report.json`

## Wiring

### Orchestrator

Refactor [backend/agents/orchestrator.py](/home/abheekp/openresearch/backend/agents/orchestrator.py)
so `run()` composes the report in a `finally` block.

Rules:

- if `state.stage >= paper_understood`, attempt report composition
- always rewrite the report with the latest state before returning
- on successful completion, the report becomes `report_status="complete"`
- on early gate failure or exception, the report remains `report_status="partial"`

### Existing outputs to reuse

The composer should read from:

- `state.paper_claim_map`
- `state.baseline_result`
- `state.experiment_artifacts`
- `state.gate_1`, `state.gate_2`, `state.gate_3`
- `state.path_results`
- `state.research_map`
- `state.assumption_ledger`
- `state.run_group_id`
- `runs/<project>/agent_telemetry.jsonl`
- `runs/<project>/baseline/provenance.json`
- `runs/paperbench/<run_group_id>/status.json` when present

### Frontend

Extend the existing lab payload instead of creating a parallel fetch path:

- `node-runner.ts` reads `final_report.json`
- `LiveDemoRunState.payload` gains `finalReport`
- `LiveDemoClient` renders `<FinalReportPanel />` when present

## Test plan

### Backend

1. `test_compose_partial_report_after_gate_2_failure`
2. `test_compose_complete_report_reuses_reproduction_eval`
3. `test_path_summary_derives_status_without_research_map_schema_change`
4. `test_missing_evidence_ref_rejects_rendered_section`
5. `test_atomic_persist_replaces_existing_report`
6. `test_paperbench_status_is_optional`

### Frontend

7. `test_load_demo_run_includes_final_report_when_present`
8. `test_final_report_panel_renders_partial_report`
9. `test_final_report_panel_renders_scorecard_and_paths`

### Fixture policy

Do not use committed `runs/...` directories as fixtures.
Use `tests/fixtures/final_report/...` or `tmp_path`-generated state instead.

## Acceptance gate

- [ ] No new parallel scoring formula is introduced
- [ ] No change to `backend.agents.schemas.ResearchMap` is required for v1
- [ ] `final_report.json` is produced for successful runs and for failed runs
      that reached at least `paper_understood`
- [ ] Report generation is deterministic and atomic
- [ ] Every rendered report section has at least one concrete evidence ref
- [ ] `/lab` renders the final report from the existing run payload
- [ ] Focused backend and frontend tests cover partial and complete reports

## Summary

The elegant version is smaller than the first draft:

- reporting layer, not agent layer
- reuse eval models, do not re-score the same run twice
- keep `ResearchMap` stable, derive richer summaries in the report only
- use artifact evidence now, full citation integration later
- reuse the existing lab payload, do not create a second data path for the MVP
