# Root-driven adaptive environment build + reproduction repair — design

> **Doc status:** Draft · spec tier · authored 2026-07-08. Direction confirmed with the
> operator (two forks): environment mechanism = **hybrid (validated base + build-on-miss)**;
> adaptation owner = **the root model, over a deterministic floor**. Policy:
> `docs/policies/documentation.md`. Grounded by the 2026-07-07 UCPO GKE bring-up (the manual
> fixes this design automates) and the deterministic execute-synth core
> (`2026-07-07-deterministic-any-paper-execute-mode-design.md`, shipped 1a).

## 1. Problem — move the operator's intelligence into the harness

Making UCPO reproduce on GKE this session took a human-in-the-loop (the operator / Claude Code
session) doing an *adaptive* loop the autonomous harness cannot yet do itself:

1. **Build the right environment** — the base cell image (Python 3.11rc1 + no verl stack)
   couldn't run verl; the operator diagnosed it (broken interpreter, missing flash-attn, ABI),
   assembled a coherent Dockerfile, and built a validated image via Cloud Build.
2. **Fix the launch** — resolve the repo's shell-vars, pick the downscale, discover the reward
   key (`critic/rewards/mean`).
3. **Debug in a loop** — dispatch → read the structured error (`ModuleNotFoundError: flash_attn`)
   → fix → redispatch, until a real reward.

The **root model** — the harness's "researcher" — must own that loop, for **any** paper. Not
the operator, and not silently the executor sub-agent (which only writes code when directed).
The deterministic evidence layer stays the fitness signal (the red line); the root supplies the
*adaptation intelligence*, the harness supplies the *mechanism* (build, run, diagnose).

Today's gap (code): on `gcp`/`gke` the image is a fixed pre-baked `gcp_base_image` and
`build_environment` is a **no-op** — there is no mechanism for the agent to adapt the image, and
no structured environment-failure signal the root can act on.

## 2. Goal / non-goals

**Goal:** an autonomous, root-driven `build → run → diagnose → fix → retry` loop for GKE
reproduction, where (a) a validated framework image is the floor, (b) a per-run image is built
on-miss via Cloud Build, and (c) the root model reasons over **structured failure evidence** to
add deps / adjust the launch / pick the reward key and re-drive the environment+experiment
primitives. Robust for any framework; deterministic fast-path for known ones (verl).

**Non-goals (this spec):** replacing the evidence/verdict layer (unchanged, value-preserving);
multi-node; private-repo auth; making the *root's grade* a fitness signal (never). The
executor sub-agent's code-writing role is unchanged — it acts only when the root directs.

## 3. Architecture — three layers

```
 ┌─ Layer 1: DETERMINISTIC FLOOR (shipped 1a; no LLM, instant) ──────────────┐
 │  framework_detector -> execute_planner -> execute_cell_synth               │
 │  known framework (verl) => cells.json + generic train_cell.py, verbatim    │
 │  framework -> VALIDATED image (verl -> gke-cell-verl)  [E1]                 │
 └───────────────────────────────────────────────────────────────────────────┘
             │ run_experiment (GKE cell matrix)
             ▼
 ┌─ Layer 2: ENVIRONMENT RESOLUTION — hybrid (validated base + build-on-miss) ┐
 │  build_environment(EnvSpec) on gcp/gke  [E2]:                              │
 │   • resolve framework -> validated pre-baked image (the heavy-stack floor) │
 │   • light paper deps installed at cell runtime (setup, --no-deps for heavy)│
 │   • BUILD-ON-MISS: assemble Dockerfile (base template + deps + build-time  │
 │     import assertions) -> Cloud Build -> content-hash tag in Artifact Reg   │
 │     -> cache (skip build if tag exists) -> set the run's image             │
 └───────────────────────────────────────────────────────────────────────────┘
             │ failure (ModuleNotFoundError / dep conflict / reward-key miss)
             ▼
 ┌─ Layer 3: ROOT-DRIVEN ADAPTATION (the intelligence)  [E3] ────────────────┐
 │  run_experiment surfaces a STRUCTURED env_diagnosis:                       │
 │   {kind: missing_module|dep_conflict|reward_key_miss|oom, detail, evidence}│
 │  deterministic first-pass fills the obvious fix (missing 'flash_attn' ->    │
 │   add dep); the ROOT model reasons for the judgment cases and re-drives     │
 │   build_environment(add deps / new base) + run_experiment(fixed spec).      │
 │  bounded by the existing repair loop; progress keyed to evidence change.    │
 └───────────────────────────────────────────────────────────────────────────┘
```

The **deterministic floor is the fast path** (known framework → instant validated image, no
LLM). The **root loop is the fallback** (novel framework / novel error → reason + build-on-miss).
This is exactly the operator's manual process, internalized.

## 4. Contracts

```python
@dataclass(frozen=True)
class EnvSpec:
    framework: str            # "verl" | "hf_trainer" | ... | "unknown"
    base_image_key: str       # "verl" | "base"  -> validated pre-baked image (floor)
    extra_pip: tuple[str, ...]        # light paper deps to install at cell runtime
    build_layers: tuple[str, ...]     # deps that MUST be baked (compiled/heavy: flash_attn, ...)
    assertions: tuple[str, ...]       # build-time `python -c "import x"` gates
    reason: str

@dataclass(frozen=True)
class BuiltImage:
    image_ref: str            # full Artifact Registry ref (content-hash tag)
    built: bool               # False => cache hit (tag already existed)
    base_image_key: str
    content_hash: str         # hash(base + build_layers + assertions) -> the tag

@dataclass(frozen=True)
class EnvDiagnosis:
    kind: str                 # missing_module | dep_conflict | reward_key_miss | oom | unknown
    detail: str               # e.g. module name "flash_attn", conflicting pins
    suggested_fix: dict       # deterministic first-pass: {"add_pip":[...]} | {"try_reward_keys":[...]}
    evidence: dict            # log signature + citation (never fabricated)
```

- **Framework→image registry** (deterministic): `verl -> gke-cell-verl:v1`, default `base`.
  A cache/floor, not the whole answer.
- **Build-on-miss** mints `BuiltImage` from an `EnvSpec`: content-hash the (base + build_layers
  + assertions); if the tag exists in Artifact Registry, skip (cache hit); else assemble a
  Dockerfile from a per-framework base template + `build_layers` + `assertions`, `gcloud builds
  submit`, tag = hash. Build-time assertions fail the build ($0) not a GPU node — the lesson
  from the rc1/flash-attn saga.
- **EnvDiagnosis** is the structured signal Layer 3 needs. Reward extraction stays
  value-preserving; a diagnosis is advisory input to the root, never a fabricated metric.

## 5. Flags (all default-OFF; byte-identical off)

- **`OPENRESEARCH_FRAMEWORK_IMAGES`** (E1) — resolve framework→validated image for the cell
  (verl→gke-cell-verl); off ⇒ today's single `gcp_base_image`. Deterministic, no build.
- **`OPENRESEARCH_ENV_BUILD`** (E2, master) — make `build_environment` build+cache a per-run
  GKE image via Cloud Build on-miss. Off ⇒ `build_environment` stays the current no-op for gcp.
  Requires the framework→image floor; only builds when the base is insufficient.
- **`OPENRESEARCH_ENV_REPAIR`** (E3) — surface `EnvDiagnosis` + let the root drive
  build_environment→run_experiment repair on an environment failure. Off ⇒ no diagnosis
  surfaced, no env-repair drive (today's behavior).
- Config: `gcp_framework_images` (framework→image map), `gcp_cloud_build_project` /
  `gcp_artifact_registry` (build target — default to the existing GCP project/registry).

## 6. Money + safety (the operator's discipline, internalized)

- **Build-time assertions** gate every built image (import the full stack on CPU Cloud Build);
  a broken interpreter/ABI fails at $0, never on an A100 — the exact rc1/flash-attn lesson.
- **Content-hash cache** — never rebuild an identical image; the second UCPO run reuses the tag.
- **Bounded repair** — the root's env-repair reuses the existing repair-iteration cap; progress
  keyed to the evidence fingerprint CHANGING (a rebuild that doesn't change the failure signature
  is not progress). No unbounded build loops.
- **GPU only after a green image** — a cell never dispatches to an A100 until its image passed
  the build-time assertions (or is a cache hit of one that did).

## 7. Increments (each flag-gated, reviewable, additive-first)

- **E1 — framework→image floor.** Registry + wire into the cell dispatch so a detected verl
  paper auto-uses gke-cell-verl (removes the manual `OPENRESEARCH_GCP_BASE_IMAGE`). Small.
- **E2 — build-on-miss env builder.** `env_builder.py` (assemble Dockerfile + Cloud Build +
  content-hash cache) + `build_environment` gcp/gke branch. The core autonomy.
- **E3 — structured diagnosis + root repair.** `env_diagnosis.py` (parse failures) + surface to
  the root + bounded env-repair drive.
- **E4 — wire the deterministic synth (1a) into implement_baseline** as the execute floor (the
  prior "1b"): short-circuit `run_with_sdk` on a confident ExecuteSpec.
- **E5 — GPU validation.** End-to-end autonomous on UCPO **and** a 2nd verl paper (proves it's
  not paper-specific): the harness picks the image, runs, and — for a deliberately dep-broken
  case — diagnoses + rebuilds + retries to a real reward, with no operator intervention.

## 8. Reference

The manual UCPO bring-up is the reference implementation of Layer 2+3: `docker/gke-cell-verl/`
(the assembled Dockerfile + build-time assertions), the ABI introspection, the flash-attn
build-on-miss, and the dispatch→error→fix→redispatch loop. `env_builder` generalizes the
Dockerfile assembly; `env_diagnosis` generalizes the error reading; the framework→image
registry generalizes the image selection. Reward stays via `verl_metrics_adapter`
(value-preserving). Memory: `reference_gke_verl_cell_image`.
