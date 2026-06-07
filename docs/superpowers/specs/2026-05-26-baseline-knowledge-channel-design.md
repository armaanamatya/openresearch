# Baseline Knowledge Channel Design

## 1. ROOT FAILURE

The first broken link was B, `plan_reproduction` populating `contract.data_recipes`; the second broken link was C, `implement_baseline` treating the failed plan as an empty valid contract. Link A did not cause this incident, and link D was bypassed.

Link A is not a regex path at all. `find_recipes_in_text()` lowercases the whole input, builds `canonical_name + aliases`, and performs a plain substring check (`backend/agents/dataset_recipes.py:249`, `backend/agents/dataset_recipes.py:253`, `backend/agents/dataset_recipes.py:255`, `backend/agents/dataset_recipes.py:257`). The Frey recipe exists with canonical name `Frey Face`, aliases including `frey face`, `frey`, `freyfaces`, and `frey_face`, and a canonical GitHub pickle loader (`backend/agents/dataset_recipes.py:102`, `backend/agents/dataset_recipes.py:103`, `backend/agents/dataset_recipes.py:104`, `backend/agents/dataset_recipes.py:105`, `backend/agents/dataset_recipes.py:106`, `backend/agents/dataset_recipes.py:107`, `backend/agents/dataset_recipes.py:108`). That means capitalization variants match via lowercasing, `frey faces` matches because it contains the literal `frey face`, and whitespace variants still match because the broad alias `frey` is enough (`backend/agents/dataset_recipes.py:103`, `backend/agents/dataset_recipes.py:253`, `backend/agents/dataset_recipes.py:257`). This is imprecise and false-positive prone, but it would not miss the VAE paper's Frey mention.

Link B broke in two ways. First, the recipe scan is scoped to `method_spec + env_spec`, not the paper text or rubric text (`backend/agents/rlm/primitives.py:997`, `backend/agents/rlm/primitives.py:1006`, `backend/agents/rlm/primitives.py:1007`, `backend/agents/rlm/primitives.py:1009`, `backend/agents/rlm/primitives.py:1011`, `backend/agents/rlm/primitives.py:1012`). The preserved `full_plan` passed to `implement_baseline` shows a stale SDAR `paper_claim_map` with datasets `ALFWorld`, `WebShop`, and `HotpotQA/Search-QA`, not the VAE datasets (`runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:928`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:929`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:930`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:964`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:965`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:966`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:967`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:968`; same stale inputs in `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:943`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:944`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:979`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:980`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:981`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:982`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:983`). Frey appears in the preserved paper/rubric context, but that context is outside the scan input (`runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:205`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:254`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:287`).

Second, `plan_reproduction` failed before it could return any usable contract. The primitive prompts the LLM with `method_spec` and `environment_spec` and asks for all `ReproductionContract` fields (`backend/agents/rlm/primitives.py:910`, `backend/agents/rlm/primitives.py:911`, `backend/agents/rlm/primitives.py:912`, `backend/agents/rlm/primitives.py:913`, `backend/agents/rlm/primitives.py:914`). `ReproductionContract.compute_scope` only accepts `ComputeScope | None` (`backend/agents/schemas.py:387`), but the preserved plan returned a string-valued `compute_scope` validation error and became `{"success": false, "error": ..., "outcome": "repairable"}` (`runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:922`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:923`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:924`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:925`; same failure in `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:937`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:938`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:939`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:940`). Current code only sanitizes `compute_scope` when clipping is active; if clipping is not active and the LLM still emits a string, the string reaches `ReproductionContract(**data)` (`backend/agents/rlm/primitives.py:938`, `backend/agents/rlm/primitives.py:940`, `backend/agents/rlm/primitives.py:959`, `backend/agents/rlm/primitives.py:1017`).

Link C then failed open. `implement_baseline` constructs a `ReproductionContract` directly from `plan["reproduction_contract"]` without checking whether the dict is an error result (`backend/agents/rlm/primitives.py:1064`, `backend/agents/rlm/primitives.py:1065`). The schema ignores extra keys (`backend/agents/schemas.py:372`), and `data_recipes` defaults to an empty list (`backend/agents/schemas.py:406`, `backend/agents/schemas.py:408`, `backend/agents/schemas.py:409`). The preserved `full_plan` passed to `implement_baseline` contains only the failed contract shape, not recipe data (`runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:1020`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:1021`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:1022`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0002.json:1023`; `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1035`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1036`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1037`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1038`). After coercion, `implement_baseline` extracts `data_recipes` from the contract and passes `None` when empty (`backend/agents/rlm/primitives.py:1216`, `backend/agents/rlm/primitives.py:1219`, `backend/agents/rlm/primitives.py:1220`, `backend/agents/rlm/primitives.py:1221`, `backend/agents/rlm/primitives.py:1252`, `backend/agents/rlm/primitives.py:1254`).

Link D did not execute for this run because `_compute_constraint_guidance()` only adds `_data_recipes_binding_block()` when `data_recipes` is non-empty (`backend/agents/baseline_implementation.py:1629`, `backend/agents/baseline_implementation.py:1633`, `backend/agents/baseline_implementation.py:1634`). If it had executed, it would still be prompt-only: it renders a table and says to use the import and loader expressions exactly (`backend/agents/baseline_implementation.py:1479`, `backend/agents/baseline_implementation.py:1480`, `backend/agents/baseline_implementation.py:1483`, `backend/agents/baseline_implementation.py:1490`), with no emitted helper module, no import contract, and no post-emit proof.

The emitted code confirms the failure path. The main run's Frey loader hardcoded `https://cs.nyu.edu/~roweis/data/frey_rawface.mat`, wrapped it in a request, and returned `None, None` on failure (`runs/prj_03271ba130d423fe/code/train.py:136`, `runs/prj_03271ba130d423fe/code/train.py:139`, `runs/prj_03271ba130d423fe/code/train.py:147`, `runs/prj_03271ba130d423fe/code/train.py:148`, `runs/prj_03271ba130d423fe/code/train.py:151`, `runs/prj_03271ba130d423fe/code/train.py:152`, `runs/prj_03271ba130d423fe/code/train.py:157`, `runs/prj_03271ba130d423fe/code/train.py:159`). The run metrics recorded `frey_face` failing with `HTTP Error 403: Forbidden` (`runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1937`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1939`, `runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1940`), and the log shows it downloaded from the NYU URL and hit 403 (`runs/_preserved_vae_score_0.6457_prj_03271ba130d423fe/iterations/iteration_0003.json:1990`). Two neighboring runs repeated the same hardcoded URL pattern (`runs/prj_db45c0304ce455a6/code/train.py:423`, `runs/prj_db45c0304ce455a6/code/train.py:426`, `runs/prj_db45c0304ce455a6/code/train.py:431`, `runs/prj_db45c0304ce455a6/code/train.py:433`, `runs/prj_db45c0304ce455a6/code/train.py:436`; `runs/prj_3080fe2a02c20164/code/train.py:253`, `runs/prj_3080fe2a02c20164/code/train.py:258`, `runs/prj_3080fe2a02c20164/code/train.py:261`, `runs/prj_3080fe2a02c20164/code/train.py:263`).

Bug family: fail-open curated knowledge delivery. Curated facts are treated as optional prompt text derived from a fallible planning contract. When upstream validation fails, downstream code silently erases the knowledge channel and the sub-agent falls back to stale prior knowledge.

## 2. CANONICAL MECHANISM

Use a helper-module knowledge channel with a post-emit contract check. For every matched curated fact, `implement_baseline` writes generated files into the code directory before invoking the sub-agent:

- `_openresearch_curated.py`: canonical helper functions and constants, for example `load_frey_face(...)`.
- `_openresearch_curated_manifest.json`: recipe ids, aliases matched, required import/use patterns, and hashes of rendered helper bodies.

The prompt then requires `train.py` to import from `_openresearch_curated` rather than writing the loader body. After the sub-agent returns, `implement_baseline` runs a local postflight check over `train.py` and the helper manifest. If a required curated helper is missing, shadowed by a local function, or contradicted by a known-bad literal such as `cs.nyu.edu/~roweis/data/frey_rawface.mat`, the emission fails before `run_experiment`. This turns curated knowledge from advisory text into a file-level contract.

Comparison:

- Alpha, harder prompt binding, is cheapest and has low code blast radius, but it still relies on the same failure mode as today. The current D path already says `use these canonical loaders verbatim` (`backend/agents/baseline_implementation.py:1480`, `backend/agents/baseline_implementation.py:1483`); making the prose louder does not prove compliance. It has weak robustness to agent drift, generalizes poorly to optimizer and environment policy, and has poor runtime observability.
- Beta, AST patcher, is strongest against drift for known code shapes, but it has the largest implementation surface and the highest blast radius. Rewriting arbitrary emitted training code can break surrounding preprocessing, metrics, imports, or type assumptions. It also does not generalize cleanly to optimizer policy or base-image constraints without many domain-specific rewriters.
- Gamma, helper module injection, is the best base mechanism. It keeps canonical code out of the sub-agent's authorship path, is small enough to implement inside the existing `implement_baseline` emission pipeline, and has limited blast radius because the generated helper is isolated. With the post-emit contract check, it is robust to drift without mutating agent code. The same manifest pattern generalizes beyond datasets because curated facts can be rendered as functions, constants, package pins, or assertions.

Recommendation: implement gamma plus a strict post-emit verifier. Do not do a general AST patcher in the first pass. The verifier may use AST parsing to inspect imports and function definitions, but it should fail or request repair rather than rewrite `train.py`.

## 3. GENERALIZATION

Download/network robustness:

Curated knowledge is a resource policy: primary URL, mirrors, expected shape/checksum when available, timeout, retry count, and soft-failure behavior. For Frey Face, the current registry already knows the GitHub raw pickle is canonical and the NYU URL is only a fallback (`backend/agents/dataset_recipes.py:105`, `backend/agents/dataset_recipes.py:106`, `backend/agents/dataset_recipes.py:107`, `backend/agents/dataset_recipes.py:110`, `backend/agents/dataset_recipes.py:111`, `backend/agents/dataset_recipes.py:112`, `backend/agents/dataset_recipes.py:115`). Enforcement is a generated helper such as `load_frey_face()` that owns the timeout, mirror fallback, decoding, shape normalization, and declared soft failure. Postflight verifies that `train.py` imports and calls the helper and does not contain banned stale literals for that dataset.

Optimizer/hyperparameter choice:

Curated knowledge is a paper-specific training policy: optimizer family, learning rate, batch size, latent dimensions, epoch budget, and allowed smoke-test reductions. Enforcement is a generated config object or builder function in `_openresearch_curated.py`, for example `build_optimizer(model, params)` plus `CURATED_TRAINING_PLAN`. `train.py` imports that policy and may scale only through explicit helper APIs. Postflight checks for contradictory local optimizer construction when the manifest marks optimizer choice as locked.

Base-image/environment mismatch:

Curated knowledge is a runtime compatibility policy: framework version, CUDA version, Python version, system packages, and known incompatible combinations. Enforcement has two parts: `build_environment` consumes the manifest package pins when building the image, and `_openresearch_curated.py` exposes `assert_runtime_environment()` for `train.py` to call at startup. Postflight verifies the call is present. Runtime assertion emits a structured failure before training if the image/framework does not match the curated policy.

The pattern is the same in all three cases: curated fact -> generated helper/constant/assertion -> required import/use contract -> postflight and runtime proof.

## 4. FAILURE-MODE CONTRACT

The run must prove curated knowledge was respected before expensive work starts.

Post-emit proof:

- Parse `train.py` after sub-agent emission.
- Verify required imports from `_openresearch_curated`.
- Verify required helper calls are present.
- Verify no local function shadows a required helper name.
- Verify known-bad literals from the manifest are absent.
- Verify `_openresearch_curated.py` helper body hashes match `_openresearch_curated_manifest.json`.

Runtime proof:

- Each helper records a small `curated_usage.json` event when called: recipe id, helper name, helper hash, and resource actually used.
- `train.py` calls `assert_curated_usage()` before writing final metrics, or `rubric_guard` learns to check the same file.
- If a required curated helper was never called, the run emits a structured preflight or postflight violation rather than silently reporting partial success.

Operator visibility:

- Emit an SSE warning when curated knowledge is required but missing from the emitted code.
- Include violation details in `repair_context.preflight_violations` so the existing patch-mode path can repair the exact import/use gap; patch-mode already triggers from structured contract or preflight violations (`backend/agents/rlm/primitives.py:1087`, `backend/agents/rlm/primitives.py:1088`, `backend/agents/rlm/primitives.py:1091`, `backend/agents/rlm/primitives.py:1092`, `backend/agents/rlm/primitives.py:1095`, `backend/agents/rlm/primitives.py:1101`).

## 5. CONCRETE DIFF PLAN

- `backend/agents/dataset_recipes.py`
  - Add stable recipe ids and optional `banned_literals`.
  - Replace broad substring matching with normalized alias matching that collapses whitespace and separators while preserving declared aliases. Keep `frey` only if accepted as an intentional broad alias.
  - Add rendering metadata for helper module generation: required function name, expected helper hash inputs, and runtime usage event name.

- `backend/agents/rlm/primitives.py`
  - In `plan_reproduction`, sanitize `compute_scope` whenever the key is present, not only when clipping is active, so string-valued `compute_scope` cannot abort recipe population (`backend/agents/rlm/primitives.py:938`, `backend/agents/rlm/primitives.py:940`, `backend/agents/rlm/primitives.py:959`, `backend/agents/rlm/primitives.py:1017`).
  - In `implement_baseline`, do not coerce `{"success": false, "error": ...}` into an empty `ReproductionContract`; detect failed primitive envelopes before constructing the schema (`backend/agents/rlm/primitives.py:1064`, `backend/agents/rlm/primitives.py:1065`).
  - Add fallback recipe recovery in `implement_baseline` from `paper_claim_map` and `environment_spec` when the reproduction contract is absent or failed, so curated knowledge is not solely coupled to a valid plan.

- `backend/agents/baseline_implementation.py`
  - Add a pre-agent step in `run_with_sdk()` after `code_dir` creation and context assembly (`backend/agents/baseline_implementation.py:1718`, `backend/agents/baseline_implementation.py:1719`, `backend/agents/baseline_implementation.py:1720`, `backend/agents/baseline_implementation.py:1723`) to write `_openresearch_curated.py` and `_openresearch_curated_manifest.json`.
  - Replace `_data_recipes_binding_block()` table-only language with an import contract that names required helper imports. Keep the current table only as explanatory context (`backend/agents/baseline_implementation.py:1418`, `backend/agents/baseline_implementation.py:1479`, `backend/agents/baseline_implementation.py:1483`).
  - Add a post-agent step immediately after `collect_agent_text()` returns (`backend/agents/baseline_implementation.py:1798`, `backend/agents/baseline_implementation.py:1805`) to verify import/use/hash/banned-literal constraints before returning a `BaselineResult`.

- New file: `backend/agents/baseline_knowledge.py`
  - Own helper rendering, manifest creation, AST/postflight checks, hash computation, and violation objects.
  - Keep it sibling to `baseline_implementation.py` so the fix stays inside the implement-baseline emission pipeline.

- Tests
  - Add dataset recipe tests for Frey capitalization, plural, whitespace, and false-positive behavior.
  - Add plan tests proving string-valued `compute_scope` is dropped and `data_recipes` still survives.
  - Add emission tests proving `_openresearch_curated.py` is written, `train.py` must import the required helper, stale URLs are rejected, and helper hash mismatch fails.
  - Add regression fixtures for the three observed train.py Frey loaders to prove the postflight checker flags all of them.

## 6. RISKS AND REJECTED OPTIONS

Rejected: prompt-only hardening. The existing binding already uses strong prompt language, but the observed failure bypassed the block entirely because `data_recipes` became empty (`backend/agents/baseline_implementation.py:1629`, `backend/agents/baseline_implementation.py:1633`, `backend/agents/baseline_implementation.py:1634`). Louder prose does not address the fail-open contract path.

Rejected for first pass: automatic AST patching. It is attractive for replacing a known loader body, but it creates a broad code mutation surface. If the patcher misidentifies a helper or rewrites a partially correct local loader, it can introduce harder-to-debug training failures. Use AST for verification first; add targeted patching later only if repeated repair loops remain expensive.

Rejected: moving the fix into root-model behavior or adding a new primitive. The existing primitive surface already passes reproduction context into `implement_baseline`, and the failure is inside the emission pipeline. Adding root behavior would widen the blast radius without making sub-agent code more compliant.

Risks in the recommended option:

- A generated helper can be wrong. Mitigation: hash-pin the helper body, unit-test rendered helpers, and keep rollback to prompt-only binding via a feature flag.
- The sub-agent may ignore the import contract. Mitigation: postflight fails before execution and feeds exact violations into repair context.
- The manifest may over-constrain legitimate paper-specific variations. Mitigation: each curated fact should declare whether it is strict, preferred, or advisory; only strict facts block execution.
- Helper-module APIs can grow into a large abstraction. Mitigation: start with dataset loaders, resource policy, optimizer policy, and runtime assertion only; no general framework until another family proves it needs one.

Rollback story: disable the postflight enforcement flag and leave helper generation plus prompt guidance in place. That returns behavior to today's prompt-level binding while preserving artifacts for debugging. If helper generation itself is suspected, disable the entire knowledge channel and fall back to current `_data_recipes_binding_block()` behavior.
