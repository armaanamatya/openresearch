# Build spec — agentic SDAR environments (2026-06-01)

**Goal.** Restore the SDAR paper's full scope (ALFWorld + WebShop + Search-QA) as
**real multi-turn agentic environments**, replacing the closed-book / faked
surrogates that floored the rubric at ~0.36. Maximum-fidelity, production-ready,
fully tested. Retrieval is **dense E5 over wiki-18, cached** (download/build once,
reuse), with a BM25 fallback so a cold cache never blocks the grid.

This spec is the contract for parallel module authors. Each module is a **new
copyable helper** auto-copied into every run's `code/` dir (like
`gpu_cell_runner.py`). They import the agentic interface from `sdar_env_base`.

---

## 0. Hard rules for every module author

1. **Create ONLY your one module file + its one unit test.** Do **not** edit any
   existing file (the tech lead wires `_HARNESS_CODE_HELPERS`, guidance,
   `env_cache`, and the run pipeline separately). Touching shared files causes
   merge conflicts with the parallel agents.
2. Module files live in `backend/agents/rlm/`. Tests live in
   `tests/agents/rlm/`. Run tests with the repo venv:
   `/home/sww35/openresearch/.venv/bin/python -m pytest <your_test> -q`.
   The venv has: torch, transformers, datasets, alfworld, textworld, numpy.
   It does **not** yet have rank_bm25 / sentence-transformers / faiss / requests —
   so **guard those imports** (lazy, inside methods) and make your unit test
   **not require them** (inject fakes / monkeypatch). The per-run venv installs
   them at run time; your test must pass on the base venv today.
3. **Fail-soft, never crash the grid.** No method raises on bad input. A failed
   retrieval / malformed action / missing server returns a degraded observation
   or a zero-reward terminal step — never an exception that kills the cell.
4. **Determinism.** Given `seed` + `task`, `reset`/`step` are reproducible.
5. **Stdlib + your declared deps only.** No new repo-internal imports beyond
   `from sdar_env_base import AgenticEnv, StepResult`.
6. Match the house style of `sdar_env_base.py` / `env_cache.py`: module
   docstring explaining the *why*, dataclasses for results, type hints, focused
   comments on the non-obvious.

---

## 1. The interface you build against (already written — `sdar_env_base.py`)

```python
from sdar_env_base import AgenticEnv, StepResult

class AgenticEnv(BaseEnv):
    max_turns: int = 1            # override per env
    def reset(self, *, seed=None, task=None) -> str: ...      # abstract — you implement
    def step(self, action: str) -> StepResult: ...            # abstract — you implement
    # provided for free (override only if needed):
    #   _start_episode(system="")  _record_obs(t)  _record_act(t)  _finish(reward, info=None)
    #   render_transcript() -> str   build_student_prompt()   build_teacher_prompt()
    #   episode_reward() -> float    .done   .turns_taken   .last_info

@dataclass
class StepResult:
    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)
```

**Episode protocol the trainer runs** (so you know how reset/step are called):

```python
env.reset(seed=s, task=task)
for _ in range(env.max_turns):
    prompt = env.build_student_prompt()
    action = model_generate(prompt)          # one turn of text
    res = env.step(action)
    if res.done: break
reward = env.episode_reward()                # scalar; trainer also reads env.last_info
```

Inside `reset`: call `self._start_episode(system=<instructions>)`, set up task
state, `self._record_obs(<initial observation>)`, return that observation string.

Inside `step(action)`: `self._record_act(action)`, parse the action, advance the
env, compute the new observation, `self._record_obs(obs)`. If the episode ends
(success/failure/last turn), call `self._finish(reward, info={...})` and return
`StepResult(obs, reward=reward, done=True, info={...})`. Otherwise return
`StepResult(obs, reward=0.0, done=False)`.

---

## 2. Module: `agentic_rollout.py`  (author: AGENT-ROLLOUT)

**Why.** The multi-turn → flat-token-sequence + response-mask conversion is the
single most bug-prone part of agentic RL (off-by-one masks silently zero the
loss). Centralise it once, tested, so every env + the trainer share it.

**Deliverables** in `backend/agents/rlm/agentic_rollout.py`:

```python
@dataclass
class Turn:
    prompt_text: str
    prompt_ids: list[int]
    response_ids: list[int]      # the model's generated tokens this turn

@dataclass
class Trajectory:
    turns: list[Turn]
    sequence_ids: list[int]      # full interleaved prompt+response token ids
    response_mask: list[int]     # 1 at student-generated positions, else 0; len == len(sequence_ids)
    reward: float
    info: dict                   # env.last_info merged with rollout stats (n_turns, etc.)

def rollout_episode(env, *, generate, tokenizer, max_turns=None,
                    max_new_tokens=64) -> Trajectory:
    """Drive ONE multi-turn episode and return its flat trajectory.

    `generate(prompt_text) -> (response_text, response_token_ids)` is injected
    (the trainer wraps its HF model.generate). `tokenizer` is the HF tokenizer
    (only `.encode`/`.__call__` used). Builds sequence_ids by concatenating, per
    turn, the *delta* prompt tokens (the new transcript tail since last turn) then
    the response tokens, marking only response positions in response_mask.
    """
```

Key correctness points (cover in tests with a FAKE tokenizer = char-ord encoder
and a FAKE env + FAKE generate):
- `len(response_mask) == len(sequence_ids)`; `sum(response_mask) == total response tokens`.
- Response positions in the mask line up exactly with `response_ids` of each turn.
- Episode stops at `env.done` or `max_turns`.
- `reward == env.episode_reward()`; `info` includes `n_turns` and `env.last_info`.
- A `generate` that returns `""` (empty) still advances a turn and does not crash.

No torch/transformers needed in the module logic itself (operate on the injected
`generate`/`tokenizer`), so the unit test runs on the base venv with fakes.

---

## 3. Module: `search_qa_env.py`  (author: AGENT-SEARCH)

**Why.** The 2026-05-31 run made Search-QA *closed-book* (`Question:\nAnswer:`,
no retrieval) → 1.7B floored at ~0.05 F1 vs paper 0.38–0.46. Restore **real
retrieval**: the model issues `search(query)` actions, reads top-k passages, then
`answer(text)`. Reward = max-alias token-F1 (SQuAD norm).

**Class** `SearchQAEnv(AgenticEnv)` in `backend/agents/rlm/search_qa_env.py`:

- `max_turns = 6` (a few searches + an answer).
- `reset(seed, task)`: `task` is `{"question": str, "answers": [str], "contexts":
  [str]|None, "source": "nq"|"hotpotqa"}`. **Keep HotpotQA contexts** — when the
  task carries `contexts`, they seed a per-question candidate pool (this is the
  fix for the loader that discarded `row["context"]`). System prompt explains the
  action grammar.
- **Action grammar** (parse defensively, case-insensitive, tolerate code fences):
  - `search(<query>)` or `search: <query>` → retrieve top-k (k=3) passages,
    `_record_obs` them as `Observation: <passage texts>`. Counts toward turns.
  - `answer(<text>)` or `answer: <text>` or a bare final line → end episode,
    reward = `token_f1(text, task.answers)`, `_finish(reward, info=...)`.
  - Unparseable → observation `"Use search(<query>) or answer(<text>)."`, no
    reward, not done (wastes a turn).
  - If `max_turns` reached without an answer → reward 0.0, done.
- **Retrieval backend** — a `Retriever` abstraction with two implementations,
  selected by env vars set by `env_cache` provisioning:
  - **Dense E5 (preferred)**: when `SEARCH_QA_INDEX_DIR` is set and loadable —
    load the cached FAISS index + passage store from that dir, encode the query
    with `intfloat/e5-base-v2` (sentence-transformers, prefix `"query: "`), search
    top-k. Lazy-import sentence_transformers + faiss INSIDE the retriever so the
    module imports fine without them.
  - **BM25 fallback**: rank_bm25 `BM25Okapi` over (a) the task's own `contexts`
    if present, else (b) a small corpus the env was given. Used when the dense
    index is unavailable OR `SEARCH_QA_RETRIEVER=bm25`. Lazy-import rank_bm25;
    if that's *also* missing, fall back to a trivial lexical-overlap ranker (pure
    python) so the env always returns *something* real.
  - The retriever choice + fallbacks must be logged once (print) and surfaced in
    `last_info["retriever"]`.
- `info` on the terminal step: `{"f1": float, "em": 0/1, "n_search": int,
  "answered": bool, "retriever": "e5"|"bm25"|"overlap", "source": str}`.
- Provide module-level helpers `normalize_answer(s)`, `token_f1(pred, golds)`,
  `exact_match(pred, golds)` (SQuAD-style) — reused by the trainer/aggregator.
- Provide a `load_search_qa_tasks(n_per_source=256, sources=("nq","hotpotqa"))`
  that loads NQ-open (validation) + HotpotQA distractor (validation) via
  `datasets`, **carrying HotpotQA `context` into `contexts`** (flatten the
  sentence lists into paragraph strings). Lazy-import `datasets`; on load failure
  return `[]` for that source (fail-soft) and note it.

**Tests** (base venv, no rank_bm25/faiss/datasets — inject/monkeypatch):
- Action parsing: search/answer/garbage/code-fenced variants.
- `token_f1`/`normalize_answer`/`exact_match` against known SQuAD cases.
- A full episode with a FAKE retriever: search → obs contains passage → answer →
  reward == expected F1 → done, info populated.
- max_turns exhaustion → reward 0.0, done.
- Retriever selection logic with env vars monkeypatched + the dense/bm25 libs
  absent → falls back to overlap ranker without raising.

---

## 4. Module: `alfworld_env.py`  (author: AGENT-ALFWORLD)

**Why.** ALFWorld was de-scoped entirely. Restore it as a **real** TextWorld
agentic env via the installed `alfworld` package (`get_environment`).

**Class** `ALFWorldEnv(AgenticEnv)` in `backend/agents/rlm/alfworld_env.py`:

- `max_turns = 30` (ALFWorld episodes are long-horizon).
- Real backend via `from alfworld.agents.environment import get_environment`.
  Config: build a minimal config dict/yaml pointing at `ALFWORLD_DATA` (env var
  set by `env_cache`). Use the **TextWorld** (`AlfredTWEnv`) variant, not THOR
  (no rendering/GPU display). Lazy-import alfworld inside `reset`.
- `reset(seed, task)`: seed the env, load one game, `_start_episode(system=<the
  ALFWorld task instruction + admissible-action guidance>)`, record the initial
  observation (room description + goal). Return it.
- `step(action)`: send the action string to the TW env, get
  `(obs, reward, done, info)`. Record obs. ALFWorld actions are natural-language
  (`go to fridge 1`, `take apple 1 from countertop 2`). Strip any model
  scaffolding (a leading `>` or `action:`), pass the cleaned command. On `done`,
  `_finish(float(won), info={"success": won, "steps": turns})` where `won` comes
  from the env's success signal (`info["won"]` / goal-condition).
- **Fail-soft availability**: if `ALFWORLD_DATA` is unset OR `get_environment`
  import/construction fails OR no games found → the env must signal
  "unavailable" cleanly. Expose a classmethod `available() -> (bool, reason)` the
  trainer can check; if used anyway, `reset` returns a short observation and the
  first `step` returns `done=True, reward=0.0` with
  `info={"unavailable": True, "reason": ...}`. (The harness already converts an
  unavailable env into a verified rubric Exclusion via `env_cache`; this is the
  in-cell safety net.)

**Tests** (base venv has alfworld+textworld but NOT the downloaded data — so the
real game path can't run in CI): 
- `available()` returns `(False, reason)` when `ALFWORLD_DATA` unset → assert no
  raise, reason mentions data.
- Construct `ALFWorldEnv` with a **FAKE tw env injected** (constructor takes an
  optional `tw_env_factory` / `_env` seam): drive reset → step("go to fridge 1")
  → step to a terminal `won=True`; assert reward 1.0, done, info.success True,
  action cleaning strips a leading `> `.
- Bad action against the fake env → no raise, obs returned.

Design the constructor so the real `get_environment` path is the default and the
fake is injectable (mirror `env_cache`'s injected `downloader`/`probe` seams).

---

## 5. Module: `webshop_env.py`  (author: AGENT-WEBSHOP)

**Why.** WebShop was de-scoped. Restore it as a **real** agentic env talking to
the WebShop server (`env_cache.acquire_webshop` provides `WEBSHOP_URL`).

**Class** `WebShopEnv(AgenticEnv)` in `backend/agents/rlm/webshop_env.py`:

- `max_turns = 15`.
- Talks to the server at `WEBSHOP_URL` (env var). Use `requests` (lazy-import) or
  stdlib `urllib` — prefer stdlib `urllib.request` so no extra dep (more robust).
- `reset(seed, task)`: `task` selects a goal/instruction index (server exposes
  goals). `_start_episode(system=<WebShop action grammar: search[...], click[...]>)`,
  fetch the initial page (instruction + search box), record it, return.
- **Action grammar** (WebShop canonical): `search[<query>]` and `click[<button>]`.
  Parse defensively. Map to the server's step API; record the returned page text
  (available actions / product list / item page). On terminal `done`, `_finish`
  with the server's reward (∈[0,1], the WebShop matching score),
  `info={"success": reward>0.5, "reward": reward, "steps": turns}`.
- **Fail-soft availability**: `available() -> (bool, reason)` probing `WEBSHOP_URL`
  (env var present + a quick GET). If unavailable and used anyway: first `step`
  returns `done=True, reward=0.0, info={"unavailable": True, ...}` — no raise.
- Injectable HTTP seam (`http_get`/`http_post` or a `client` object) so the unit
  test drives a FAKE server without network.

**Tests** (base venv, no network):
- `available()` False + reason when `WEBSHOP_URL` unset, no raise.
- With an injected FAKE client: reset → `search[red shoes]` → obs lists items →
  `click[item]` → `click[buy]` → terminal reward from fake → done, info.
- Malformed action → nudge observation, no raise.

---

## 6. What the tech lead wires (NOT your job — listed for context)

- Add the four modules to `_HARNESS_CODE_HELPERS` (`baseline_implementation.py`).
- `env_cache.py`: build+cache the dense E5 wiki-18 index (export
  `SEARCH_QA_INDEX_DIR`/`SEARCH_QA_RETRIEVER`); fix the ALFWorld downloader to use
  the venv console-script path + run it once into the shared cache (export
  `ALFWORLD_DATA`); keep WebShop bring-up (`WEBSHOP_URL`).
- Wire `provision_scope` into the run pipeline; inject env-vars into the per-run
  env; fold verified `env_setup_failed` exclusions into `metrics.json::scope`.
- `FULL_SCOPE_ENV_GUIDANCE`: tell the agent to subclass `AgenticEnv`, import the
  shipped env modules + `agentic_rollout`, add cells for all three envs, and use
  the deeper-training defaults.
- `preflight_ast`: accept `AgenticEnv` as a valid base (currently only `BaseEnv`).
- Deeper training defaults: STEPS ↑ (≥400), GROUP_SIZE 8, larger token budgets.

---

## 7. Definition of done (per module)

- File created in `backend/agents/rlm/`, imports cleanly on the base venv
  (`python -c "import <mod>"` — heavy deps lazy/guarded).
- Unit test in `tests/agents/rlm/` passes on the base venv (no network, no heavy
  deps) using injected fakes.
- Fail-soft verified by a test (bad input / missing dep / missing data → no raise).
- Docstring explains the *why* + the action grammar / reward.
