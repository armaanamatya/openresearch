"""Sandbox execution contract — single source of truth for the agent↔runtime
interface.

Every agent that emits commands or scripts the experiment sandbox will run
(baseline-implementation, improvement-path, composition) MUST include this
contract in its prompt. The runtime in ``backend/services/runtime/local_docker``
(docker), ``local_process.py`` (local), and ``runpod_backend.py`` (runpod)
all expose the same mount model and the same env vars — so this single
contract is uniform across the three sandbox modes. Write scripts to the
contract and they work on all three.

Two design rules for the contract text:
- Brace-free. The constant is concatenated into prompts that may or may not
  be ``.format()``-ed by the orchestrator (e.g. COMPOSITION_AGENT_PROMPT is
  formatted; BASELINE_IMPLEMENTATION_PROMPT is not). Avoiding ``{`` / ``}``
  in this constant keeps it spliceable into both without escaping.
- Domain-agnostic. The contract describes mounts, env vars, and write
  patterns — never paper-specific or framework-specific terminology. It
  applies identically to RL, vision, NLP, tabular, and any other domain.
"""

SANDBOX_EXECUTION_CONTRACT = """
# Sandbox Execution Contract
Every command, script, and code path that runs inside the experiment sandbox
MUST follow this contract. It is identical across the docker, local, and
runpod sandboxes — write scripts to the contract and they work on all three.

## Mounts (inside the running container)
- Your project directory is mounted READ-ONLY at `/code` (the container working
  dir). You may READ your code freely, but ANY write under it — `mkdir`, `tee`,
  shell `>`, `cp` into the tree, `git clone .`, file output from a script —
  WILL FAIL with a "Read-only file system" error.
- A separate writable volume is mounted for outputs. Its path inside the
  container is in the env var `OUTPUT_DIR`. Treat `$OUTPUT_DIR` as the
  ONLY writable surface.

## Environment variables (set for every command, every sandbox mode)
- `OUTPUT_DIR`            -> writable artifact directory; use for ALL outputs
- `OPENRESEARCH_ARTIFACT_DIR` -> alias of OUTPUT_DIR
- `MPLCONFIGDIR`          -> `$OUTPUT_DIR/.matplotlib` (matplotlib cache target)
- `PYTHONUNBUFFERED`      -> `1`

## Required patterns (apply to EVERY script and command you write)
1. Reference `$OUTPUT_DIR` by name — do NOT hardcode `/artifacts`; the
   resolved path differs across sandbox modes.
2. Direct EVERY output — logs, metrics, plots, checkpoints, model weights,
   generated configs, intermediate files — under `$OUTPUT_DIR`. Create the
   subdirectory explicitly if you need one. Concrete pattern:

       mkdir -p "$OUTPUT_DIR/results"
       python train.py --output-dir "$OUTPUT_DIR/results"
       bash smoke_test.sh 2>&1 | tee "$OUTPUT_DIR/smoke_test.log"

3. Point cache-hungry tools at `$OUTPUT_DIR` explicitly so they don't try
   to write under the read-only project. Set as many as apply BEFORE the
   command that needs them:

       HF_HOME="$OUTPUT_DIR/hf_cache"
       TRANSFORMERS_CACHE="$OUTPUT_DIR/hf_cache"
       XDG_CACHE_HOME="$OUTPUT_DIR/xdg_cache"
       TRITON_CACHE_DIR="$OUTPUT_DIR/triton_cache"
       TORCH_HOME="$OUTPUT_DIR/torch_cache"
       PIP_CACHE_DIR="$OUTPUT_DIR/pip_cache"
       TMPDIR="$OUTPUT_DIR/tmp"

4. Inside Python, prefer `os.environ["OUTPUT_DIR"]` over relative paths.
   A function that takes an `output_dir` argument is easier to test than
   one that hard-codes `./results`.
5. The `metrics.json` your experiment reports MUST be written to
   `$OUTPUT_DIR/metrics.json`. The orchestrator reads it from there to
   populate the run's final report. If your run produces no metrics,
   write an empty JSON object to that path — never omit the file.

Violating this contract is the single most common reason a reproduction
fails in the sandbox even when the code itself is correct. Before returning
your output, verify every command and every generated script against rules
1-5 above.
"""
