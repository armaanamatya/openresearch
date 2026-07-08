#!/usr/bin/env bash
# Resume the UCPO (arXiv 2605.00365) reproduction prj_618445173e9ae4f2 on GKE.
#
# Replays the original run_config.json env verbatim, then applies this run's overrides:
#   - GPU_COUNT 4 -> 1        : reliable single A100-80 on the available a100-80-rw pool
#                              (the 4-GPU pool can stock out at node-provision time).
#   - GKE_SYNTH_CELL=1        : the §5 code-staging fix (route commands.json-only via cell-matrix).
#   - PREFLIGHT_UNION_SCOPE=1 : landed fix, was OFF — widen preflight to the training-file union.
#   - IMPL_ABANDON_GUARD=1    : landed fix, was OFF — no "ok" on an aclose-stall give-up.
#   - HARDEXIT_CLEANUP=1      : landed fix, was OFF — bounded child cleanup on hard-exit.
#
# cli.py does NOT load .env into os.environ, hence the load_dotenv wrapper (Foundry key + GCP).
# Run from a clean shell:  bash scripts/resume_ucpo_618.sh
set -u
cd /home/abheekp/openresearch
export PATH="$HOME/.local/bin:$PATH"

LOG=/tmp/ucpo_618_resume.log
nohup .venv/bin/python -c "
import json, os, sys, runpy
from dotenv import load_dotenv
load_dotenv('.env')
cfg = json.load(open('runs/prj_618445173e9ae4f2/run_config.json'))
os.environ.update({k: str(v) for k, v in cfg.get('env_flags', {}).items()})
# --- this-resume overrides ---
os.environ['OPENRESEARCH_GPU_COUNT'] = '1'
os.environ['OPENRESEARCH_GKE_SYNTH_CELL'] = '1'
os.environ['OPENRESEARCH_PREFLIGHT_UNION_SCOPE'] = '1'
os.environ['OPENRESEARCH_IMPL_ABANDON_GUARD'] = '1'
os.environ['OPENRESEARCH_HARDEXIT_CLEANUP'] = '1'
os.environ['OPENRESEARCH_BASELINE_EXTRA_GUIDANCE'] = (
    'EXECUTE MODE - RUN THE AUTHOR REPO, DO NOT RE-IMPLEMENT. The cloned UCPO repo is '
    'seeded into code/ (README, ucpo/, verl/, dataset/, scripts_c/). Reproduce by RUNNING '
    'THE AUTHORS pipeline, never a hand-written train.py. '
    '(1) Install the BUNDLED verl (a valid pip project): cd verl and pip install -e . '
    '(NOT -e ./verl from the repo root); cap MAX_JOBS=4, avoid flash-attn (use eager/sdpa), '
    'pin a torch-compatible tensordict. '
    '(2) Train via python3 -m ucpo.main_run with the args from scripts_c/run_ucpo_1_5b.sh '
    '(algorithm.adv_estimator=iq, algorithm.tau=0.2, algorithm.alpha=0.0, '
    'model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, data.train_files=dataset/train_data_10k.parquet, '
    'data.val_files=dataset/valid_data.parquet, actor_rollout_ref.rollout.name=vllm). '
    '(3) ADAPT ONLY for a fast smallest-slice proof on ONE A100-80: cut training to a few '
    'steps (small NUM_EPISODES / a handful of PPO updates), set trainer.n_gpus_per_node=1 and '
    'tensor_model_parallel_size=1, reduce data.train_batch_size and actor.ppo_mini_batch_size, '
    'set rollout.gpu_memory_utilization about 0.6, enable_gradient_checkpointing=True. '
    'Goal: a NON-ZERO RLVR reward (the boxed math-answer reward rising above 0), not full-scale '
    'training. '
    '(4) Use the AUTHORS reward (verl prime_math grader on boxed answers) - do NOT reinvent '
    'the reward or answer-matching. '
    '(5) Write the final reward/accuracy to OUTPUT_DIR/metrics.json. '
    '(6) IGNORE and delete the SDAR scaffold files in code/ (alfworld_env.py, webshop_env.py, '
    'search_qa_env.py, sdar_env_base.py, agentic_rollout.py) - they are from a DIFFERENT paper '
    'and irrelevant to UCPO math-reasoning RL. commands.json must invoke ucpo.main_run, never a '
    'hand-written train.py.'
)
sys.argv = ['backend.cli','reproduce','2605.00365',
            '--project-id','prj_618445173e9ae4f2','--resume',
            '--mode','rlm','--sandbox','gcp','--model','opus-foundry','--provider','anthropic',
            '--models','executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry,spec_validator=grok',
            '--max-usd','60','--max-repair-iterations','3']
runpy.run_module('backend.cli', run_name='__main__')
" > "$LOG" 2>&1 &
echo "launched pid $! -> log $LOG"
