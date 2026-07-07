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
    'DEPENDENCY INSTALL (GKE cell) - the prior cell failed at bootstrap: pip could not '
    'install -e ./verl (not a valid editable requirement). NEVER put -e ./verl or any '
    '-e ./<dir> in requirements.txt unless that dir is a real staged pip project '
    '(pyproject.toml or setup.py present in code/). Install the verl RL framework as a '
    'resolvable spec instead: prefer pip install verl (it is on PyPI) or '
    'pip install git+https://github.com/volcengine/verl.git. verl builds native extensions '
    '- ensure g++ and ninja exist, cap parallelism with MAX_JOBS=4 to avoid compile OOM, '
    'pin a tensordict compatible with the cell torch, and avoid flash-attn compilation '
    '(use eager or sdpa attention) for robustness. requirements.txt must list only concrete '
    'installable specs (no bare editable paths to missing dirs). For the smallest-slice '
    'DeepSeek-1.5B UCPO run keep the dependency footprint minimal.'
)
sys.argv = ['backend.cli','reproduce','2605.00365',
            '--project-id','prj_618445173e9ae4f2','--resume',
            '--mode','rlm','--sandbox','gcp','--model','opus-foundry','--provider','anthropic',
            '--models','executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry,spec_validator=grok',
            '--max-usd','60','--max-repair-iterations','3']
runpy.run_module('backend.cli', run_name='__main__')
" > "$LOG" 2>&1 &
echo "launched pid $! -> log $LOG"
