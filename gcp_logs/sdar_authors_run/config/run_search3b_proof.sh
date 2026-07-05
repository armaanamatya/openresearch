#!/bin/bash
# Faithful authors' SDAR run: Search-QA / Qwen2.5-3B (verl + vLLM 0.11.0, 4xA100, 150 steps).
# Self-contained (mirrors sdar_authors_repro.sh run_script + retriever_start). WANDB offline.
set -uo pipefail
CACHE=/mnt/sdar-cache
REPO=$CACHE/SDAR
ENV_SDAR=$CACHE/conda/envs/sdar
ENV_RET=$CACHE/conda/envs/retriever
LOG=$CACHE/logs; mkdir -p "$LOG"
export HF_HOME=$CACHE/hf
ln -sfn "$CACHE/data" "$HOME/data"

echo "=== $(date -u) PREFLIGHT: verl.trainer.main_sdar import ==="
cd "$REPO"
if ! "$ENV_SDAR/bin/python3" -c 'import verl.trainer.main_sdar' 2>/tmp/pf.err; then
  echo "PREFLIGHT FAILED:"; cat /tmp/pf.err; exit 3
fi
echo "preflight OK"

echo "=== $(date -u) START retriever (faiss_gpu, e5, port 8000) ==="
( export PATH=$ENV_RET/bin:$PATH HF_HOME=$CACHE/hf
  exec python3 "$REPO/examples/search/retriever/retrieval_server.py" \
    --index_path  "$CACHE/data/searchR1/e5_Flat.index" \
    --corpus_path "$CACHE/data/searchR1/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu --port 8000 >> "$LOG/retrieval_server.log" 2>&1 ) &
RET=$!; echo "$RET" > /tmp/sdar_retriever.pid
echo "retriever pid=$RET; polling /retrieve health up to 6 min (64GB index load + e5 dl)"
HEALTHY=0
for i in $(seq 1 18); do
  sleep 20
  if curl -sf --max-time 8 -H 'Content-Type: application/json' \
       -d '{"query":"test","topk":1}' http://0.0.0.0:8000/retrieve >/dev/null 2>&1; then
    echo "RETRIEVER HEALTHY after $((i*20))s"; HEALTHY=1; break
  fi
  kill -0 "$RET" 2>/dev/null || { echo "RETRIEVER DIED early; log tail:"; tail -8 "$LOG/retrieval_server.log"; exit 4; }
done
[ $HEALTHY -eq 1 ] || echo "WARN: retriever not healthy after 6min (continuing; training may error) — tail: $(tail -5 "$LOG/retrieval_server.log" | tr '\n' '|')"

echo "=== $(date -u) RUN run_search_3b (Qwen2.5-3B, 4xA100, vllm, 150 steps, wandb offline) ==="
( cd "$REPO"
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=$CACHE/hf WANDB_MODE=offline
  export PATH=$ENV_SDAR/bin:$PATH
  bash examples/sdar_trainer/run_search_3b.sh
) 2>&1 | tee "$LOG/run_search_3b.log"
RC=${PIPESTATUS[0]}
echo "=== $(date -u) run_search_3b EXIT rc=$RC ==="
kill "$(cat /tmp/sdar_retriever.pid 2>/dev/null)" 2>/dev/null || true
echo "=== SEARCH-3B PROOF COMPLETE rc=$RC ==="
