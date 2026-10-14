#!/usr/bin/env bash
# Sweep micro_batch_size 1..4 on 16 dies with the tiny dataset, reporting
# per-rank peak memory. Run from the repo root (NPU required).
set -u

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS="$BASE_DIR/logs/mb_sweep_results.txt"
mkdir -p "$BASE_DIR/logs"
> "$RESULTS"

cleanup() {
  python3 - <<'PY'
import os, signal
needles=('examples.qwen3_8.train_vlm','torchrun')
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid == os.getpid():
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline','rb').read().decode(errors='ignore').replace('\x00',' ')
    except Exception:
        continue
    if any(x in cmd for x in needles) and 'mb_sweep' not in cmd:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
PY
}
trap cleanup EXIT

run_mb() {
  local mb=$1
  local yaml=$BASE_DIR/examples/qwen3_8/train_vlm_tiny_16dies.yaml
  local LOGDIR=$BASE_DIR/logs/16npu_mb${mb}_$(date +%m%d_%H%M%S)
  mkdir -p "$LOGDIR"
  echo "$LOGDIR" > "$BASE_DIR/latest_logdir"
  echo "=== micro_batch=$mb $(date) ===" | tee -a "$RESULTS"

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=29760
  export RDZV_ID=qwen38_vlm_mb${mb}_sweep
  export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export HYPER_PARALLEL_PLATFORM=torch
  export HCCL_CONNECT_TIMEOUT=1800
  export HCCL_EXEC_TIMEOUT=1800
  export HCCL_NPU_SOCKET_PORT_RANGE=22200-22400
  export OMP_NUM_THREADS=4
  export PYTHONPATH=/tmp/qwen38_vlm_smoke/vendor:"$BASE_DIR"${PYTHONPATH:+:$PYTHONPATH}

  cd "$BASE_DIR"
  torchrun \
    --nnodes=1 \
    --nproc_per_node=16 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29760 \
    --rdzv_id="$RDZV_ID" \
    --module examples.qwen3_8.train_vlm \
    "$yaml" \
    --training.micro_batch_size="$mb" \
    --training.global_batch_size=$((mb * 16)) \
    > "$LOGDIR/train.log" 2>&1
  local status=$?
  echo "$status" > "$LOGDIR/exit_code"

  local max_alloc=$(grep -o 'memory/device_max_allocated_gb=[0-9.]*' "$LOGDIR/train.log" | sed 's/.*=//' | sort -rn | head -1)
  local max_reserved=$(grep -o 'memory/device_max_reserved_gb=[0-9.]*' "$LOGDIR/train.log" | sed 's/.*=//' | sort -rn | head -1)
  local loss=$(grep -o 'training/total_loss=[0-9.]*' "$LOGDIR/train.log" | sed 's/.*=//' | tail -1)
  local step_time=$(grep -o 'performance/step_time=[0-9.]*' "$LOGDIR/train.log" | sed 's/.*=//' | tail -1)
  echo "  mb=$mb status=$status max_alloc=${max_alloc:-N/A}GB max_reserved=${max_reserved:-N/A}GB loss=${loss:-N/A} step_time=${step_time:-N/A}s" | tee -a "$RESULTS"
  echo "  log: $LOGDIR/train.log" | tee -a "$RESULTS"
  sleep 10
}

for mb in 1 2 3 4; do
  run_mb "$mb"
done
echo "=== sweep complete $(date) ===" | tee -a "$RESULTS"
