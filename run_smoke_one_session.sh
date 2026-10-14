#!/usr/bin/env bash
set -u
LOGDIR=/tmp/hyper-parallel-fork-smoke/logs/$(date +%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "$LOGDIR" > /tmp/hyper-parallel-fork-smoke/latest_logdir

cleanup() {
  python3 - <<'PY'
import os, signal
needles=('examples.qwen3_8.train_vlm','torchrun --nnodes=1 --nproc_per_node=16')
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
    if any(x in cmd for x in needles):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
PY
}
trap cleanup EXIT

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29752
export RDZV_ID=qwen38_vlm_tiny_16npu_cp1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYPER_PARALLEL_PLATFORM=torch
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_NPU_SOCKET_PORT_RANGE=22000-22200
export OMP_NUM_THREADS=4
export PYTHONPATH=/tmp/qwen38_vlm_smoke/vendor:/tmp/hyper-parallel-fork-smoke${PYTHONPATH:+:$PYTHONPATH}

cd /tmp/hyper-parallel-fork-smoke
torchrun \
  --nnodes=1 \
  --nproc_per_node=16 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29752 \
  --rdzv_id=qwen38_vlm_tiny_16npu_cp1 \
  --module examples.qwen3_8.train_vlm \
  examples/qwen3_8/train_vlm_tiny.yaml \
  > "$LOGDIR/train.log" 2>&1
status=$?
echo "$status" > "$LOGDIR/exit_code"
exit "$status"
