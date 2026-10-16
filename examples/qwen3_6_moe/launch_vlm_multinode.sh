#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Launch Qwen3.6-35B-A3B MoE VLM training on one or more nodes (910C A3).
# Run the same command on every node. c10d assigns ranks; NODE_RANK is not used.
# RDZV_ID must be identical on every node. The default is stable for a given
# topology so nodes do not need to invent one; set it explicitly for concurrent jobs.
#
# Usage (every node):
#   MASTER_ADDR=<rank0-ip> NNODES=4 \
#   MODEL_PATH=/mnt/sfs/models/Qwen3.6-35B-A3B \
#   DATA_PATH=/mnt/sfs/datasets/train.parquet \
#   OUTPUT_DIR=/mnt/sfs/outputs/sft \
#   bash examples/qwen3_6_moe/launch_vlm_multinode.sh
#
# Topology (VLM keeps TP=CP=PP=1):
#   dp_shard = NNODES * NPROC_PER_NODE
#   edp_shard = dp_shard / EP_SIZE
# Image budgets match upstream run_vlm_train.sh (document pages stay readable):
#   MAX_SEQ_LEN=8192  MIN_PIXELS=65536  MAX_PIXELS=4194304

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# ---- topology ----
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
EP_SIZE=${EP_SIZE:-1}
if [ "${EP_SIZE}" -lt 1 ]; then
    echo "ERROR: EP_SIZE must be >= 1 (got ${EP_SIZE})" >&2
    exit 1
fi

if [ "${NNODES}" -gt 1 ] && [ -z "${MASTER_ADDR:-}" ]; then
    echo "ERROR: set MASTER_ADDR to the rank0 address when NNODES>1" >&2
    exit 1
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29760}
RDZV_ID=${RDZV_ID:-qwen36moe_vlm_${NNODES}n${NPROC_PER_NODE}}

# ---- derived topology ----
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
DP_SIZE=$(( WORLD_SIZE ))
EDP_SHARD=$(( DP_SIZE / EP_SIZE ))

if [ $(( DP_SIZE % EP_SIZE )) -ne 0 ]; then
    echo "ERROR: DP_SIZE ($DP_SIZE) not divisible by EP_SIZE ($EP_SIZE)" >&2
    exit 1
fi

# ---- paths (yaml defaults apply when unset) ----
MODEL_PATH=${MODEL_PATH:-/data1/models/Qwen3.6-35B-A3B}
DATA_PATH=${DATA_PATH:-/data1/f00538480/dataset/HW_Chinese/hw_chinese_ocr_train_tiny.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/qwen3_6_moe_vlm_multinode}

# ---- hyper parameters ----
TRAIN_ITERS=${TRAIN_ITERS:-100}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-$(( MICRO_BATCH * DP_SIZE ))}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
MAX_PIXELS=${MAX_PIXELS:-4194304}
MIN_PIXELS=${MIN_PIXELS:-65536}
SAVE_EVERY=${SAVE_EVERY:-50}
LR=${LR:-1.0e-5}
WARMUP=${WARMUP:-10}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MIN_LR=${MIN_LR:-1.0e-6}
SEED=${SEED:-42}
SAVE_OPTIMIZER=${SAVE_OPTIMIZER:-false}
SAVE_HF_WEIGHTS=${SAVE_HF_WEIGHTS:-false}
SCHEMA_ADAPTER=${SCHEMA_ADAPTER:-auto}
RECURSIVE=${RECURSIVE:-false}
EAGER_FILTER=${EAGER_FILTER:-false}

if [ $(( GLOBAL_BATCH % (MICRO_BATCH * DP_SIZE) )) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH ($GLOBAL_BATCH) must be divisible by MICRO_BATCH×dp_size ($MICRO_BATCH×$DP_SIZE)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
RUN_TAG=$(date +%m%d_%H%M%S)
LOGDIR="$OUTPUT_DIR/logs/${RUN_TAG}_$(hostname -s)"
mkdir -p "$LOGDIR"

# ---- runtime env ----
export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $(( NPROC_PER_NODE - 1 )))}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-22000-22200}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

VENDOR_DIR=${VENDOR_DIR:-/tmp/qwen38_vlm_smoke/vendor}
if [ -d "${VENDOR_DIR}" ]; then
    export PYTHONPATH="${VENDOR_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# ---- build overrides ----
OVERRIDES=()
OVERRIDES+=("--model.pretrained_model_name_or_path=${MODEL_PATH}")
OVERRIDES+=("--training.train_iters=${TRAIN_ITERS}")
OVERRIDES+=("--training.global_batch_size=${GLOBAL_BATCH}")
OVERRIDES+=("--training.micro_batch_size=${MICRO_BATCH}")
OVERRIDES+=("--fsdp_config.dp_shard_size=${DP_SIZE}")
OVERRIDES+=("--fsdp_config.edp_shard_size=${EDP_SHARD}")
OVERRIDES+=("--accelerator.ep_size=${EP_SIZE}")
OVERRIDES+=("--optimizer.fp32_main_params=true")
OVERRIDES+=("--optimizer.adamw_config.adamw_lr=${LR}")
OVERRIDES+=("--optimizer.adamw_config.adamw_weight_decay=${WEIGHT_DECAY}")
OVERRIDES+=("--training.max_grad_norm=${MAX_GRAD_NORM}")
OVERRIDES+=("--lr_scheduler.lr_config.lr_warmup_steps=${WARMUP}")
OVERRIDES+=("--lr_scheduler.lr_config.min_lr=${MIN_LR}")
OVERRIDES+=("--training.seed=${SEED}")
OVERRIDES+=("--dataset.data_transform.min_pixels=${MIN_PIXELS}")
OVERRIDES+=("--dataset.data_transform.max_seq_len=${MAX_SEQ_LEN}")
OVERRIDES+=("--dataset.data_transform.max_pixels=${MAX_PIXELS}")
OVERRIDES+=("--checkpoint.checkpoint_dir=${OUTPUT_DIR}/checkpoints")
OVERRIDES+=("--checkpoint.save_steps=${SAVE_EVERY}")
OVERRIDES+=("--checkpoint.save_hf_weights=${SAVE_HF_WEIGHTS}")
OVERRIDES+=("--checkpoint.save_optimizer=${SAVE_OPTIMIZER}")

if [ -n "${MANIFEST:-}" ]; then
    EFFECTIVE_MANIFEST="$MANIFEST"
    if [ -n "${WEIGHTS:-}" ]; then
        EFFECTIVE_MANIFEST="$LOGDIR/manifest_adjusted.json"
        python3 - "$MANIFEST" "$EFFECTIVE_MANIFEST" "$WEIGHTS" <<'PY'
import json, sys
src, dst, weights = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as handle:
    manifest = json.load(handle)
overrides = {}
for pair in weights.split(","):
    name, _, value = pair.partition(":")
    overrides[name.strip()] = float(value)
for dataset in manifest.get("datasets", []):
    if dataset.get("name") in overrides:
        dataset["weight"] = overrides[dataset["name"]]
with open(dst, "w") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)
print(f"weights adjusted: {overrides}")
PY
    fi
    OVERRIDES+=("--dataset.data_config.manifest=${EFFECTIVE_MANIFEST}")
    OVERRIDES+=("--dataset.data_config.eager_filter=${EAGER_FILTER}")
elif [ -f "$DATA_PATH" ] && [[ "$DATA_PATH" == *.json ]]; then
    OVERRIDES+=("--dataset.data_config.manifest=${DATA_PATH}")
    OVERRIDES+=("--dataset.data_config.eager_filter=${EAGER_FILTER}")
else
    OVERRIDES+=("--dataset.data_path=${DATA_PATH}")
    OVERRIDES+=("--dataset.data_config.source_type=parquet")
    OVERRIDES+=("--dataset.data_config.schema_adapter=${SCHEMA_ADAPTER}")
    OVERRIDES+=("--dataset.data_config.lazy_parquet=true")
    OVERRIDES+=("--dataset.data_config.recursive=${RECURSIVE}")
    OVERRIDES+=("--dataset.data_config.eager_filter=${EAGER_FILTER}")
fi

if [ -n "${KEEP_LAST:-}" ]; then
    OVERRIDES+=("--checkpoint.save_total_limit=${KEEP_LAST}")
fi
if [ -n "${RESUME_FROM:-}" ]; then
    OVERRIDES+=("--checkpoint.restore_from=${RESUME_FROM}")
fi

echo "================================================================"
echo " Qwen3.6-35B-A3B MoE VLM — ${NNODES} nodes × ${NPROC_PER_NODE} die"
echo " world=${WORLD_SIZE} dp=${DP_SIZE} ep=${EP_SIZE} edp_shard=${EDP_SHARD}"
echo " gbs=${GLOBAL_BATCH} mbs=${MICRO_BATCH} iters=${TRAIN_ITERS}"
echo " seq/pixels=${MAX_SEQ_LEN} / min ${MIN_PIXELS} max ${MAX_PIXELS}"
echo " master=${MASTER_ADDR}:${MASTER_PORT} rdzv=${RDZV_ID}"
echo " model=${MODEL_PATH}"
echo " data=${MANIFEST:-$DATA_PATH}"
echo " output=${OUTPUT_DIR}"
echo " log=${LOGDIR}/train.log"
echo "================================================================"

cd "${PROJECT_ROOT}"
set +e
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.qwen3_8.train_vlm \
    "${SCRIPT_DIR}/train_vlm_qwen36moe_8dies.yaml" \
    "${OVERRIDES[@]}" \
    ${EXTRA_OVERRIDES:-} \
    "$@" \
    2>&1 | tee "${LOGDIR}/train.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "${LOGDIR}/exit_code"
echo "training exit: $status (log: ${LOGDIR}/train.log)"
exit "$status"
