#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Launch Qwen3.8-27B VLM training on 32 nodes with 8 NPUs each after sourcing CANN.
# Topology: world_size=256, dp_size=256, dp_shard=256, CP=1 (VLM requires TP=CP=PP=1).
# global_batch_size=256 → num_micro_batches = 256/(1×256) = 1 per optimizer step.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous node address}"
MASTER_PORT=${MASTER_PORT:-29760}
RDZV_ID=${RDZV_ID:-qwen38_vlm_32nodes_256dies}

export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-22000-22200}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Add Pillow/torchvision vendor path if present (needed for VLM image decoding).
VENDOR_DIR=/tmp/qwen38_vlm_smoke/vendor
if [ -d "${VENDOR_DIR}" ]; then
    export PYTHONPATH="${VENDOR_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cd "${PROJECT_ROOT}"
torchrun \
    --nnodes=32 \
    --nproc_per_node=8 \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.qwen3_8.train_vlm \
    "${SCRIPT_DIR}/train_vlm_32nodes_256dies.yaml" \
    "$@"
