#!/usr/bin/env bash
# ============================================================================
# One-shot Qwen3.8-27B VLM SFT launcher.
#
# All runtime choices come from environment variables; everything else is
# inherited from the YAML baseline. Extra CLI overrides can be appended as
# positional arguments (dotted --field=value form).
#
# Usage:
#   MODEL_PATH=/data1/models/Qwen3.8-27B \
#   DATA_PATH='/data1/datasets/my_ds/**/*.parquet' \
#   OUTPUT_DIR=/data1/outputs/qwen38_sft \
#   bash run_vlm_train.sh
#
# Environment variables:
#   CODE_DIR          (optional) repo root when this script is launched from
#                                outside the checkout (default: script dir).
#   MODEL_PATH        (required) HF weights dir of the base model.
#   DATA_PATH         (required) Parquet file, glob, directory, or manifest
#                                JSON. Directories need RECURSIVE=true when
#                                the parquets are nested.
#   OUTPUT_DIR        (required) SFT output root. SFT weights land in
#                               OUTPUT_DIR/hf_export/ (HF format).
#   NPROC_PER_NODE    (default 16) NPUs per node.
#   NNODES            (default 1)  number of nodes.
#   NODE_RANK         (default 0)  rank of this node (multi node).
#   MASTER_ADDR       (default 127.0.0.1; required on non-rank-0 nodes)
#   MASTER_PORT       (default 29760)
#   MICRO_BATCH       (default 1)
#   GLOBAL_BATCH      (default = MICRO_BATCH * dp_size, one micro-step)
#   TRAIN_ITERS       (default 10)
#   MAX_SEQ_LEN       (default 8192)
#   MAX_PIXELS        (default 4194304) image token budget (~area/784);
#                                doc pages need ~1-4M px to stay readable
#   MIN_PIXELS        (default 65536)
#   SCHEMA_ADAPTER    (default auto)    qwen_vl | fintabnet_otsl | llava |
#                                       image_text_pair | auto
#   RECURSIVE         (default false)   recursive scan for dir DATA_PATH
#   EAGER_FILTER      (default false)   true: full-corpus pre-scan at startup
#   MANIFEST          (optional) dataset_manifest.json path; overrides
#                                DATA_PATH when set.
#   WEIGHTS           (optional) comma list "name:weight,name:weight" to
#                                override manifest sampling weights.
#   SAVE_EVERY        (default 50)   checkpoint interval (optimizer steps).
#   KEEP_LAST         (optional)     checkpoint rotation limit.
#   RESUME_FROM       (optional)     checkpoint dir (global_step_N) to resume.
#   LR                (default 1.0e-5)
#   WARMUP            (default 10)
#   EXTRA_OVERRIDES   (optional) additional --field=value tokens appended
#                                verbatim to the trainer command line.
#
# SFT weight output:
#   The trainer saves a DCP checkpoint every SAVE_EVERY steps and always
#   HF-format weights to <OUTPUT_DIR>/checkpoints/global_step_<N>/hf_ckpt/
#   at each save point (Qwen3.8-27B tie_word_embeddings=false, so this is safe).
# ============================================================================
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}
PROJECT_ROOT=$(cd "${CODE_DIR}" && pwd)
SCRIPT_DIR="${PROJECT_ROOT}"

# ---- required arguments ------------------------------------------------------
: "${MODEL_PATH:?Set MODEL_PATH to the base model directory (HF format)}"
: "${DATA_PATH:?Set DATA_PATH to a parquet file/glob/dir or manifest JSON}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the SFT output root directory}"

# ---- training topology -------------------------------------------------------
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29760}
RDZV_ID=${RDZV_ID:-qwen38_vlm_$$}

# ---- training hyper parameters ----------------------------------------------
MICRO_BATCH=${MICRO_BATCH:-1}
TRAIN_ITERS=${TRAIN_ITERS:-10}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
MAX_PIXELS=${MAX_PIXELS:-4194304}
MIN_PIXELS=${MIN_PIXELS:-65536}
SCHEMA_ADAPTER=${SCHEMA_ADAPTER:-auto}
RECURSIVE=${RECURSIVE:-false}
EAGER_FILTER=${EAGER_FILTER:-false}
SAVE_EVERY=${SAVE_EVERY:-50}
LR=${LR:-1.0e-5}
WARMUP=${WARMUP:-10}

# ---- derived -----------------------------------------------------------------
DP_SIZE=$(( NNODES * NPROC_PER_NODE ))
GLOBAL_BATCH=${GLOBAL_BATCH:-$(( MICRO_BATCH * DP_SIZE ))}
if [ $(( GLOBAL_BATCH % (MICRO_BATCH * DP_SIZE) )) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH ($GLOBAL_BATCH) must be divisible by MICRO_BATCH×dp_size ($MICRO_BATCH×$DP_SIZE)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
RUN_TAG=$(date +%m%d_%H%M%S)
LOGDIR="$OUTPUT_DIR/logs/$RUN_TAG"
mkdir -p "$LOGDIR"

# ---- build CLI overrides ------------------------------------------------------
OVERRIDES=()
OVERRIDES+=("--model.pretrained_model_name_or_path=${MODEL_PATH}")
OVERRIDES+=("--training.train_iters=${TRAIN_ITERS}")
OVERRIDES+=("--training.micro_batch_size=${MICRO_BATCH}")
OVERRIDES+=("--training.global_batch_size=${GLOBAL_BATCH}")
OVERRIDES+=("--fsdp_config.dp_shard_size=${DP_SIZE}")
OVERRIDES+=("--optimizer.fp32_main_params=true")
OVERRIDES+=("--dataset.data_transform.max_seq_len=${MAX_SEQ_LEN}")
OVERRIDES+=("--dataset.data_transform.max_pixels=${MAX_PIXELS}")
OVERRIDES+=("--dataset.data_transform.min_pixels=${MIN_PIXELS}")
OVERRIDES+=("--optimizer.adamw_config.adamw_lr=${LR}")
OVERRIDES+=("--lr_scheduler.lr_config.lr_warmup_steps=${WARMUP}")
OVERRIDES+=("--checkpoint.checkpoint_dir=${OUTPUT_DIR}/checkpoints")
OVERRIDES+=("--checkpoint.save_steps=${SAVE_EVERY}")
OVERRIDES+=("--checkpoint.save_hf_weights=true")

if [ -n "${MANIFEST:-}" ]; then
    # Apply weight overrides to a temp manifest copy.
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
    # A manifest passed through DATA_PATH also works.
    OVERRIDES+=("--dataset.data_config.manifest=${DATA_PATH}")
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

# ---- runtime environment ------------------------------------------------------
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

echo "=============================================================="
echo " MODEL_PATH      : ${MODEL_PATH}"
if [ -n "${MANIFEST:-}" ]; then echo " DATA_PATH       : manifest=${MANIFEST}"; else echo " DATA_PATH       : ${DATA_PATH}"; fi
echo " OUTPUT_DIR      : ${OUTPUT_DIR}"
echo " SFT weights to  : ${OUTPUT_DIR}/checkpoints/global_step_<N>/hf_ckpt/"
echo " topology        : ${NNODES} nodes × ${NPROC_PER_NODE} NPUs (dp=${DP_SIZE})"
echo " batch           : micro=${MICRO_BATCH} × dp=${DP_SIZE} → global=${GLOBAL_BATCH}"
echo " iters/lr        : ${TRAIN_ITERS} / ${LR} (warmup ${WARMUP})"
echo " seq/pixels      : ${MAX_SEQ_LEN} / max ${MAX_PIXELS}"
echo " schema/manifest : ${SCHEMA_ADAPTER} / ${MANIFEST:-inline}"
echo " checkpoint      : every ${SAVE_EVERY} steps → ${OUTPUT_DIR}/checkpoints"
echo " resume          : ${RESUME_FROM:-none}"
echo " log             : ${LOGDIR}/train.log"
echo "=============================================================="

cd "${PROJECT_ROOT}"
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.qwen3_8.train_vlm \
    "${SCRIPT_DIR}/examples/qwen3_8/train_vlm_oneshot.yaml" \
    "${OVERRIDES[@]}" \
    ${EXTRA_OVERRIDES:-} \
    2>&1 | tee "${LOGDIR}/train.log"
status=${PIPESTATUS[0]}
echo "$status" > "${LOGDIR}/exit_code"
echo "training exit: $status (log: ${LOGDIR}/train.log)"
exit "$status"
