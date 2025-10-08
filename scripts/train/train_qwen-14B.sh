#!/bin/bash
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_GDR_LEVEL=4
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_TC=186
# export NCCL_NVLS_ENABLE=0
# export NCCL_IB_GID_INDEX=3
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_RETRY_CNT=7
# export NCCL_IB_HCA=^=mlx5_3,mlx5_4,mlx5_5,mlx5_bond_0
# ulimit -l unlimited

# 分布式训练参数
nnodes=$PET_NNODES
current_rank=$PET_NODE_RANK
master_addr=$MASTER_ADDR
master_port=$MASTER_PORT
input_path=${1}
output_path=${2}

# for rerun the task (only on single node setup)
cd /inspire/hdd/project/qproject-fundationmodel/public/sunjie/slime
pip3 config set global.extra-index-url ''
pip install -e . -i http://nexus.sii.shaipower.online/repository/pypi/simple --trusted-host nexus.sii.shaipower.online



# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export WANDB_DISABLED=true

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source /inspire/hdd/project/qproject-fundationmodel/public/sunjie/slime_env_pkgs/slime/scripts/models/qwen3-14B.sh

CKPT_ARGS=(
   --hf-checkpoint /inspire/hdd/project/qproject-fundationmodel/public/sunjie/workspace/LLMs/Qwen3-14B
   --ref-load /inspire/hdd/project/qproject-fundationmodel/public/sunjie/sft/ckpts/Qwen3-14B-Megatron
   --save ${output_path}
   --save-interval 1000
   --no-save-optim
   --no-save-rng
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${input_path}
   --input-key messages
   --tool-key tools
   --rollout-shuffle
   --num-epoch 4
   --rollout-batch-size 64
   --global-batch-size 64

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)


PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 65536
)


OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-warmup-iters 2
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.9
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)



MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
if [ "$current_rank" = "0" ]; then
    ray start --head --port=6379 --num-gpus 8
elif [ "$current_rank" != "0" ]; then
    sleep 20
    ray start --address=${master_addr}:6379 --num-gpus 8 --block
fi

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

if [ "$current_rank" = "0" ]; then
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py\
   --actor-num-nodes 8 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
fi