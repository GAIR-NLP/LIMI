cd slime
source scripts/models/glm4.5-355B-A32B.sh
torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=2 --node-rank ${PET_NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint original_huggingface_model_path \
   --save megatron_model_path 