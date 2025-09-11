cd slime
python slime/tools/convert_torch_dist_to_hf.py \
  --input-dir megatron_ckpt_path \
  --output-dir huggingface_ckpt_path \
  --origin-hf-dir original_huggingface_model_path