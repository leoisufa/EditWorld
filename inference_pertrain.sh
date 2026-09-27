#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  inference.py \
  --inference_mode pretrain \
  --control_type cam \
  --txt_file assets/batch_infer_samples.txt \
  --checkpoint_root ckpt \
  --size '480*832' \
  --sp_size 8 \
  --latent_window_size 4 \
  --sink_chunks 1 \
  --recent_chunks 2 \
  --sparse_mem_topk 2 \
  --sparse_mem_offload \
  --sampling_steps 70 \
  --shift 10.0 \
  --guide_scale 5.0 \
  --seed 42 \
  --save_dir outputs/pretrain \
  "$@"
