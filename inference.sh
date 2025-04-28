NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --include localhost:1 \
  inference.py --deepspeed --config /path/to/config.toml \
  --dataset_config /path/to/dataset.toml \
  --ckpt /path/to/ckpt_dir \
  --output_dir /path/to/output \