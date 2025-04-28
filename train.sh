NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --include localhost:1 \
  train.py --deepspeed --config configs/wan.toml
