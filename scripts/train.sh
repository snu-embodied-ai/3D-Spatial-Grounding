CUDA_VISIBLE_DEVICE=0,1 torchrun --nnodes=1 --nproc-per-node=2 scripts/run.py \
    --run_type train \
    --config cfg/train.yaml \
    --model_config cfg/Spatial3D.yaml \
    --data_config cfg/dataset.yaml \
    