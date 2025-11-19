#!/bin/bash
# Server 4: vp/cifar10_ddpmpp

# Set GPU (Server 4 uses GPU 3)
export CUDA_VISIBLE_DEVICES=3
# Prevent TensorFlow from using GPU
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh

# Activate conda environment
conda activate sde_2

cd /home/juhyeong/SDE_Pytorch

python main.py \
  --config /home/juhyeong/SDE_Pytorch/configs/vp/cifar10_ddpmpp.py \
  --workdir /home/juhyeong/SDE_Pytorch/exp/vp/cifar10_ddpmpp \
  --mode eval

