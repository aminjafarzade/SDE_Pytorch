#!/bin/bash
# Server 3 ODE: vp/cifar10_ddpmpp_continuous (ODE version)

# Set GPU (Server 3 uses GPU 2)
export CUDA_VISIBLE_DEVICES=2
# Prevent TensorFlow from using GPU
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh

# Activate conda environment
conda activate sde_2

cd /home/juhyeong/SDE_Pytorch

python main.py \
  --config /home/juhyeong/SDE_Pytorch/configs/vp/cifar10_ddpmpp_continuous_ode.py \
  --workdir /home/juhyeong/SDE_Pytorch/exp/vp/cifar10_ddpmpp_continuous_ode \
  --mode eval

