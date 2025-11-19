#!/bin/bash
# Server 1 ODE: ve/cifar10_ncsnpp (ODE version)

# Set GPU (Server 1 uses GPU 0)
export CUDA_VISIBLE_DEVICES=0
# Prevent TensorFlow from using GPU
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh

# Activate conda environment
conda activate sde_2

cd /home/juhyeong/SDE_Pytorch

python main.py \
  --config /home/juhyeong/SDE_Pytorch/configs/ve/cifar10_ncsnpp_ode.py \
  --workdir /home/juhyeong/SDE_Pytorch/exp/ve/cifar10_ncsnpp_ode \
  --mode eval
