import torch
import numpy as np
import os
import functools
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector)
import controllable_generation
from models.ema import ExponentialMovingAverage
import sampling

# Config (simplified from configs/ve/cifar10_ncsnpp_continuous.py)
import ml_collections

def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.sde = 'VESDE'
  training.continuous = True
  training.reduce_mean = True
  training.likelihood_weighting = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.num_channels = 3
  data.centered = False
  data.uniform_dequantization = False

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.embedding_type = 'fourier'
  model.dropout = 0.1
  model.beta_min = 0.1
  model.beta_max = 20.

  config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  return config

def restore_checkpoint(ckpt_path, state, device):
  if not os.path.exists(ckpt_path):
    print(f"No checkpoint found at {ckpt_path}")
    return state
  
  loaded_state = torch.load(ckpt_path, map_location=device)
  state['model'].load_state_dict(loaded_state['model'], strict=False)
  
  if 'ema' in state and 'ema' in loaded_state:
    state['ema'].load_state_dict(loaded_state['ema'])
    state['ema'].copy_to(state['model'].parameters())
    
  return state

def main():
    config = get_config()
    print("Device:", config.device)

    # Initialize SDE
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5

    # Initialize Model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    
    # Load checkpoint if available
    ckpt_path = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        state = dict(model=score_model, ema=ema)
        state = restore_checkpoint(ckpt_path, state, config.device)
        score_model = state['model']
    else:
        print("Checkpoint not found, using random weights.")

    score_model.eval()

    # Compressive Sensing Setup
    # Measurement matrix A: [M, D]
    # Image x: [B, C, H, W] -> [B, D]
    B = 1
    C = config.data.num_channels
    H = config.data.image_size
    W = config.data.image_size
    D = C * H * W
    M = D // 4 # 25% measurements

    # Random Gaussian Matrix
    A = torch.randn(M, D).to(config.device) / np.sqrt(M)
    
    # Ground Truth
    # Create a dummy image or load one
    # Let's create a synthetic image (e.g. circles) or random
    # Random image from prior? No, we want to reconstruct a "real" image.
    # We can generate one using the model first!
    print("Generating ground truth image...")
    # Standard PC sampler
    shape = (B, C, H, W)
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    snr = config.sampling.snr
    n_steps = config.sampling.n_steps_each
    
    sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                          lambda x: x, snr, n_steps=n_steps,
                                          probability_flow=False, continuous=True,
                                          eps=sampling_eps, device=config.device)
    
    x_true, _ = sampling_fn(score_model)
    
    # Measurements
    x_true_flat = x_true.reshape(B, -1)
    y = torch.matmul(x_true_flat, A.t()) # [B, M]
    
    print("Reconstructing...")
    # CS Sampler
    cs_fn = controllable_generation.get_pc_compressive_sensing(
        sde, predictor, corrector, lambda x: x, snr,
        n_steps=n_steps, probability_flow=False, continuous=True,
        denoise=True, eps=sampling_eps, measurement_matrix=A, y=y, scale=1.0, shape=shape
    )
    
    x_recon = cs_fn(score_model)
    
    # Compute error
    mse = torch.mean((x_recon - x_true)**2)
    print(f"Reconstruction MSE: {mse.item()}")
    
    # Save images
    # We need PIL or matplotlib
    import matplotlib.pyplot as plt
    
    # Normalize to [0, 1]
    def normalize(x):
        x = x - x.min()
        x = x / x.max()
        return x.permute(0, 2, 3, 1).cpu().numpy()
        
    img_true = normalize(x_true)[0]
    img_recon = normalize(x_recon)[0]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(img_true)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(img_recon)
    plt.axis('off')
    
    plt.savefig("cs_result.png")
    print("Saved cs_result.png")

if __name__ == "__main__":
    main()
