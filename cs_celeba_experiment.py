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
import ml_collections
import matplotlib.pyplot as plt

def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.sde = 'vesde'
  training.continuous = True
  training.reduce_mean = False
  training.likelihood_weighting = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CelebAHQ'
  data.image_size = 256
  data.num_channels = 3
  data.centered = False
  data.uniform_dequantization = False

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnpp'
  model.sigma_max = 348
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

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
    
    # Load checkpoint
    ckpt_path = "/mnt/tmp/mas473_project_1/workdir/ve/celebahq_256_ncsnpp_continuous/checkpoints/checkpoint_48.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        state = dict(model=score_model, ema=ema)
        state = restore_checkpoint(ckpt_path, state, config.device)
        score_model = state['model']
    else:
        print(f"Checkpoint not found at {ckpt_path}, using random weights.")

    score_model.eval()

    # Compressive Sensing Setup
    B = 1
    C = config.data.num_channels
    H = config.data.image_size
    W = config.data.image_size
    D = C * H * W
    M = D // 16 # 6.25% measurements (aggressive compression for speed/demo)
    
    # Random Gaussian Matrix
    # For 256x256, D = 3*256*256 = 196608. M ~ 12288.
    # Matrix A is [M, D] ~ [12288, 196608]. This is huge (2.4e9 elements).
    # 2.4e9 floats * 4 bytes ~ 9.6 GB. It might fit in 3080 (10GB) but it's tight.
    # Let's use a smaller image size for CS or a structured matrix?
    # Or just use a very small M?
    # Or maybe we should use a masking operator (inpainting is a special case of CS)?
    # But the user asked for "compressive sensing".
    # Let's try to construct A on the fly or use a smaller problem?
    # No, I must use the model which is 256x256.
    # I can use a measurement operator that is memory efficient, e.g. subsampled Fourier or just random mask (inpainting).
    # But standard CS uses random Gaussian.
    # Let's try a very small M, e.g. D // 100.
    # Or better, let's use a structured random matrix (e.g. random convolution) to save memory?
    # My implementation `get_pc_compressive_sensing` supports `measurement_matrix` as a tensor.
    # If I pass a huge tensor, it might OOM.
    # Let's try with M = D // 32.
    # A [6144, 196608] ~ 1.2e9 elements ~ 4.8 GB. This should fit.
    
    M = D // 32
    print(f"Creating measurement matrix [{M}, {D}]...")
    # Create on CPU first to save GPU memory, then move chunks?
    # Or just create on GPU.
    try:
        A = torch.randn(M, D, device=config.device) / np.sqrt(M)
    except RuntimeError as e:
        print(f"OOM creating matrix: {e}")
        print("Switching to CPU for matrix (might be slow)")
        A = torch.randn(M, D) / np.sqrt(M)
        # We need A on device for matmul in loop.
        # If it doesn't fit, we are in trouble with this implementation.
        # Let's hope it fits.
    
    # Ground Truth
    print("Generating ground truth image...")
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
    if A.device != x_true.device:
        A = A.to(x_true.device)
        
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
    
    plt.savefig("cs_celeba_result.png")
    print("Saved cs_celeba_result.png")

if __name__ == "__main__":
    main()
