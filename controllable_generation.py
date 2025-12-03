from models import utils as mutils
import torch
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools


def get_pc_inpainter(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create an image inpainting function that uses PC samplers.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    An inpainting function.
  """
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_inpaint_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def inpaint_update_fn(model, data, mask, x, t):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        masked_data_mean, std = sde.marginal_prob(data, vec_t)
        masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
        x = x * (1. - mask) + masked_data * mask
        x_mean = x * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    return inpaint_update_fn

  projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
  corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

  def pc_inpainter(model, data, mask):
    """Predictor-Corrector (PC) sampler for image inpainting.

    Args:
      model: A score model.
      data: A PyTorch tensor that represents a mini-batch of images to inpaint.
      mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
        and value `0` marks pixels that require inpainting.

    Returns:
      Inpainted (complete) images.
    """
    with torch.no_grad():
      # Initial sample
      x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask)
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t)
        x, x_mean = projector_inpaint_update_fn(model, data, mask, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_inpainter


def get_pc_colorizer(sde, predictor, corrector, inverse_scaler,
                     snr, n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create a image colorization function based on Predictor-Corrector (PC) sampling.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for correctors.
    n_steps: An integer. The number of corrector steps per update of the predictor.
    probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
    continuous: `True` indicates that the score-based model was trained with continuous time steps.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The SDE/ODE will start from `eps` to avoid numerical stabilities.

  Returns: A colorization function.
  """

  # `M` is an orthonormal matrix to decouple image space to a latent space where the gray-scale image
  # occupies a separate channel
  M = torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                   [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                   [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
  # `invM` is the inverse transformation of `M`
  invM = torch.inverse(M)

  # Decouple a gray-scale image with `M`
  def decouple(inputs):
    return torch.einsum('bihw,ij->bjhw', inputs, M.to(inputs.device))

  # The inverse function to `decouple`.
  def couple(inputs):
    return torch.einsum('bihw,ij->bjhw', inputs, invM.to(inputs.device))

  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_colorization_update_fn(update_fn):
    """Modify update functions of predictor & corrector to incorporate information of gray-scale images."""

    def colorization_update_fn(model, gray_scale_img, x, t):
      mask = get_mask(x)
      vec_t = torch.ones(x.shape[0], device=x.device) * t
      x, x_mean = update_fn(x, vec_t, model=model)
      masked_data_mean, std = sde.marginal_prob(decouple(gray_scale_img), vec_t)
      masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
      x = couple(decouple(x) * (1. - mask) + masked_data * mask)
      x_mean = couple(decouple(x) * (1. - mask) + masked_data_mean * mask)
      return x, x_mean

    return colorization_update_fn

  def get_mask(image):
    mask = torch.cat([torch.ones_like(image[:, :1, ...]),
                      torch.zeros_like(image[:, 1:, ...])], dim=1)
    return mask

  predictor_colorize_update_fn = get_colorization_update_fn(predictor_update_fn)
  corrector_colorize_update_fn = get_colorization_update_fn(corrector_update_fn)

  def pc_colorizer(model, gray_scale_img):
    """Colorize gray-scale images using Predictor-Corrector (PC) sampler.

    Args:
      model: A score model.
      gray_scale_img: A minibatch of gray-scale images. Their R,G,B channels have same values.

    Returns:
      Colorized images.
    """
    with torch.no_grad():
      shape = gray_scale_img.shape
      mask = get_mask(gray_scale_img)
      # Initial sample
      x = couple(decouple(gray_scale_img) * mask + \
                 decouple(sde.prior_sampling(shape).to(gray_scale_img.device)
                          * (1. - mask)))
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_colorize_update_fn(model, gray_scale_img, x, t)
        x, x_mean = predictor_colorize_update_fn(model, gray_scale_img, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_colorizer


def get_pc_compressive_sensing(sde, predictor, corrector, inverse_scaler, snr,
                               n_steps=1, probability_flow=False, continuous=False,
                               denoise=True, eps=1e-5, measurement_matrix=None, y=None, scale=1.0, shape=None):
  """Create a compressive sensing function."""

  Args:
    y: A PyTorch tensor of shape [B, M] representing the observed data.
    scale: Scale of the gradient guidance.
    shape: Shape of the image [B, C, H, W].

  Returns:
    A compressive sensing function.
                                          snr=snr,
                                          n_steps=n_steps)

  def get_cs_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""


    def cs_update_fn(model, x, t):
      def wrapper(x_in, t_in):
        # Gradient guidance
        with torch.enable_grad():
          x_in = x_in.detach().requires_grad_(True)
          score = model(x_in, t_in)
        
          # Estimate x_0_hat using Tweedie's formula
          ones = torch.ones_like(x_in)
          if t_in.mean() > 1.0:
            std_t = t_in
            mean_ones = ones
          else:
            mean_ones, std_t = sde.marginal_prob(ones, t_in)
          alpha_t = mean_ones / ones
          
          # Clamp score to prevent explosion
          score = torch.clamp(score, -100.0, 100.0)
          
          x_0_hat = (x_in + std_t[:, None, None, None]**2 * score) / (alpha_t + 1e-8)
          
          # Clamp x_0_hat
          x_0_hat = torch.clamp(x_0_hat, -1.0, 1.0)
          
          x_0_flat = x_0_hat.reshape(x_0_hat.shape[0], -1)
          
          # Apply A
          if measurement_matrix is not None:
               if measurement_matrix.dim() == 2:
                   Ax = torch.matmul(x_0_flat, measurement_matrix.t())
               else:
                   Ax = torch.bmm(x_0_flat.unsqueeze(1), measurement_matrix.transpose(1, 2)).squeeze(1)
          else:
               raise ValueError("Measurement matrix required")
               
          # Loss = || y - Ax ||^2
          residual = y - Ax
          norm = torch.norm(residual, dim=1)
          loss = torch.sum(norm**2)
          
          # Grad
          grad = torch.autograd.grad(loss, x_in)[0]
          
          # Check for nans
          if torch.isnan(grad).any():
              # print("NaN in gradient!")
              grad = torch.zeros_like(grad)
          
          # Robustness: Clamp gradient
          # grad = torch.clamp(grad, -0.1, 0.1)
          
          return score - scale * grad

      wrapper.eval = lambda: None
      return update_fn(x, t, model=wrapper)

    return cs_update_fn

  predictor_cs_update_fn = get_cs_update_fn(predictor_update_fn)
  corrector_cs_update_fn = get_cs_update_fn(corrector_update_fn)

  def pc_cs(model):
    """Predictor-Corrector (PC) sampler for compressive sensing.

    Args:
      model: A score model.
    
    Returns:
      Reconstructed images.
    """
    with torch.no_grad():
      # Initial sample
      if shape is None:
        raise ValueError("Shape must be provided for compressive sensing.")
        
      x = sde.prior_sampling(shape).to(y.device)
      print(f"DEBUG: sde.T={sde.T}, eps={eps}, sde.N={sde.N}")
      timesteps = torch.linspace(sde.T, eps, sde.N)
      print(f"DEBUG: timesteps[0]={timesteps[0]}, timesteps[-1]={timesteps[-1]}")
      
      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(x.shape[0], device=x.device) * t
        # Corrector step
        x, x_mean = corrector_cs_update_fn(model, x, vec_t)
        # Predictor step
        x, x_mean = predictor_cs_update_fn(model, x, vec_t)

      return inverse_scaler(x_mean if denoise else x)
  
  return pc_cs
