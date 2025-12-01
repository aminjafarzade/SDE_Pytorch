# Score SDE Pytorch – Project Version

This repo is a **homework-friendly fork** of `score_sde_pytorch`:

- Uses **PyTorch 2.3 + CUDA 12.1** (L40S / Ada-compatible).
- Keeps the **original TensorFlow + TF-GAN evaluation pipeline** (IS/FID/KID, TFDS).
- **Removes all custom C++/CUDA extensions** and `torchvision` so it runs cleanly on clusters.
- Adds a **pure PyTorch `upfirdn2d`** implementation so `model.fir = True` still works.

The goal is to reproduce and experiment with score-based / diffusion models (NCSN++, DDPM++, etc.) without fighting the environment for hours.

---

## 1. Environment Setup

### 1.1. Create the conda env

From the repo root:

```bash
# (Optional) remove old env
conda env remove -n sde_2

# Create env from the pinned file
conda env create -f environment.yml
conda activate sde_2
The environment.yml sets up:

Python 3.8

MKL (pinned <2024.1)

PyTorch 2.3.* + pytorch-cuda=12.1

Scientific + Jupyter stack

Logging / config utilities

A modern typing-extensions (needed for PyTorch 2.x)

Note: environment.yml does not install TensorFlow – we add that next with --no-deps to avoid version conflicts.

1.2. Install TensorFlow + TF-GAN stack (no deps)
Still inside the sde_2 env:

bash
Copy code
conda activate sde_2

pip install \
  "tensorflow==2.4.0" \
  "tensorflow-estimator==2.4.0" \
  "tensorflow-gan==2.0.0" \
  "tensorflow-datasets==3.1.0" \
  "tensorflow-hub==0.16.0" \
  "tensorflow-probability==0.12.2" \
  "tensorflow-io==0.34.0" \
  "tensorflow-io-gcs-filesystem==0.34.0" \
  "tensorflow-metadata==1.12.0" \
  "tensorflow-addons==0.12.0" \
  --no-deps
The --no-deps flag is important:

Keeps typing-extensions at a version compatible with PyTorch 2.3.

Avoids pip trying to downgrade things to satisfy old TF 2.4 metadata.

## Compressive Sensing Evaluation

To run the full Compressive Sensing evaluation using the `controllable_generation` module:

```bash
# Example for CelebA-HQ 256 (Synthetic Evaluation)
python main.py --mode=eval_cs \
  --config=configs/ve/celebahq_256_ncsnpp_continuous.py \
  --workdir=workdir/ve/celebahq_256_ncsnpp_continuous \
  --eval_folder=eval_cs_celeba \
  --config.eval.begin_ckpt=1 \
  --config.eval.end_ckpt=48 \
  --config.eval.batch_size=1
```

This will:
1. Load the model and checkpoints from `workdir`.
2. Generate synthetic ground truth data (sampling from the model) if the dataset is not available.
3. Generate random Gaussian measurements (32x compression for CelebA-HQ).
4. Reconstruct images using the PC sampler with gradient guidance.
5. Save reconstruction samples and MSE/PSNR statistics to `workdir/eval_cs_celeba`.

**Stability & Results:**
- **Numerical Instability:** High noise levels in VESDE ($\sigma_{max}=348$) can cause gradient explosions, leading to `nan` values.
- **Fixes Implemented:** We have implemented robust gradient clamping (score, $x_0$, and grad) and tuned hyperparameters (`snr=0.05`, `scale=0.5`) to ensure stability.
- **Expected Results:** With these settings, you should expect **MSE $\approx$ 0.02** and **PSNR $\approx$ 23 dB** for CelebA-HQ 256.