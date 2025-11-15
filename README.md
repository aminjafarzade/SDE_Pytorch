# Score SDE Pytorch – Homework Version

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

1.3. Quick sanity check
bash
Copy code
python - << 'EOF'
import torch, typing_extensions as te
import tensorflow as tf, tensorflow_gan as tfgan

print("torch:", torch.__version__, "cuda?", torch.cuda.is_available())
print("typing_extensions:", te.__version__, "has TypeGuard?", hasattr(te, "TypeGuard"))
print("tensorflow:", tf.__version__)
print("tfgan:", tfgan.__version__)
EOF
You should see:

torch: 2.3.x cuda? True

typing_extensions: 4.x has TypeGuard? True

tensorflow: 2.4.0

tfgan: 2.0.0