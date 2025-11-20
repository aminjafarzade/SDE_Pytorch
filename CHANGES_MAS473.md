# MAS473 Changes Summary

This document summarizes all modifications made to the upstream
`score_sde_pytorch` repository to make it easier to run in our
environment (local .venv, remote GPU server), stabilize evaluation, and
speed up iteration.

## Overview

- Simplified dependencies to install cleanly in a `.venv` on Apple
  Silicon and Linux servers.
- Made JAX optional (this is the PyTorch repo) to avoid unnecessary
  requirements.
- Added safe CPU fallbacks for CUDA-only custom ops to prevent
  segmentation faults when CUDA is unavailable.
- Hardened evaluation: TLS issues with TF‑Hub, local TF‑Hub cache
  support, and a knob to cap evaluation to a small number of batches.
- Added developer notes and pinned evaluation assets in-repo for
  reproducibility.

## Dependency Changes

File: `requirements.txt`
- Updated to versions compatible with TensorFlow 2.13 and TF‑GAN 2.0:
  - `tensorflow==2.13.0`
  - `tensorflow-gan==2.0.0`
  - `tensorflow-probability==0.21.0`
  - `tensorflow_datasets==4.9.3`
  - `tensorflow-addons==0.23.0`
  - `tensorboard==2.13.0`
  - `absl-py>=1.4.0`
  - `scipy>=1.9`
  - Kept: `ml-collections`, `tensorflow_io`, `torch>=1.7.0`,
    `torchvision`, `ninja`
- Removed hard dependency on JAX/JAXLIB; the code now works without
  them.

## CUDA Extension Fallbacks

Files: `op/fused_act.py`, `op/upfirdn2d.py`
- Load custom CUDA kernels only when a CUDA device is available;
  otherwise fall back to native PyTorch implementations. This avoids
  compilation on CPU-only systems and prevents crashes when `CUDA_HOME`
  is not set.

## JAX Optionality (PyTorch‑only)

Files: `datasets.py`, `evaluation.py`
- Import JAX lazily and provide device-count fallbacks if JAX is not
  installed.
- `evaluation.run_inception_distributed` now runs a non-JAX path if no
  JAX devices are detected.

## Evaluation Robustness

File: `evaluation.py`
- Added TLS handling for TF‑Hub downloads using `certifi` and a safe
  retry without certificate verification when needed.
- Support for loading a local TF‑Hub Inception model from
  `assets/tfhub_modules/tfgan_eval_inception/` to avoid network issues.
- Added `compute_fid_stats.py` so dataset statistics can be regenerated
  remotely without reaching out to third-party storage.
- Evaluation now tolerates dataset stats that contain only `mu`/`sigma`
  (e.g. OpenAI’s `VIRTUAL_lsun_bedroom256.npz`); FID is computed from
  moments, and KID is skipped in that case.
- Added `assets/stats/lsun_bedroom_256_stats.npz` (OpenAI ADM
  reference mu/sigma) so LSUN bedroom evals work without recomputing
  activations.

## Evaluation Runtime Control

Files: `configs/default_cifar10_configs.py`,
`configs/default_celeba_configs.py`, `configs/default_lsun_configs.py`,
`run_lib.py`
- Added `eval.max_batches` (default `-1` for unlimited) to cap the
  number of batches processed during loss/BPD, and to cap sampling
  rounds. This enables 1-batch sanity checks and prevents long 12‑hour
  eval loops during smoke tests.

## Developer Notes

File: `DEV_NOTES.md`
- Documented remote GPU workflow with `scripts/remote_run.sh`, how to
  keep everything in `.venv`, and how to run short smoke tests locally
  and full runs remotely.
- Included instructions for FID/KID/IS evaluation.
- Added guidance on reusing the cached `/dev/shm/proj_sync_XXXXXX`
  workspace recorded in `.remote_workdir`, plus the `REMOTE_SKIP_SYNC=1`
  fast path to skip re-rsyncing once the server copy is up to date.

## Assets Added (for Evaluation)

- `assets/stats/cifar10_stats.npz` — official CIFAR‑10 statistics
  (downloaded per README link) for FID/KID/IS.
- `assets/stats/celeba_stats.npz` — CelebA (64px) stats.
- `assets/stats/celebahq_256_stats.npz` — CelebA‑HQ 256px stats.
- `assets/stats/lsun_church_outdoor_256_stats.npz` — LSUN church 256px
  stats.
- `assets/tfhub_modules/tfgan_eval_inception/` — local TF‑Hub Inception
  module so evaluation does not depend on network availability.

## Notes and Rationale

- These changes minimize environment drift, remove an unnecessary JAX
  requirement, and make evaluation predictable in both offline and
  limited-network environments.
- The new `eval.max_batches` knob is off by default to preserve upstream
  behavior, and can be used from the CLI for quick validations.
- Importing PyTorch before TensorFlow in `run_lib.py` fixes a CUDA
  initialization order issue we hit on Linux servers (TensorFlow's stub
  libcuda would crash the process otherwise).
- `compute_fid_stats.py` provides a reproducible way to recompute
  dataset statistics entirely within the repo, so evaluation metrics no
  longer depend on external downloads at runtime.
