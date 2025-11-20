## MAS473 Dev Notes

This fork adds a couple of quality-of-life improvements for the project course. The
goal is to always work inside `.venv` locally while using the remote GPU server for
heavy jobs and benchmarking.

### Remote GPU runs

1. Make sure your local `.venv` has the dependencies installed:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure the remote host and virtualenv once. The defaults live in
   `scripts/remote_common.sh`, so either edit `DEFAULT_HOST`/`DEFAULT_ENV_ACTIVATE`
   there or override them per-command:
   ```bash
   HOST=my-gpu-box ENV_ACTIVATE='source ~/projects/score/.venv/bin/activate' \
     scripts/remote_run.sh -- \
     python main.py --config configs/ve/cifar10_ncsnpp.py --mode train \
     --workdir workdir/cifar10_exp --config.training.batch_size=128
   ```
   * `scripts/remote_run.sh` rsyncs the repo to a persistent remote dir (recorded
     in `.remote_workdir` after the first run), bootstraps a Linux `.venv` there
     on demand (`python3 -m venv .venv && pip install -r score_sde_pytorch/requirements.txt`),
     prints `nvidia-smi`, and runs your command **inside the cached directory**
     (e.g. `/dev/shm/proj_sync_6dmyXk` on the GPU box). Set `REMOTE_SKIP_SYNC=1`
     to reuse the existing checkout without syncing again; every subsequent run
     will just `cd` into the same remote folder unless you delete `.remote_workdir`.
     Override `ENV_ACTIVATE` or `REMOTE_BOOTSTRAP_VENV=0` if you already manage
     the environment.
   * `scripts/remote_debugpy.sh` works the same way but exposes a debugpy port.
3. Outputs (checkpoints, samples, logs) are pulled back automatically when the remote
   command finishes.

### Python environment (no Conda required)

The repo now installs cleanly on Apple silicon using the built-in `.venv`; Conda
is not required. Simply run `python3 -m venv .venv && source .venv/bin/activate`
before installing `requirements.txt`. The custom CUDA extensions (fused activations,
upfirdn2d) automatically fall back to PyTorch implementations when CUDA is not
available, eliminating earlier segmentation faults on CPU-only machines.

### Running training/eval safely

To smoke-test a config without a long run, override the training knobs:

```bash
python main.py --config configs/ve/cifar10_ncsnpp.py --mode train \
  --workdir workdir/cifar10_debug \
  --config.training.batch_size=4 \
  --config.training.n_iters=1 \
  --config.training.snapshot_sampling=False
```

This verifies dataloaders, checkpointing, and logging in a few seconds.

### Benchmark metrics (FID / KID / IS)

1. Download the dataset stats referenced in the README (e.g.
   `assets/stats/cifar10_stats.npz` from the provided Google Drive link).
2. Enable sampling/eval flags either inside the config or from the CLI. Example:
   ```bash
   python main.py --config configs/ve/cifar10_ncsnpp.py --mode eval \
     --workdir /path/to/training_run \
     --config.eval.enable_sampling=True \
     --config.eval.enable_loss=True \
     --config.eval.batch_size=512
   ```
3. Evaluation writes metrics (FID, KID, Inception score, BPD if enabled) under
   `workdir/<eval_folder>` along with generated samples for manual inspection.
   Make sure the checkpoints referenced by `config.eval.begin_ckpt`/`end_ckpt` exist.

With these steps in place you can iterate locally using `.venv`, offload the heavy
jobs to the GPU box with the helper scripts, and rely on the updated requirements
to compute benchmark metrics without additional manual tweaks.
