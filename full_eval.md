REMOTE_SKIP_SYNC=1 ENTRY="python" ./scripts/remote_run.sh -- \
  score_sde_pytorch/main.py \
  --config score_sde_pytorch/configs/subvp/cifar10_ddpmpp_continuous.py \
  --mode eval \
  --workdir score_sde_pytorch/workdir/subvp_cifar10_pretrained/cifar10_ddpmpp_continuous \
  --config.eval.begin_ckpt=15 \
  --config.eval.end_ckpt=15 \
  --config.eval.enable_sampling=True \
  --config.eval.enable_loss=False \
  --config.eval.batch_size=512 \
  --config.eval.num_samples=50000
