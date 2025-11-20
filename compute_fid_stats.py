#!/usr/bin/env python3
# coding=utf-8
"""Compute dataset Inception statistics (FID/KID/IS) for Score SDE configs."""

import io
import os
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import datasets
import evaluation

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration to reuse for dataset + image size.",
  lock_config=True)
flags.DEFINE_enum("split", "eval", ["train", "eval"],
                  "Which dataset split to use (train_ds vs eval_ds).")
flags.DEFINE_integer("batch_size", 0,
                     "Override config.eval.batch_size for stats computation.")
flags.DEFINE_integer("max_images", -1,
                     "Maximum number of images to process. -1 means use the full split.")
flags.DEFINE_bool("inceptionv3", False,
                  "Use InceptionV3 (set True for 299px datasets).")
flags.DEFINE_string(
  "output", None,
  "Path to the output .npz (defaults to assets/stats/<dataset>_stats.npz).")
flags.DEFINE_bool("skip_logits", False,
                  "If True, skip saving logits (saves only pool_3 activations).")
flags.DEFINE_string(
  "tfds_data_dir", None,
  "Optional TFDS data directory to reuse downloads/cache.")
flags.DEFINE_string(
  "tfds_manual_dir", None,
  "Optional TFDS manual directory for datasets that require manual downloads.")


def _default_output_path(config) -> str:
  """Match evaluation.load_dataset_stats default locations."""
  dataset = config.data.dataset.upper()
  stats_dir = os.path.join("assets", "stats")
  if dataset == "CIFAR10":
    filename = "cifar10_stats.npz"
  elif dataset == "CELEBA":
    filename = "celeba_stats.npz"
  elif dataset == "LSUN":
    filename = f"lsun_{config.data.category}_{config.data.image_size}_stats.npz"
  else:
    filename = f"{config.data.dataset.lower()}_{FLAGS.split}_stats.npz"
  return os.path.join(stats_dir, filename)


def _maybe_num_examples(dataset_builder, split_name: str) -> Optional[int]:
  info = getattr(dataset_builder, "info", None)
  if info is None:
    return None
  split = info.splits.get(split_name)
  if split is None:
    return None
  return split.num_examples


def main(argv):
  del argv
  if FLAGS.config is None:
    raise ValueError("--config must point to a config file.")

  # Copy config so we can tweak batch size without mutating the flag value.
  config = FLAGS.config.copy_and_resolve_references()
  if FLAGS.batch_size > 0:
    config.eval.batch_size = FLAGS.batch_size

  split_name = "train" if FLAGS.split == "train" else "eval"
  logging.info("Loading %s split with batch size %d", split_name,
               config.eval.batch_size)
  download_config = None
  if FLAGS.tfds_manual_dir:
    download_config = tfds.download.DownloadConfig(
      manual_dir=FLAGS.tfds_manual_dir)
  train_ds, eval_ds, dataset_builder = datasets.get_dataset(
    config,
    evaluation=True,
    drop_remainder=False,
    data_dir=FLAGS.tfds_data_dir,
    download_config=download_config)
  ds = train_ds if FLAGS.split == "train" else eval_ds
  total_images_hint = _maybe_num_examples(dataset_builder, split_name)
  if total_images_hint:
    logging.info("Dataset split reports %d images", total_images_hint)

  inception_model = evaluation.get_inception_model(inceptionv3=FLAGS.inceptionv3)
  pools = []
  logits = []
  processed = 0

  for batch_id, batch in enumerate(tfds.as_numpy(ds)):
    images = batch["image"]
    images = np.clip(images * 255., 0, 255).astype(np.uint8)
    latents = evaluation.run_inception_distributed(
      images, inception_model, num_batches=1, inceptionv3=FLAGS.inceptionv3)
    pools.append(latents["pool_3"])
    if not FLAGS.inceptionv3 and not FLAGS.skip_logits:
      logits.append(latents["logits"])
    processed += images.shape[0]
    if batch_id % 10 == 0:
      logging.info("Processed %d images so far", processed)
    if FLAGS.max_images > 0 and processed >= FLAGS.max_images:
      break

  if processed == 0:
    raise RuntimeError("No images were processed; check dataset + split settings.")
  pools_arr = np.concatenate(pools, axis=0)
  pools_arr = pools_arr[:processed]
  if logits:
    logits_arr = np.concatenate(logits, axis=0)[:processed]
  else:
    logits_arr = None

  output_path = FLAGS.output or _default_output_path(config)
  output_dir = os.path.dirname(output_path)
  if output_dir:
    tf.io.gfile.makedirs(output_dir)
  logging.info("Writing %s (images=%d)", output_path, processed)
  buf = io.BytesIO()
  if logits_arr is not None:
    np.savez_compressed(buf, pool_3=pools_arr, logits=logits_arr)
  else:
    np.savez_compressed(buf, pool_3=pools_arr)
  with tf.io.gfile.GFile(output_path, "wb") as fout:
    fout.write(buf.getvalue())
  logging.info("Done.")


if __name__ == "__main__":
  app.run(main)
