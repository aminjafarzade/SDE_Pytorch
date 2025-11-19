# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import os
# Save CUDA_VISIBLE_DEVICES for PyTorch before TensorFlow import
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import tensorflow as tf
# Force TensorFlow to use CPU only (don't interfere with PyTorch GPU)
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass
# Restore CUDA_VISIBLE_DEVICES for PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
import sys

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  # Create the working directory (for both train and eval)
  tf.io.gfile.makedirs(FLAGS.workdir)

  # ----------------- Logging setup: file + console -----------------
  log_file = os.path.join(FLAGS.workdir, 'stdout.txt')

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  # Avoid duplicate handlers if main is called multiple times
  logger.handlers.clear()

  formatter = logging.Formatter(
      '%(levelname)s - %(filename)s - %(asctime)s - %(message)s'
  )

  # File handler
  file_stream = open(log_file, 'w')
  file_handler = logging.StreamHandler(file_stream)
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)

  # Console handler
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  # --------------------------------------------------------------

  if FLAGS.mode == "train":
    logging.info("Starting in TRAIN mode")
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    logging.info("Starting in EVAL mode")
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")



if __name__ == "__main__":
  app.run(main)
