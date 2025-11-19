import argparse
import os
import sys
import subprocess
from typing import List


def build_ddim_argv(args: argparse.Namespace) -> List[str]:
    """Build a fake argv list for the original DDIM repo's main.py.

    We always run in sampling mode from this integration script.
    """
    ddim_args = [
        "ddim_main",
        "--config",
        args.config,
        "--exp",
        args.exp,
        "--doc",
        args.doc,
        "--sample",
        "--ni",  # no interaction / overwrite prompts
        "--sample_type",
        args.sample_type,
        "--skip_type",
        args.skip_type,
        "--timesteps",
        str(args.timesteps),
        "--eta",
        str(args.eta),
        "-i",
        args.image_folder,
    ]

    if args.fid:
        ddim_args.append("--fid")
    if args.interpolation:
        ddim_args.append("--interpolation")
    if args.sequence:
        ddim_args.append("--sequence")
    if args.use_pretrained:
        ddim_args.append("--use_pretrained")

    return ddim_args


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Thin wrapper to run the original DDIM codebase from within the "
            "SDE_Pytorch project for sampling / experiments."
        )
    )
    parser.add_argument(
        "--ddim_root",
        type=str,
        default="/home/juhyeong/SDE_Pytorch/ddim_external",
        help="Path to the (vendored) DDIM repository (contains main.py, configs/, etc.).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cifar10.yml",
        help="Name of the DDIM config yaml (relative to DDIM configs/ directory).",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="/home/juhyeong/SDE_Pytorch/ddim_external/exp",
        help="DDIM experiment root directory (same semantics as original repo's --exp).",
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help=(
            "Experiment name (same semantics as original repo's --doc). "
            "Used to locate checkpoints under exp/logs/<doc>."
        ),
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="ddim_samples",
        help=(
            "Subfolder name for saving sampled images (under exp/image_samples/). "
            "Passed through to the original DDIM code as -i/--image_folder."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Total number of samples to generate in the DDIM FID loop.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Number of DDIM time steps to use for sampling.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0.0 = pure DDIM, 1.0 ≈ DDPM-style sampling).",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        choices=["generalized", "ddpm_noisy"],
        help="Sampling approach used inside the DDIM code.",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        choices=["uniform", "quad"],
        help="Schedule for picking DDIM time indices.",
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="If set, run the FID sampling path in the DDIM code.",
    )
    parser.add_argument(
        "--interpolation",
        action="store_true",
        help="If set, run the interpolation sampling path in the DDIM code.",
    )
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="If set, save the full denoising sequence (DDIM code path).",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use the pretrained DDIM/DDPM checkpoint logic from the original repo.",
    )

    args = parser.parse_args()

    # Resolve and validate DDIM root.
    ddim_root = os.path.abspath(args.ddim_root)
    if not os.path.isdir(ddim_root):
        raise SystemExit(f"DDIM root directory not found: {ddim_root}")

    # Build the argument list for the underlying DDIM main.py.
    ddim_argv = build_ddim_argv(args)
    # Pass through the requested total number of samples.
    ddim_argv.extend(["--total_n_samples", str(args.num_samples)])

    # We call the original DDIM entry point via a subprocess to mimic
    # `cd $ddim_root && python main.py ...` exactly. This avoids any issues
    # with relative imports inside the original repository.
    cmd = [sys.executable, "main.py"] + ddim_argv[1:]  # drop fake program name

    completed = subprocess.call(cmd, cwd=ddim_root)
    raise SystemExit(completed)


if __name__ == "__main__":
    main()


