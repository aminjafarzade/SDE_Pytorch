#!/usr/bin/env bash
set -euo pipefail

# Base repo dir (change if needed)
BASE="/lustre/home/aminjafarzade/temp/score_sde_pytorch"

CONFIG_DIR="$BASE/configs/subvp"
EXP_ROOT="$BASE/exp/subvp"

for cfg in "$CONFIG_DIR"/*.py; do
    name="$(basename "$cfg" .py)"          # e.g. cifar10_ddpmpp_continuous
    workdir="$EXP_ROOT/$name"             # e.g. exp/subvp/cifar10_ddpmpp_continuous

    echo "======================================================="
    echo "Running eval for config: $cfg"
    echo "Workdir: $workdir"
    echo "======================================================="

    # Run evaluation and log output
    python "$BASE/main.py" \
        --config "$cfg" \
        --workdir "$workdir" \
        --mode eval \
        
done

echo "All evaluations finished."
