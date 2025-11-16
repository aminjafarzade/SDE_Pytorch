import numpy as np

# ---- CHANGE THIS TO YOUR CHECKPOINT ----
ckpt_path = "/lustre/home/aminjafarzade/temp/score_sde_pytorch/exp/subvp/cifar10_ddpmpp_continuous/eval/ckpt_15_inception_metrics.npz"

# Load the npz file
d = np.load(ckpt_path)

# Print available keys first
print("Available keys:", d.files)

# Helper to safely get metric
def get_metric(d, names):
    for name in names:
        if name in d:
            return float(d[name])
    return None

fid = get_metric(d, ["FID", "fid"])
kid = get_metric(d, ["KID", "kid"])
iscore = get_metric(d, ["IS", "is"])

print("\n=== Metrics for:", ckpt_path, "===")
print("FID :", fid)
print("KID :", kid)
print("IS  :", iscore)
