import torch
import numpy as np
import os
import pandas as pd
from scipy.linalg import sqrtm

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRED_DIR = "inference_test_no_tokenizer/pred"
GT_DIR   = "inference_test_no_tokenizer/gt"

NORM_STATS = "norm_stats.pt"

# =========================
# Load normalization
# =========================
norm = torch.load(NORM_STATS, map_location="cpu")
mean = norm["mean"]
std  = norm["std"]

if mean.shape[0] < 181:
    pad = 181 - mean.shape[0]
    mean = torch.cat([mean, torch.zeros(pad)])
    std  = torch.cat([std, torch.ones(pad)])

mean = mean.numpy()
std  = std.numpy()

def normalize_np(x):
    return (x - mean) / std

# =========================
# Metric containers
# =========================
all_l1 = []
all_vel_l1 = []
pred_all = []
gt_all = []

files = sorted(os.listdir(PRED_DIR))

for file in files:

    if not file.endswith("_pred.npy"):
        continue

    base = file.replace("_pred.npy", "")
    gt_file = base + "_gt.npy"

    pred_path = os.path.join(PRED_DIR, file)
    gt_path   = os.path.join(GT_DIR, gt_file)

    if not os.path.exists(gt_path):
        continue

    pred = np.load(pred_path)  # (2,150,181)
    gt   = np.load(gt_path)

    # -------- L1 --------
    l1 = np.mean(np.abs(pred - gt))
    all_l1.append(l1)

    # -------- Velocity L1 --------
    pred_v = pred[:,1:] - pred[:,:-1]
    gt_v   = gt[:,1:] - gt[:,:-1]
    vel_l1 = np.mean(np.abs(pred_v - gt_v))
    all_vel_l1.append(vel_l1)

    # -------- FD raw --------
    pred_all.append(pred.reshape(-1, 181))
    gt_all.append(gt.reshape(-1, 181))

pred_all = np.concatenate(pred_all, axis=0)
gt_all   = np.concatenate(gt_all, axis=0)

# =========================
# Frechet Distance (raw)
# =========================
mu_pred = np.mean(pred_all, axis=0)
mu_gt   = np.mean(gt_all, axis=0)

sigma_pred = np.cov(pred_all, rowvar=False)
sigma_gt   = np.cov(gt_all, rowvar=False)

diff = mu_pred - mu_gt
covmean = sqrtm(sigma_pred @ sigma_gt)

if np.iscomplexobj(covmean):
    covmean = covmean.real

fd_raw = diff @ diff + np.trace(sigma_pred + sigma_gt - 2*covmean)

# =========================
# Final Results
# =========================
results = {
    "L1": np.mean(all_l1),
    "Velocity_L1": np.mean(all_vel_l1),
    "FD_raw": fd_raw
}

df = pd.DataFrame([results])
df.to_csv("evaluation_results.csv", index=False)

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(df)
