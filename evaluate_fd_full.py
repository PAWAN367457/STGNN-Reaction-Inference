import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.linalg import sqrtm

from stgnn_dataset import STGNNReactionDataset
from stgnn_model import STGNNModel
from models import FaceMotionTokenizerV2


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DMM_META = "rewritten_metadata/video_3dmm_features_metadata.csv"
AUDIO_META = "rewritten_metadata/audio_features_metadata.csv"

STGNN_CKPT = "checkpoints/stgnn_no_tokenizer_best.pt"
TOKENIZER_CKPT = "best_tokenizer_fsq_256_v2.pt"
NORM_STATS = "norm_stats.pt"

BATCH_SIZE = 8


# =========================
# LOAD NORMALIZATION
# =========================
norm = torch.load(NORM_STATS, map_location="cpu")
mean = norm["mean"]
std  = norm["std"]

if mean.shape[0] < 181:
    pad = 181 - mean.shape[0]
    mean = torch.cat([mean, torch.zeros(pad)])
    std  = torch.cat([std, torch.ones(pad)])

mean = mean.to(DEVICE)
std  = std.to(DEVICE)

def normalize(x):
    return (x - mean) / std


# =========================
# LOAD DATASET
# =========================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    speaker = torch.stack([b["speaker_past"] for b in batch])
    listener1 = torch.stack([b["listener_future"][0] for b in batch])
    listener2 = torch.stack([b["listener_future"][1] for b in batch])

    motion = torch.stack([speaker, listener1, listener2], dim=1)
    audio = torch.stack([b["audio_past"] for b in batch])

    return {
        "motion": motion,
        "audio": audio
    }

dataset = STGNNReactionDataset(
    dmm_metadata_csv=DMM_META,
    audio_metadata_csv=AUDIO_META,
    split="test"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("Test clips:", len(dataset))


# =========================
# LOAD MODELS
# =========================
model = STGNNModel().to(DEVICE)
model.load_state_dict(torch.load(STGNN_CKPT, map_location=DEVICE))
model.eval()

tokenizer = FaceMotionTokenizerV2(
    input_dim=181,
    down_t=3,
    stride_t=2,
    quantizer="fsq",
    embed=256,
    levels=[8,5,5]
).to(DEVICE)

ckpt = torch.load(TOKENIZER_CKPT, map_location=DEVICE)
tokenizer.load_state_dict({k.replace("_orig_mod.", ""): v for k,v in ckpt.items()})
tokenizer.eval()

print("Models loaded")


# =========================
# METRICS
# =========================
l1_total = 0
vel_total = 0
n_batches = 0

gt_l1_feat = []
gt_l2_feat = []
pred_l1_feat = []
pred_l2_feat = []


def velocity_loss(pred, gt):
    pred_v = pred[:, :, 1:] - pred[:, :, :-1]
    gt_v   = gt[:, :, 1:] - gt[:, :, :-1]
    return torch.mean(torch.abs(pred_v - gt_v))


# =========================
# INFERENCE + FEATURE EXTRACTION
# =========================
with torch.no_grad():
    for batch in tqdm(loader):

        if batch is None:
            continue

        motion = batch["motion"].to(DEVICE)
        audio  = batch["audio"].to(DEVICE)

        motion_norm = normalize(motion)

        # Forward
        pred = model(motion_norm, audio)

        pred_l = pred[:,1:]
        gt_l   = motion_norm[:,1:]

        # Metrics
        l1_total += torch.mean(torch.abs(pred_l - gt_l)).item()
        vel_total += velocity_loss(pred_l, gt_l).item()
        n_batches += 1

        # ---- Tokenizer feature extraction ----
        for b in range(pred_l.shape[0]):

            # Listener 1
            gt_tok, _ = tokenizer(gt_l[b,0].unsqueeze(0))
            pr_tok, _ = tokenizer(pred_l[b,0].unsqueeze(0))

            gt_l1_feat.append(gt_tok.mean(dim=1).cpu().numpy())
            pred_l1_feat.append(pr_tok.mean(dim=1).cpu().numpy())

            # Listener 2
            gt_tok, _ = tokenizer(gt_l[b,1].unsqueeze(0))
            pr_tok, _ = tokenizer(pred_l[b,1].unsqueeze(0))

            gt_l2_feat.append(gt_tok.mean(dim=1).cpu().numpy())
            pred_l2_feat.append(pr_tok.mean(dim=1).cpu().numpy())


# =========================
# STACK FEATURES
# =========================
gt_l1_feat = np.concatenate(gt_l1_feat, axis=0)
pred_l1_feat = np.concatenate(pred_l1_feat, axis=0)

gt_l2_feat = np.concatenate(gt_l2_feat, axis=0)
pred_l2_feat = np.concatenate(pred_l2_feat, axis=0)


# =========================
# FD FUNCTION
# =========================
def compute_fd(mu1, sigma1, mu2, sigma2):
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*covmean)

def fd_between(X, Y):
    mu1, sigma1 = X.mean(axis=0), np.cov(X, rowvar=False)
    mu2, sigma2 = Y.mean(axis=0), np.cov(Y, rowvar=False)
    return compute_fd(mu1, sigma1, mu2, sigma2)


# =========================
# COMPUTE FDs
# =========================
fd_l1 = fd_between(gt_l1_feat, pred_l1_feat)
fd_l2 = fd_between(gt_l2_feat, pred_l2_feat)

fd_pool = fd_between(
    np.concatenate([gt_l1_feat, gt_l2_feat]),
    np.concatenate([pred_l1_feat, pred_l2_feat])
)


print("\n===== FINAL RESULTS =====")
print("L1:", l1_total / n_batches)
print("Velocity L1:", vel_total / n_batches)
print("FD Listener1:", fd_l1)
print("FD Listener2:", fd_l2)
print("FD Pooled:", fd_pool)


# =========================
# OUTPUT DIRECTORY
# =========================
OUT_DIR = "evaluation_results"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# SAVE SUMMARY CSV
# =========================
summary_dict = {
    "L1": [l1_total / n_batches],
    "Velocity_L1": [vel_total / n_batches],
    "FD_Listener1": [fd_l1],
    "FD_Listener2": [fd_l2],
    "FD_Pooled": [fd_pool]
}

summary_df = pd.DataFrame(summary_dict)
summary_df.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)

print("Saved summary_metrics.csv")


# =========================
# SAVE PER-LISTENER FD CSV
# =========================
fd_per_listener_df = pd.DataFrame({
    "Listener": ["Listener1", "Listener2"],
    "FD": [fd_l1, fd_l2]
})

fd_per_listener_df.to_csv(
    os.path.join(OUT_DIR, "fd_per_listener.csv"),
    index=False
)

print("Saved fd_per_listener.csv")


# =========================
# BAR PLOT
# =========================
plt.figure()
plt.bar(["L1", "Vel_L1", "FD_L1", "FD_L2", "FD_Pool"],
        [l1_total / n_batches,
         vel_total / n_batches,
         fd_l1,
         fd_l2,
         fd_pool])

plt.title("Evaluation Metrics")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metrics_bar_plot.png"))
plt.close()

print("Saved metrics_bar_plot.png")


# =========================
# HISTOGRAM OF TOKEN FEATURES
# =========================
plt.figure()
plt.hist(gt_l1_feat.flatten(), bins=50, alpha=0.5, label="GT")
plt.hist(pred_l1_feat.flatten(), bins=50, alpha=0.5, label="Pred")
plt.legend()
plt.title("Feature Distribution (Listener1)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_distribution_l1.png"))
plt.close()

print("Saved feature_distribution_l1.png")


print("\n📊 All evaluation results saved inside:", OUT_DIR)
