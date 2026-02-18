import torch
import numpy as np
import os
from tqdm import tqdm

from stgnn_dataset import STGNNReactionDataset
from stgnn_model import STGNNModel

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DMM_META = "rewritten_metadata/video_3dmm_features_metadata.csv"
AUDIO_META = "rewritten_metadata/audio_features_metadata.csv"
NORM_STATS = "norm_stats.pt"

CHECKPOINT = "checkpoints/stgnn_no_tokenizer_best.pt"

PAST_FRAMES = 150

OUTPUT_ROOT = "inference_test_no_tokenizer"
PRED_DIR = os.path.join(OUTPUT_ROOT, "pred")
GT_DIR   = os.path.join(OUTPUT_ROOT, "gt")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

# ======================
# LOAD NORMALIZATION
# ======================
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

def denormalize(x):
    return x * std + mean

print("✅ Normalization loaded")

# ======================
# LOAD MODEL
# ======================
model = STGNNModel().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

print("✅ ST-GNN loaded")

# ======================
# LOAD DATASET (TEST)
# ======================
dataset = STGNNReactionDataset(
    dmm_metadata_csv=DMM_META,
    audio_metadata_csv=AUDIO_META,
    split="test"
)

print(f"📦 Total test clips: {len(dataset)}")

# ======================
# INFERENCE LOOP
# ======================
for idx in tqdm(range(len(dataset))):

    sample = dataset[idx]
    if sample is None:
        continue

    # ---------------- Inputs ----------------
    speaker = sample["speaker_past"].unsqueeze(0).to(DEVICE)   # (1,150,181)
    audio   = sample["audio_past"].unsqueeze(0).to(DEVICE)     # (1,150,768)
    gt      = sample["listener_future"].to(DEVICE)             # (2,150,181)

    B = 1

    # Build full motion tensor like training
    motion_raw = torch.zeros((B, 3, PAST_FRAMES, 181), device=DEVICE)
    motion_raw[:, 0] = speaker

    # Normalize (same as training)
    motion_norm = normalize(motion_raw)

    # ---------------- Forward ----------------
    with torch.no_grad():
        pred_norm = model(motion_norm, audio)  # (1,3,150,181)

    # Listeners only
    pred_norm_list = pred_norm[:, 1:]          # (1,2,150,181)
    gt_norm_list   = normalize(
        torch.stack([speaker.squeeze(0), gt[0], gt[1]]).unsqueeze(0)
    )[:, 1:]  # same format

    # ---------------- De-normalize ----------------
    pred_list = denormalize(pred_norm_list).squeeze(0).cpu().numpy()
    gt_list   = gt.cpu().numpy()

    # ---------------- Save ----------------
    recording = sample["meta"]["recording"]
    clip_idx  = sample["meta"]["clip_idx"]

    base_name = f"{recording}_clip{clip_idx}"

    np.save(os.path.join(PRED_DIR, base_name + "_pred.npy"), pred_list)
    np.save(os.path.join(GT_DIR, base_name + "_gt.npy"), gt_list)

print("🎉 Full test inference complete.")
