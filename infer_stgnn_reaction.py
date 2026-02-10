import torch
import numpy as np
import os

from stgnn_dataset import STGNNReactionDataset
from stgnn_model import STGNNModel
from models import FaceMotionTokenizerV2

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DMM_META = "rewritten_metadata/video_3dmm_features_metadata.csv"
AUDIO_META = "rewritten_metadata/audio_features_metadata.csv"
TOKENIZER_CKPT = "best_tokenizer_fsq_256_v2.pt"
STGNN_CKPT = "stgnn_best.pt"
NORM_STATS = "norm_stats.pt"

OUTPUT_DIR = "reaction_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAST_FRAMES = 150
MAX_FRAMES = 300

# choose clip
TARGET_RECORDING = "recording13"
TARGET_CLIP_IDX = 20


# ======================
# LOAD NORM STATS
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


# ======================
# LOAD TOKENIZER
# ======================
tokenizer = FaceMotionTokenizerV2(
    input_dim=181,
    down_t=3,
    stride_t=2,
    quantizer="fsq",
    embed=256,
    levels=[8, 5, 5]
).to(DEVICE)

ckpt = torch.load(TOKENIZER_CKPT, map_location=DEVICE)
tokenizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
tokenizer.eval()

for p in tokenizer.parameters():
    p.requires_grad = False

print("Tokenizer loaded")


# ======================
# LOAD ST-GNN MODEL
# ======================
model = STGNNModel().to(DEVICE)
model.load_state_dict(torch.load(STGNN_CKPT, map_location=DEVICE))
model.eval()

print(" ST-GNN loaded")


# ======================
# LOAD DATASET
# ======================
dataset = STGNNReactionDataset(
    dmm_metadata_csv=DMM_META,
    audio_metadata_csv=AUDIO_META,
    split="test"   # or train/val
)

sample = None
for i in range(len(dataset)):
    item = dataset[i]
    if item is None:
        continue
    if (item["meta"]["recording"] == TARGET_RECORDING and
        item["meta"]["clip_idx"] == TARGET_CLIP_IDX):
        sample = item
        break

if sample is None:
    raise RuntimeError(" Clip not found in dataset")

print(" Clip loaded")


# ======================
# PREPARE INPUT
# ======================
speaker_past = sample["speaker_past"].unsqueeze(0).to(DEVICE)      # (1,150,181)
audio_past   = sample["audio_past"].unsqueeze(0).to(DEVICE)        # (1,150,768)

# build motion tensor (speaker + empty listeners)
motion_raw = torch.zeros((1, 3, PAST_FRAMES, 181), device=DEVICE)
motion_raw[:, 0] = speaker_past


# ======================
# TOKENIZER
# ======================
with torch.no_grad():
    motion_clean = torch.zeros_like(motion_raw)

    for n in range(3):
        norm_m = normalize(motion_raw[:, n])
        clean_m, _ = tokenizer(norm_m)
        T = motion_raw.shape[2]
        if clean_m.shape[1] < T:
            pad = torch.zeros(1, T - clean_m.shape[1], 181, device=DEVICE)
            clean_m = torch.cat([clean_m, pad], dim=1)
        motion_clean[:, n] = clean_m[:, :T]


# ======================
# ST-GNN INFERENCE
# ======================
with torch.no_grad():
    pred = model(motion_clean, audio_past)  # (1,3,150,181)

listener_future = pred[:, 1:]   # (1,2,150,181)
listener_future = listener_future.squeeze(0).cpu().numpy()

# ======================
# SAVE
# ======================
out_path = os.path.join(
    OUTPUT_DIR,
    f"{TARGET_RECORDING}_clip{TARGET_CLIP_IDX}_listener_future_181.npy"
)

np.save(out_path, listener_future)
print(f" Saved inference: {out_path}")
