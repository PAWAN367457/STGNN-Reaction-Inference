import torch
import numpy as np
import os

from stgnn_model import STGNNModel

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STGNN_CKPT = "checkpoints/stgnn_no_tokenizer_best.pt"
NORM_STATS = "norm_stats.pt"

OUTPUT_DIR = "reaction_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAST_FRAMES = 150
LISTENER_INDEX = 0  # 0 = listener1, 1 = listener2


# ======================
# INPUT PATHS (YOU CONTROL)
# ======================
SPEAKER_MOTION_PATH = "/home/mudasir/Pawan/MPII/stacked_npy/test/recording08/subjectPos2/speaker/clip_1_1_cropped.npy"
AUDIO_PATH = "/home/mudasir/Pawan/MPII/facial_reaction_clips/test/recording08/audio_features/clip_1_speaker.npy"


# ======================
# LOAD NORMALIZATION
# ======================
norm = torch.load(NORM_STATS, map_location="cpu")
mean = norm["mean"]
std = norm["std"]

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


# ======================
# LOAD ST-GNN
# ======================
model = STGNNModel().to(DEVICE)
model.load_state_dict(torch.load(STGNN_CKPT, map_location=DEVICE))
model.eval()

print("✅ ST-GNN loaded")


# ======================
# LOAD INPUT DATA
# ======================
def pad_or_truncate_np(x, target_len):
    T, D = x.shape
    if T >= target_len:
        return x[:target_len]
    pad = np.zeros((target_len - T, D), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)

speaker_motion = np.load(SPEAKER_MOTION_PATH).astype(np.float32)
audio = np.load(AUDIO_PATH).astype(np.float32)

speaker_motion = pad_or_truncate_np(speaker_motion, PAST_FRAMES)
audio = pad_or_truncate_np(audio, PAST_FRAMES)

speaker_motion = torch.from_numpy(speaker_motion).unsqueeze(0).to(DEVICE)
audio = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)

# Normalize (must match training)
speaker_motion = normalize(speaker_motion)


# ======================
# BUILD MODEL INPUT
# ======================
motion_input = torch.zeros((1, 3, PAST_FRAMES, 181), device=DEVICE)
motion_input[:, 0] = speaker_motion   # speaker only


# ======================
# INFERENCE
# ======================
with torch.no_grad():
    pred = model(motion_input, audio)  # (1,3,150,181)

listener_pred = pred[:, 1 + LISTENER_INDEX]  # (1,150,181)

# Denormalize back to raw 3DMM
listener_pred = denormalize(listener_pred)

listener_pred = listener_pred.squeeze(0).cpu().numpy()


# ======================
# SAVE
# ======================
out_path = os.path.join(
    OUTPUT_DIR,
    f"single_listener_{LISTENER_INDEX}_future_181.npy"
)

np.save(out_path, listener_pred)

print(f"💾 Saved single-listener prediction: {out_path}")
