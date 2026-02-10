import torch
import numpy as np
import sys
import os

# Add the STGNN folder (one level up from utils) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stgnn_model import STGNNModel
from models import FaceMotionTokenizerV2

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER_CKPT = "best_tokenizer_fsq_256_v2.pt"
STGNN_CKPT = "stgnn_best.pt"
NORM_STATS = "norm_stats.pt"

OUTPUT_DIR = "reaction_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAST_FRAMES = 150
LISTENER_INDEX = 0  # 0 = listener-1, 1 = listener-2


# ======================
# INPUT PATHS (YOU CONTROL THESE)
# ======================
SPEAKER_MOTION_PATH = "/home/mudasir/Pawan/MPII/stacked_npy/test/recording13/subjectPos2/listener/clip_1_0_cropped.npy"   # (T,181)
AUDIO_PATH = "/home/mudasir/Pawan/MPII/facial_reaction_clips/test/recording13/audio_features/clip_0_speaker.npy"                # (T,768)


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

print("✅ Tokenizer loaded")


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
speaker_motion = np.load(SPEAKER_MOTION_PATH).astype(np.float32)
audio = np.load(AUDIO_PATH).astype(np.float32)

speaker_motion = speaker_motion[:PAST_FRAMES]
audio = audio[:PAST_FRAMES]

speaker_motion = torch.from_numpy(speaker_motion).unsqueeze(0).to(DEVICE)
audio = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)


# ======================
# BUILD MODEL INPUT
# ======================
motion_raw = torch.zeros((1, 3, PAST_FRAMES, 181), device=DEVICE)
motion_raw[:, 0] = speaker_motion


# ======================
# TOKENIZER
# ======================
with torch.no_grad():
    motion_clean = torch.zeros_like(motion_raw)

    for n in range(3):
        norm_m = normalize(motion_raw[:, n])
        clean_m, _ = tokenizer(norm_m)

        if clean_m.shape[1] < PAST_FRAMES:
            pad = torch.zeros(
                1, PAST_FRAMES - clean_m.shape[1], 181, device=DEVICE
            )
            clean_m = torch.cat([clean_m, pad], dim=1)

        motion_clean[:, n] = clean_m[:, :PAST_FRAMES]


# ======================
# INFERENCE
# ======================
with torch.no_grad():
    pred = model(motion_clean, audio)  # (1,3,150,181)

listener_pred = pred[:, 1 + LISTENER_INDEX]   # (1,150,181)
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
