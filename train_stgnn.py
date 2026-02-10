import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stgnn_dataset import STGNNReactionDataset
from stgnn_model import STGNNModel
from models import FaceMotionTokenizerV2


# ======================
# CONFIG
# ======================
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DMM_META = "rewritten_metadata/video_3dmm_features_metadata.csv"
AUDIO_META = "rewritten_metadata/audio_features_metadata.csv"
TOKENIZER_CKPT = "best_tokenizer_fsq_256_v2.pt"
NORM_STATS = "norm_stats.pt"

MAX_FRAMES = 300


# ======================
# Padding helper
# ======================
def pad_time(x, target_len):
    B, T, D = x.shape
    if T >= target_len:
        return x[:, :target_len]
    pad = torch.zeros(B, target_len - T, D, device=x.device)
    return torch.cat([x, pad], dim=1)


# ======================
# Load norm stats
# ======================
norm = torch.load(NORM_STATS, map_location="cpu")
mean = norm["mean"]
std = norm["std"]

if mean.shape[0] < 181:
    pad = 181 - mean.shape[0]
    mean = torch.cat([mean, torch.zeros(pad)])
    std = torch.cat([std, torch.ones(pad)])

mean = mean.to(DEVICE)
std = std.to(DEVICE)

def normalize(x):
    return (x - mean) / std


# ======================
# Tokenizer (FROZEN)
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

print(" Tokenizer loaded and frozen")


# ======================
# Dataset + Loader
# ======================
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    speaker = torch.stack([b["speaker_past"] for b in batch])      # (B,150,181)
    listener1 = torch.stack([b["listener_future"][0] for b in batch])
    listener2 = torch.stack([b["listener_future"][1] for b in batch])

    # Rebuild full motion tensor: (B,3,150,181)
    motion = torch.stack([speaker, listener1, listener2], dim=1)

    audio = torch.stack([b["audio_past"] for b in batch])          # (B,150,768)

    return {
        "motion": motion,
        "audio": audio
    }


train_ds = STGNNReactionDataset(
    dmm_metadata_csv=DMM_META,
    audio_metadata_csv=AUDIO_META,
    split="train"
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_skip_none
)


# ======================
# Model
# ======================
model = STGNNModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.L1Loss()


# ======================
# Training loop
# ======================
best_loss = float("inf")

def velocity_loss(pred, gt):
    """
    pred, gt: (B, 2, T, 181)
    """
    pred_v = pred[:, :, 1:] - pred[:, :, :-1]
    gt_v   = gt[:, :, 1:] - gt[:, :, :-1]
    return torch.mean(torch.abs(pred_v - gt_v))


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        if batch is None:
            continue

        motion_raw = batch["motion"].to(DEVICE)
        audio = batch["audio"].to(DEVICE)

        with torch.no_grad():
            B, N, T, D = motion_raw.shape
            motion_clean = torch.zeros_like(motion_raw)

            for n in range(N):
                norm_m = normalize(motion_raw[:, n])
                clean_m, _ = tokenizer(norm_m)
                clean_m = pad_time(clean_m, T)
                motion_clean[:, n] = clean_m

        pred = model(motion_clean, audio)

        pred_listener = pred[:, 1:]           # (B,2,T,181)
        gt_listener   = motion_clean[:, 1:]

        loss_pos = criterion(pred_listener, gt_listener)
        loss_vel = velocity_loss(pred_listener, gt_listener)

        loss = loss_pos + 0.5 * loss_vel


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "stgnn_best.pt")
        print(" Saved best model")
