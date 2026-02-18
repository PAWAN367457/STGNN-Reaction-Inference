import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from stgnn_dataset import STGNNReactionDataset
from stgnn_model import STGNNModel


# ======================
# CONFIG
# ======================
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DMM_META = "rewritten_metadata/video_3dmm_features_metadata.csv"
AUDIO_META = "rewritten_metadata/audio_features_metadata.csv"
NORM_STATS = "norm_stats.pt"

PAST_FRAMES = 150


# ======================
# Load norm stats
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


# ======================
# Dataset + Loader
# ======================
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
    collate_fn=collate_fn
)


# ======================
# Model
# ======================
model = STGNNModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.L1Loss()


# ======================
# Velocity Loss
# ======================
def velocity_loss(pred, gt):
    pred_v = pred[:, :, 1:] - pred[:, :, :-1]
    gt_v   = gt[:, :, 1:] - gt[:, :, :-1]
    return torch.mean(torch.abs(pred_v - gt_v))


# ======================
# Training Loop
# ======================
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        if batch is None:
            continue

        motion_raw = batch["motion"].to(DEVICE)  # (B,3,T,181)
        audio = batch["audio"].to(DEVICE)

        # Normalize directly
        motion_norm = normalize(motion_raw)

        # Forward
        pred = model(motion_norm, audio)

        pred_listener = pred[:, 1:]
        gt_listener   = motion_norm[:, 1:]

        loss_pos = criterion(pred_listener, gt_listener)
        loss_vel = velocity_loss(pred_listener, gt_listener)

        loss = loss_pos + 0.5 * loss_vel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f}")
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/stgnn_no_tokenizer_best.pt")
        print("✅ Saved best model")

print("🎉 Training complete")
