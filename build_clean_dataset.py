import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ==========================
# CONFIG
# ==========================
DMM_META = "rewritten_metadata/video_3dmm_features_metadata.csv"
AUDIO_META = "rewritten_metadata/audio_features_metadata.csv"

OUT_ROOT = Path("MPII_GroupReaction_Clean")
MOTION_OUT = OUT_ROOT / "motion"
AUDIO_OUT = OUT_ROOT / "audio"
META_OUT = OUT_ROOT / "metadata"

MIN_FRAMES = 50
STATIC_STD_THRESH = 1e-4

os.makedirs(MOTION_OUT, exist_ok=True)
os.makedirs(AUDIO_OUT, exist_ok=True)
os.makedirs(META_OUT, exist_ok=True)

# ==========================
# LOAD METADATA
# ==========================
dmm_df = pd.read_csv(DMM_META)
audio_df = pd.read_csv(AUDIO_META)

groups = dmm_df.groupby(["recording", "clip_idx"])

motion_rows = []
audio_rows = []

kept = 0
dropped = 0

# ==========================
# HELPERS
# ==========================
def motion_energy(x):
    return np.mean(np.abs(x[1:] - x[:-1]))

# ==========================
# MAIN LOOP
# ==========================
for (recording, clip_idx), rows in groups:

    motions = []
    for _, r in rows.iterrows():
        path = r["3dmm_feature_path"]
        if not os.path.exists(path):
            continue
        x = np.load(path)
        motions.append({
            "path": path,
            "motion": x,
            "speaker_id": r["speaker_id"],
            "energy": motion_energy(x),
            "std": np.std(x)
        })

    if len(motions) < 3:
        dropped += 1
        continue

    # ---------------- Speaker selection ----------------
    speakers = [m for m in motions if m["speaker_id"] == 1]
    if len(speakers) == 0:
        dropped += 1
        continue

    speaker = max(speakers, key=lambda m: m["energy"])

    # ---------------- Listener selection ----------------
    listeners = [m for m in motions if m is not speaker]
    listeners = sorted(listeners, key=lambda m: m["energy"], reverse=True)

    if len(listeners) < 2:
        dropped += 1
        continue

    listener1, listener2 = listeners[:2]

    trio = [speaker, listener1, listener2]

    # ---------------- Quality checks ----------------
    bad = False
    for m in trio:
        if m["motion"].shape[0] < MIN_FRAMES:
            bad = True
        if m["std"] < STATIC_STD_THRESH:
            bad = True

    if bad:
        dropped += 1
        continue

    # ---------------- Audio ----------------
    audio_rows_clip = audio_df[
        (audio_df["recording"] == recording) &
        (audio_df["clip_idx"] == clip_idx)
    ]

    if len(audio_rows_clip) == 0:
        dropped += 1
        continue

    audio_path = audio_rows_clip.iloc[0]["audio_feature_path"]
    if not os.path.exists(audio_path):
        dropped += 1
        continue

    audio = np.load(audio_path)

    # ---------------- Save ----------------
    clip_dir = MOTION_OUT / recording / f"clip{clip_idx}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    np.save(clip_dir / "speaker.npy", speaker["motion"])
    np.save(clip_dir / "listener1.npy", listener1["motion"])
    np.save(clip_dir / "listener2.npy", listener2["motion"])

    audio_dir = AUDIO_OUT / recording
    audio_dir.mkdir(parents=True, exist_ok=True)
    np.save(audio_dir / f"clip{clip_idx}.npy", audio)

    # ---------------- Metadata ----------------
    for role, m in zip(
        ["speaker", "listener1", "listener2"],
        [speaker, listener1, listener2]
    ):
        motion_rows.append({
            "recording": recording,
            "clip_idx": clip_idx,
            "role": role,
            "motion_path": str(clip_dir / f"{role}.npy"),
            "num_frames": m["motion"].shape[0]
        })

    audio_rows.append({
        "recording": recording,
        "clip_idx": clip_idx,
        "audio_path": str(audio_dir / f"clip{clip_idx}.npy"),
        "num_frames": audio.shape[0]
    })

    kept += 1

# ==========================
# SAVE METADATA
# ==========================
pd.DataFrame(motion_rows).to_csv(META_OUT / "motion_metadata.csv", index=False)
pd.DataFrame(audio_rows).to_csv(META_OUT / "audio_metadata.csv", index=False)

print("✅ Clean dataset built")
print(f"Kept clips   : {kept}")
print(f"Dropped clips: {dropped}")
