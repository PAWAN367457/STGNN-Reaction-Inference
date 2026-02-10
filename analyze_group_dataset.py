import pandas as pd
import numpy as np
import os
from collections import defaultdict

# ==========================
# CONFIG
# ==========================
DMM_META = "rewritten_metadata/video_3dmm_features_metadata.csv"
AUDIO_META = "rewritten_metadata/audio_features_metadata.csv"

MIN_FRAMES = 50
STATIC_STD_THRESH = 1e-4

# ==========================
# LOAD METADATA
# ==========================
dmm_df = pd.read_csv(DMM_META)
audio_df = pd.read_csv(AUDIO_META)

print("Loaded motion clips:", len(dmm_df))
print("Loaded audio clips:", len(audio_df))
print("=" * 60)

# ==========================
# GROUP BY CLIP
# ==========================
groups = dmm_df.groupby(["recording", "clip_idx"])

stats = defaultdict(int)
drop_reasons = defaultdict(int)

listener_count_hist = defaultdict(int)
frame_stats = []
motion_std_stats = []

bad_clips = []

# ==========================
# ANALYSIS LOOP
# ==========================
for (recording, clip_idx), rows in groups:

    speakers = rows[rows["speaker_id"] == 1]
    listeners = rows[rows["speaker_id"] == 0]

    num_speakers = len(speakers)
    num_listeners = len(listeners)

    listener_count_hist[num_listeners] += 1

    # ---------- Structural validity ----------
    if num_speakers != 1:
        drop_reasons["invalid_speaker_count"] += 1
        bad_clips.append((recording, clip_idx, "invalid_speaker_count"))
        continue

    if num_listeners < 2:
        drop_reasons["too_few_listeners"] += 1
        bad_clips.append((recording, clip_idx, "too_few_listeners"))
        continue

    # ---------- Load motions ----------
    clip_bad = False
    for _, row in rows.iterrows():
        path = row["3dmm_feature_path"]

        if not os.path.exists(path):
            drop_reasons["missing_file"] += 1
            clip_bad = True
            break

        motion = np.load(path)
        T = motion.shape[0]
        std = np.std(motion)

        frame_stats.append(T)
        motion_std_stats.append(std)

        if T < MIN_FRAMES:
            drop_reasons["too_short"] += 1
            clip_bad = True
            break

        if std < STATIC_STD_THRESH:
            drop_reasons["static_motion"] += 1
            clip_bad = True
            break

    if clip_bad:
        bad_clips.append((recording, clip_idx, "motion_quality"))
        continue

    # ---------- Audio check ----------
    audio_rows = audio_df[
        (audio_df["recording"] == recording) &
        (audio_df["clip_idx"] == clip_idx)
    ]

    if len(audio_rows) != 1:
        drop_reasons["audio_mismatch"] += 1
        bad_clips.append((recording, clip_idx, "audio_mismatch"))
        continue

    # ---------- Valid clip ----------
    stats["valid_clips"] += 1

# ==========================
# REPORT
# ==========================
print("\n📊 DATASET SUMMARY")
print("=" * 60)
print(f"Total clips               : {len(groups)}")
print(f"Valid clips               : {stats['valid_clips']}")
print(f"Dropped clips             : {len(groups) - stats['valid_clips']}")

print("\n👥 Listener count distribution:")
for k in sorted(listener_count_hist):
    print(f"  {k} listeners: {listener_count_hist[k]}")

print("\n🚫 Drop reasons:")
for k, v in drop_reasons.items():
    print(f"  {k}: {v}")

if frame_stats:
    print("\n⏱ Motion length stats:")
    print(f"  Min frames: {min(frame_stats)}")
    print(f"  Mean frames: {np.mean(frame_stats):.1f}")
    print(f"  Max frames: {max(frame_stats)}")

if motion_std_stats:
    print("\n📉 Motion variance stats:")
    print(f"  Min std: {min(motion_std_stats):.6f}")
    print(f"  Mean std: {np.mean(motion_std_stats):.6f}")
    print(f"  Max std: {max(motion_std_stats):.6f}")

print("\n⚠️ Example bad clips (first 10):")
for c in bad_clips[:10]:
    print(" ", c)

print("\nAnalysis complete.")
