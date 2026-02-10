import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class STGNNReactionDataset(Dataset):
    """
    Reaction prediction dataset

    Returns:
      speaker_past:     (150, 181)
      listener_future:  (2, 150, 181)
      audio_past:       (150, 768)
      meta:             dict
    """

    def __init__(
        self,
        dmm_metadata_csv,
        audio_metadata_csv,
        split="train",
        max_frames=300,
        past_frames=150
    ):
        self.max_frames = max_frames
        self.past_frames = past_frames
        self.future_frames = max_frames - past_frames
        self.split = split

        # Load metadata
        self.dmm_df = pd.read_csv(dmm_metadata_csv)
        self.audio_df = pd.read_csv(audio_metadata_csv)

        if "split" in self.dmm_df.columns:
            self.dmm_df = self.dmm_df[self.dmm_df["split"] == split]
        if "split" in self.audio_df.columns:
            self.audio_df = self.audio_df[self.audio_df["split"] == split]

        # Group by recording + clip_idx
        self.groups = self.dmm_df.groupby(["recording", "clip_idx"])
        self.keys = list(self.groups.groups.keys())

        print(f"[{split.upper()}] Reaction samples: {len(self.keys)}")

    def __len__(self):
        return len(self.keys)

    def pad_or_truncate(self, x):
        T = x.shape[0]
        if T == self.max_frames:
            return x
        if T > self.max_frames:
            return x[:self.max_frames]
        pad = np.zeros((self.max_frames - T, x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)

    def load_motion(self, path):
        x = np.load(path).astype(np.float32)
        return self.pad_or_truncate(x)

    def __getitem__(self, idx):
        recording, clip_idx = self.keys[idx]
        rows = self.groups.get_group((recording, clip_idx))

        speaker_motion = None
        listeners = []

        # Identify roles
        for _, row in rows.iterrows():
            motion = self.load_motion(row["3dmm_feature_path"])
            if row["speaker_id"] == 1:
                speaker_motion = motion
            else:
                listeners.append(motion)

        # Skip invalid clips
        if speaker_motion is None:
            return None

        # Ensure exactly 2 listeners
        while len(listeners) < 2:
            listeners.append(np.zeros_like(speaker_motion))
        listeners = listeners[:2]

        # Split time
        speaker_past = speaker_motion[:self.past_frames]
        listener_future = np.stack(
            [
                listeners[0][self.past_frames:],
                listeners[1][self.past_frames:]
            ],
            axis=0
        )  # (2, 150, 181)

        # Load audio (shared)
        audio_row = self.audio_df[
            (self.audio_df["recording"] == recording) &
            (self.audio_df["clip_idx"] == clip_idx)
        ]

        if len(audio_row) == 0:
            audio = np.zeros((self.max_frames, 768), dtype=np.float32)
        else:
            audio = np.load(audio_row.iloc[0]["audio_feature_path"]).astype(np.float32)
            audio = self.pad_or_truncate(audio)

        audio_past = audio[:self.past_frames]

        return {
            "speaker_past": torch.from_numpy(speaker_past),          # (150, 181)
            "listener_future": torch.from_numpy(listener_future),    # (2, 150, 181)
            "audio_past": torch.from_numpy(audio_past),              # (150, 768)
            "meta": {
                "recording": recording,
                "clip_idx": clip_idx
            }
        }
