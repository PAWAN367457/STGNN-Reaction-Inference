import torch
import torch.nn as nn
from stgnn_block import STGNNBlock


class STGNNModel(nn.Module):
    def __init__(self, motion_dim=181, audio_dim=768, hidden_dim=256):
        super().__init__()

        # Motion embedding
        self.motion_embed = nn.Linear(motion_dim, hidden_dim)

        #  Audio encoder (simple & sufficient)
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ST-GNN
        self.stgnn = STGNNBlock(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_nodes=3
        )

        # Output head
        self.out = nn.Linear(hidden_dim, motion_dim)

    def forward(self, motion, audio):
        """
        motion: (B, 3, T, 181)
        audio:  (B, T, 768)
        """

        B, N, T, _ = motion.shape

        # Embed motion
        x = self.motion_embed(motion)  # (B, 3, T, H)
        # -------------------------------------------------
        #  BLOCK SPEAKER MOTION FROM LISTENER INPUT
        # -------------------------------------------------
        # Speaker node (0): keep motion
        # Listener nodes (1,2): ZERO OUT speaker channels
        x[:, 1:, :, :] = 0.0

        # Encode audio
        a = self.audio_embed(audio)    # (B, T, H)

        #  Inject audio into SPEAKER node only (node 0)
        x[:, 0, :, :] = x[:, 0, :, :] + a

        # ST-GNN
        h = self.stgnn(x)              # (B, 3, T, H)

        # Decode
        out = self.out(h)              # (B, 3, T, 181)
        return out
