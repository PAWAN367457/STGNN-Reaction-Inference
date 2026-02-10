import torch
import torch.nn as nn
from biased_gat import BiasedGraphAttention


class STGNNBlock(nn.Module):
    """
    Spatio-Temporal GNN Block

    Input:
        x: (B, N=3, T, D)
    Output:
        out: (B, N=3, T, H)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_nodes=3,
        gru_layers=1
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # --- Spatial ---
        self.gat = BiasedGraphAttention(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num_nodes=num_nodes
        )

        self.spatial_norm = nn.LayerNorm(hidden_dim)

        # --- Temporal ---
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        self.temporal_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: (B, N, T, D)
        """
        B, N, T, D = x.shape
        assert N == self.num_nodes

        # =========================
        # Spatial GAT (per time)
        # =========================
        spatial_out = []

        for t in range(T):
            x_t = x[:, :, t, :]            # (B, N, D)
            h_t = self.gat(x_t)            # (B, N, H)
            spatial_out.append(h_t)

        # (B, N, T, H)
        h = torch.stack(spatial_out, dim=2)
        h = self.spatial_norm(h)

        # =========================
        # Temporal GRU (per node)
        # =========================
        out = []

        for n in range(N):
            # (B, T, H)
            h_n = h[:, n, :, :]
            h_n, _ = self.gru(h_n)
            out.append(h_n)

        # (B, N, T, H)
        out = torch.stack(out, dim=1)
        out = self.temporal_norm(out)

        return out
