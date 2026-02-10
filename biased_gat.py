import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BiasedGraphAttention(nn.Module):
    """
    Biased GAT layer for small fixed graphs (e.g. 3 nodes)

    Input:
        x: (B, N, D_in)
    Output:
        out: (B, N, D_out)
    """

    def __init__(self, in_dim, out_dim, num_nodes):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Linear projections
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # 🔥 Learnable structural bias (j → i)
        self.bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # Optional output projection
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """
        x: (B, N, D_in)
        """
        B, N, _ = x.shape
        assert N == self.num_nodes

        # Project
        Q = self.W_q(x)   # (B, N, D_out)
        K = self.W_k(x)   # (B, N, D_out)
        V = self.W_v(x)   # (B, N, D_out)

        # Attention scores: (B, N, N)
        attn = torch.matmul(Q, K.transpose(-2, -1))
        attn = attn / math.sqrt(self.out_dim)

        # 🔥 Add structural bias
        attn = attn + self.bias.unsqueeze(0)

        # Normalize
        alpha = F.softmax(attn, dim=-1)

        # Aggregate
        out = torch.matmul(alpha, V)  # (B, N, D_out)

        return self.out_proj(out)
