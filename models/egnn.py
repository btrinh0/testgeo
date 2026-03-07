"""
GeoMimic-Net: Equivariant Graph Neural Network with Multimodal Fusion

Architecture (Elite-Tier v2):
- EGNNLayer with RBF edge features, LayerNorm, dropout, residual connections
- Attention-weighted pooling (replaces mean pooling)
- Geometric-Semantic Cross-Attention (ESM-2 fusion)
- Wider dimensions: node_dim=64, hidden_dim=128, embed_dim=256

Paper: E(n) Equivariant Graph Neural Networks (https://arxiv.org/abs/2102.09844)
"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Functions for encoding pairwise distances.
    Converts scalar distance into a rich feature vector, giving the EGNN
    much more information about spatial relationships between residues.
    """
    def __init__(self, num_rbf=16, cutoff=10.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('centers', centers)

        widths = torch.full((num_rbf,), (cutoff / num_rbf))
        self.register_buffer('widths', widths)

    def forward(self, dist):
        """
        Args:
            dist: Pairwise distances [E, 1] or [E]
        Returns:
            rbf: Gaussian RBF features [E, num_rbf]
        """
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        return torch.exp(-0.5 * ((dist - self.centers) / self.widths) ** 2)

class EGNNLayer(MessagePassing):
    """
    Enhanced Equivariant Graph Neural Network Layer.

    Improvements over vanilla EGNN:
    - RBF edge features (16-dim) instead of raw distance
    - LayerNorm after message aggregation
    - Dropout for regularization
    - Explicit residual connection (guaranteed, not learned)
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_rbf=16,
                 dropout=0.1, aggr='add'):
        super().__init__(aggr=aggr)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf

        self.rbf = GaussianRBF(num_rbf=num_rbf)

        phi_e_input_dim = 2 * node_dim + num_rbf + edge_dim
        self.phi_e = nn.Sequential(
            nn.Linear(phi_e_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        self.phi_h = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=h.device)
        return self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)

    def propagate(self, edge_index, size=None, **kwargs):
        h = kwargs['h']
        x = kwargs['x']
        edge_attr = kwargs['edge_attr']

        i = edge_index[1]
        j = edge_index[0]

        diff = x[i] - x[j]
        dist = torch.sqrt(torch.sum(diff**2, dim=-1, keepdim=True) + 1e-8)
        rbf_features = self.rbf(dist)

        msg_input = torch.cat([h[i], h[j], rbf_features, edge_attr], dim=-1)
        m_ij = self.phi_e(msg_input)

        coord_weight = self.phi_x(m_ij)
        weighted_diff = diff * coord_weight
        x_update = torch.zeros_like(x)
        x_update = x_update.index_add_(0, i, weighted_diff)
        x_new = x + x_update

        m_i = torch.zeros((h.size(0), self.hidden_dim), device=h.device)
        m_i = m_i.index_add_(0, i, m_ij)

        h_update = self.phi_h(torch.cat([h, m_i], dim=-1))
        h_update = self.dropout(h_update)
        h_new = self.layer_norm(h + h_update)

        return h_new, x_new

class EGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=4,
                 num_rbf=16, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim, edge_dim, hidden_dim, num_rbf=num_rbf,
                     dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h, x, edge_index, edge_attr=None):
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        return h, x

class AttentionPool(nn.Module):
    """
    Attention-weighted global pooling.

    Instead of simple mean pooling (which treats all residues equally),
    this learns which residues are most important for the final embedding.
    Critical for mimicry detection where only specific regions matter.
    """
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, batch=None):
        """
        Args:
            h: Node features [N, node_dim]
            batch: Batch assignment [N] (optional, for batched graphs)
        Returns:
            pooled: [B, node_dim] or [1, node_dim]
        """

        attn_scores = self.attention(h)

        if batch is not None:

            from torch_geometric.utils import softmax
            attn_weights = softmax(attn_scores.squeeze(-1), batch)
            attn_weights = attn_weights.unsqueeze(-1)

            weighted = h * attn_weights
            from torch_geometric.nn import global_add_pool
            pooled = global_add_pool(weighted, batch)
        else:

            attn_weights = torch.softmax(attn_scores, dim=0)
            pooled = (h * attn_weights).sum(dim=0, keepdim=True)

        return pooled

class SiameseEGNN(nn.Module):
    """
    Siamese wrapper around EGNN for contrastive learning with
    Geometric-Semantic Cross-Attention for multimodal fusion.

    Elite-Tier v2 Improvements:
    - RBF edge features (16-dim Gaussian radial basis)
    - Attention-weighted pooling (learns which residues matter)
    - Wider dimensions: node_dim=64, hidden_dim=128, embed_dim=256
    - LayerNorm + Dropout in EGNN layers
    - Deeper projector head (3 layers)

    Backward compatible: loads old weights with strict=False.
    """
    def __init__(self, input_dim=320, node_dim=64, edge_dim=0, hidden_dim=128,
                 embed_dim=256, num_layers=4,
                 geom_dim=64, seq_dim=320, num_heads=4,
                 num_rbf=16, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.geom_dim = geom_dim
        self.seq_dim = seq_dim

        self.input_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.embedding = nn.Embedding(100, node_dim)

        self.egnn = EGNN(node_dim, edge_dim, hidden_dim, num_layers,
                        num_rbf=num_rbf, dropout=dropout)

        self.attention_pool = AttentionPool(node_dim, hidden_dim=hidden_dim)

        self.projector = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.seq_proj = nn.Linear(seq_dim, geom_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=geom_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.fusion_norm = nn.LayerNorm(geom_dim)

    def forward_one(self, data):
        """
        Forward pass for a single graph with ESM-2 features.
        Cross-attention DISABLED: ablation showed it causes dimensional collapse
        (18.8% Top-3 with cross-attn vs 68.8% without).
        """

        seq_emb_raw = None

        if data.x.dim() == 2 and data.x.size(1) == self.input_dim:
            seq_emb_raw = data.x.float()
            h = self.input_projector(seq_emb_raw)
        elif data.x.dim() == 2 and data.x.size(1) == 1:
            x_indices = data.x.squeeze(-1).long()
            h = self.embedding(x_indices)
        else:
            raise ValueError(f"Unexpected input shape: {data.x.shape}. "
                           f"Expected [N, {self.input_dim}] or [N, 1]")

        h_out, _ = self.egnn(h, data.pos, data.edge_index)

        if hasattr(data, 'batch') and data.batch is not None:
            h_pooled = self.attention_pool(h_out, data.batch)
        else:
            h_pooled = self.attention_pool(h_out)

        z = self.projector(h_pooled)

        z = nn.functional.normalize(z, p=2, dim=-1)

        return z

    def forward(self, geom_emb, seq_emb):
        """
        Forward pass with Geometric-Semantic Cross-Attention fusion.
        """
        seq_proj = self.seq_proj(seq_emb)

        attn_output, _ = self.cross_attention(
            query=geom_emb,
            key=seq_proj,
            value=seq_proj
        )

        fused = geom_emb + attn_output
        fused_emb = self.fusion_norm(fused)

        return fused_emb

    def forward_siamese(self, data1, data2=None):
        """Legacy method for backward compatibility."""
        z1 = self.forward_one(data1)

        if data2 is not None:
            z2 = self.forward_one(data2)
            return z1, z2

        return z1
