import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network Layer.
    Paper: E(n) Equivariant Graph Neural Networks (https://arxiv.org/abs/2102.09844)
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, aggr='add'):
        super().__init__(aggr=aggr)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Message function: phi_e
        # Inputs: h_i, h_j, ||x_i - x_j||^2, edge_attr
        self.phi_e = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Coordinate update function: phi_x
        # Inputs: m_ij
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False) # Predicts scalar weight for coordinate difference
        )

        # Node feature update function: phi_h
        # Inputs: h_i, m_i (aggregated messages)
        self.phi_h = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        Args:
            h: Node features [N, node_dim]
            x: Node coordinates [N, 3] (or [N, d])
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, edge_dim] (optional)
        """
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=h.device)

        # Propagate messages
        # returns (h_new, x_new)
        return self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)

    def message(self, h_i, h_j, x_i, x_j, edge_attr):
        """
        Calculate message m_ij and coordinate update weight.
        """
        # Calculate squared distance
        diff = x_i - x_j
        dist_sq = torch.sum(diff**2, dim=-1, keepdim=True)

        # Concatenate inputs for phi_e
        # h_i, h_j, dist_sq, edge_attr
        msg_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        
        # Calculate message m_ij
        m_ij = self.phi_e(msg_input)
        
        return m_ij

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # We need to aggregate messages for node update, 
        # BUT we also need to update coordinates using messages BEFORE aggregation.
        # MessagePassing implementation in PyG usually aggregates directly.
        # To update coordinates *per edge* and then aggregate, we might need a custom approach or 
        # return both aggregated messages AND updated coordinates?
        # Actually, the paper says: 
        # x_i^{l+1} = x_i^l + C * \sum (x_i - x_j) * phi_x(m_ij)
        # So we can compute the coordinate update term inside 'message' or standard aggregation?
        # Standard 'propagate' returns the result of 'update'.
        # Since we have two outputs (h and x), we need to be careful.
        
        # Let's delegate to the standard scatter aggregation for m_ij
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def propagate(self, edge_index, size=None, **kwargs):
        # We override propagate to handle the dual update (h and x)
        # However, standard propagate logic is complex to replicate completely if we want to use all PyG features.
        # A simpler way for this specific layer:
        # 1. Compute messages m_ij
        # 2. Compute coordinate updates based on m_ij and relative positions
        # 3. Aggregate messages for node features
        # 4. Update node features
        
        # Deconstruct kwargs
        h = kwargs['h']
        x = kwargs['x']
        edge_attr = kwargs['edge_attr']
        
        row, col = edge_index
        
        # 1. Compute messages and coordinate weights
        diff = x[row] - x[col]
        dist_sq = torch.sum(diff**2, dim=-1, keepdim=True)
        
        # Prepare input for phi_e
        msg_input = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=-1)
        m_ij = self.phi_e(msg_input) # [E, hidden_dim]
        
        # 2. Coordinate update
        # phi_x(m_ij) produces a scalar weight
        coord_weight = self.phi_x(m_ij) # [E, 1]
        
        # The update is sum_{j} (x_i - x_j) * weight
        # This is a sparse aggregation (scatter add)
        # (x_i - x_j) is diff
        weighted_diff = diff * coord_weight
        
        # Aggregation for coordinates
        # We summarize contributions to node i (row index)
        # Note: edge_index is (source, target) or (target, source)? 
        # PyG default is source -> target (messages flow from j to i). 
        # Usually row=target, col=source in MessagePassing flow if we use flow='source_to_target' (default).
        # But wait, edge_index[0] is usually source, edge_index[1] is target.
        # In propagate, we map to (i, j). 
        # If flow is source_to_target:
        #   index i corresponds to edge_index[1] (central nodes)
        #   index j corresponds to edge_index[0] (neighbor nodes)
        # So diff = x_i - x_j = x[edge_index[1]] - x[edge_index[0]]
        
        # Let's trust 'row' and 'col' to be i and j as per 'message' signature if we used it.
        # But here we are precise.
        i = edge_index[1]
        j = edge_index[0]
        
        diff = x[i] - x[j]
        dist_sq = torch.sum(diff**2, dim=-1, keepdim=True)
        msg_input = torch.cat([h[i], h[j], dist_sq, edge_attr], dim=-1)
        m_ij = self.phi_e(msg_input)
        
        coord_weight = self.phi_x(m_ij)
        weighted_diff = diff * coord_weight
        
        # Aggregate coordinate updates
        x_update = torch.zeros_like(x)
        x_update = x_update.index_add_(0, i, weighted_diff)
        
        # Update x (usually with residual connection is implied or explict, paper says x_i + ...)
        x_new = x + x_update / (torch.bincount(i, minlength=x.size(0)).unsqueeze(-1) + 1e-9) # Mean or Sum? Paper uses sum usually but let's stick to sum as per formula. The division is for stability if needed, but paper eqn 4 is just sum.
        # Let's stick to strict sum as per Equation 4 in paper: x_i^{l+1} = x_i^l + \sum ...
        x_new = x + x_update
        
        # 3. Aggregate messages for node features
        # We need to aggregate m_ij over i (target)
        # Using index_add_ again or scatter
        m_i = torch.zeros((h.size(0), self.hidden_dim), device=h.device)
        m_i = m_i.index_add_(0, i, m_ij)
        
        # 4. Node feature update
        h_new = self.phi_h(torch.cat([h, m_i], dim=-1))
        
        # Add residual to h as well? Paper eqn 6: h_i^{l+1} = phi_h(h_i, m_i) + h_i (Residual is common, let's assume standard update first, but usually residual is good)
        # The paper equation 6 does not strictly show + h_i, but it's often used. 
        # However, phi_h input includes h_i, so it can learn to pass it through.
        # But in implementation, explicit residual is safer.
        h_new = h_new + h
        
        return h_new, x_new

class EGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim, edge_dim, hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, h, x, edge_index, edge_attr=None):
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        return h, x


class SiameseEGNN(nn.Module):
    """
    Siamese wrapper around EGNN for contrastive learning with
    Geometric-Semantic Cross-Attention for multimodal fusion.
    
    The cross-attention mechanism allows geometric embeddings (from EGNN)
    to "ask" sequence embeddings (from ESM-2) for evolutionary context.
    
    Updated for ESM-2 Bilingual Brain: input_dim=320 (ESM embedding dimension)
    Input projector shrinks 320 -> 32 immediately to prevent Curse of Dimensionality.
    """
    def __init__(self, input_dim=320, node_dim=32, edge_dim=0, hidden_dim=64, 
                 embed_dim=128, num_layers=4, 
                 geom_dim=32, seq_dim=320, num_heads=4):
        super().__init__()
        self.input_dim = input_dim   # ESM-2 embedding dimension (320)
        self.node_dim = node_dim     # Internal working dimension (32)
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.geom_dim = geom_dim
        self.seq_dim = seq_dim
        
        # ========== Input Projection (Bilingual Brain) ==========
        # Shrink ESM-2 embeddings (320 -> 32) immediately to keep training fast
        # and prevent the "Curse of Dimensionality"
        self.input_projector = nn.Linear(input_dim, node_dim)
        
        # Legacy embedding layer for atomic numbers (fallback, 0-99)
        self.embedding = nn.Embedding(100, node_dim)
        
        # Shared EGNN backbone (now works with 32-dim features)
        self.egnn = EGNN(node_dim, edge_dim, hidden_dim, num_layers)
        
        # Projection head to embedding space (for standard forward_one)
        self.projector = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # ========== Geometric-Semantic Cross-Attention Components ==========
        # Project ESM-2 sequence embeddings (320-dim) down to geometric dim (32-dim)
        self.seq_proj = nn.Linear(seq_dim, geom_dim)
        
        # Cross-attention: geometry queries sequence for evolutionary context
        # Query = geom_emb, Key = seq_proj, Value = seq_proj
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=geom_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Layer normalization for the fusion output
        self.fusion_norm = nn.LayerNorm(geom_dim)
    
    def forward_one(self, data):
        """
        Forward pass for a single graph with ESM-2 features.
        
        Now includes Geometric-Semantic Cross-Attention fusion when ESM-2
        embeddings are available, allowing the model to filter out 
        "coincidental" shape matches that lack evolutionary evidence.
        
        Args:
            data: PyG Data object with:
                - x: ESM-2 embeddings [N, 320] OR atomic numbers [N, 1]
                - pos: Coordinates [N, 3]
                - edge_index: Graph connectivity
            
        Returns:
            Normalized embedding vector [embed_dim]
        """
        # ========== Input Processing ==========
        # Store raw ESM embeddings for cross-attention (if available)
        seq_emb_raw = None
        
        # Check if we have ESM-2 embeddings (320-dim) or atomic numbers (1-dim)
        if data.x.dim() == 2 and data.x.size(1) == self.input_dim:
            # ESM-2 embeddings: [N, 320] -> project to [N, 32]
            seq_emb_raw = data.x.float()  # Store for cross-attention
            h = self.input_projector(seq_emb_raw)
        elif data.x.dim() == 2 and data.x.size(1) == 1:
            # Legacy atomic numbers: use embedding
            x_indices = data.x.squeeze(-1).long()
            h = self.embedding(x_indices)  # [N, node_dim]
        else:
            raise ValueError(f"Unexpected input shape: {data.x.shape}. "
                           f"Expected [N, {self.input_dim}] or [N, 1]")
        
        # Run through EGNN
        h_out, _ = self.egnn(h, data.pos, data.edge_index)
        
        # ========== Cross-Attention Fusion (Phase 10: Bilingual Brain) ==========
        # If we have ESM-2 embeddings, apply cross-attention to fuse
        # geometric features with evolutionary sequence context
        if seq_emb_raw is not None:
            # Add batch dimension for cross-attention: [N, D] -> [1, N, D]
            h_out = self.forward(
                h_out.unsqueeze(0),      # geom_emb: [1, N, 32]
                seq_emb_raw.unsqueeze(0)  # seq_emb: [1, N, 320]
            ).squeeze(0)  # Back to [N, 32]
        
        # Global mean pooling
        if hasattr(data, 'batch') and data.batch is not None:
            # Batched graphs
            from torch_geometric.nn import global_mean_pool
            h_pooled = global_mean_pool(h_out, data.batch)  # [B, node_dim]
        else:
            # Single graph
            h_pooled = h_out.mean(dim=0, keepdim=True)  # [1, node_dim]
        
        # Project to embedding space
        z = self.projector(h_pooled)  # [B, embed_dim] or [1, embed_dim]
        
        # L2 normalize
        z = nn.functional.normalize(z, p=2, dim=-1)
        
        return z
    
    def forward(self, geom_emb, seq_emb):
        """
        Forward pass with Geometric-Semantic Cross-Attention fusion.
        
        This method fuses geometric embeddings (from EGNN) with sequence
        embeddings (from ESM-2) using cross-attention. The geometry "asks"
        the sequence for evolutionary context.
        
        Args:
            geom_emb: EGNN output embeddings [Batch, N, 32]
            seq_emb: ESM-2 sequence embeddings [Batch, N, 320]
            
        Returns:
            fused_emb: Fused embedding for contrastive loss [Batch, N, 32]
        """
        # Step 1: Project sequence embeddings to geometric dimension
        # [Batch, N, 320] -> [Batch, N, 32]
        seq_proj = self.seq_proj(seq_emb)
        
        # Step 2: Cross-attention
        # Query = geom_emb (geometry asks for evolutionary context)
        # Key = seq_proj (sequence provides context)
        # Value = seq_proj (sequence information to retrieve)
        # Output shape: [Batch, N, 32]
        attn_output, _ = self.cross_attention(
            query=geom_emb,
            key=seq_proj,
            value=seq_proj
        )
        
        # Step 3: Residual connection - add attention output to original geom_emb
        fused = geom_emb + attn_output
        
        # Step 4: Layer normalization
        fused_emb = self.fusion_norm(fused)
        
        return fused_emb
    
    def forward_siamese(self, data1, data2=None):
        """
        Forward pass for one or two graphs (Siamese style).
        Legacy method for backward compatibility.
        
        Args:
            data1: First graph (or batch of graphs)
            data2: Second graph (optional, for pair comparison)
            
        Returns:
            z1: Embedding of first graph(s)
            z2: Embedding of second graph(s) if data2 provided, else None
        """
        z1 = self.forward_one(data1)
        
        if data2 is not None:
            z2 = self.forward_one(data2)
            return z1, z2
        
        return z1
