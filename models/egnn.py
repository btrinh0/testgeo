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
