import torch
import sys
import os

# Add project root to sys.path to allow importing from models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import EGNN

def get_random_rotation_matrix():
    """Generates a random 3D rotation matrix."""
    # QR decomposition of a random matrix
    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    return Q

def test_equivariance():
    print("Testing EGNN Equivariance...")
    
    # 1. Setup
    torch.manual_seed(42)
    node_dim = 16
    edge_dim = 4
    hidden_dim = 32
    num_nodes = 10
    num_edges = 20
    
    model = EGNN(node_dim=node_dim, edge_dim=node_dim, hidden_dim=hidden_dim, num_layers=2) # Note edge_dim param name in EGNN is edge_dim but I used node_dim just for simplicity in call, wait, let's fix.
    model = EGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=2)
    model.eval()
    
    # 2. Create random graph
    h = torch.randn(num_nodes, node_dim)
    x = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)
    
    # 3. Create random rotation
    R = get_random_rotation_matrix()
    
    # 4. Forward pass 1: Original coordinates
    h1, x1 = model(h, x, edge_index, edge_attr)
    
    # 5. Forward pass 2: Rotated coordinates
    # Rotate input coordinates: x' = x @ R.T
    x_rotated = x @ R.T 
    
    h2, x2 = model(h, x_rotated, edge_index, edge_attr)
    
    # 6. Check equivariance
    # We expect:
    # h2 should be close to h1 (invariant)
    # x2 should be close to x1 @ R.T (equivariant)
    
    # Rotate x1 to compare with x2
    x1_rotated = x1 @ R.T
    
    # Errors
    h_error = torch.max(torch.abs(h1 - h2)).item()
    x_error = torch.max(torch.abs(x1_rotated - x2)).item()
    
    print(f"Feature invariance check (Max Error): {h_error:.6f}")
    print(f"Coordinate equivariance check (Max Error): {x_error:.6f}")
    
    TOLERANCE = 1e-5
    if h_error < TOLERANCE and x_error < TOLERANCE:
        print("SUCCESS: Model is rotation equivariant!")
    else:
        print("FAILURE: Model is NOT rotation equivariant!")
        print(f"Rotation Matrix:\n{R}")

if __name__ == "__main__":
    test_equivariance()
