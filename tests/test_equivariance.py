import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import EGNN

def get_random_rotation_matrix():
    """Generates a random 3D rotation matrix."""

    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    return Q

def test_equivariance():
    print("Testing EGNN Equivariance...")

    torch.manual_seed(42)
    node_dim = 16
    edge_dim = 4
    hidden_dim = 32
    num_nodes = 10
    num_edges = 20

    model = EGNN(node_dim=node_dim, edge_dim=node_dim, hidden_dim=hidden_dim, num_layers=2)
    model = EGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=2)
    model.eval()

    h = torch.randn(num_nodes, node_dim)
    x = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)

    R = get_random_rotation_matrix()

    h1, x1 = model(h, x, edge_index, edge_attr)

    x_rotated = x @ R.T

    h2, x2 = model(h, x_rotated, edge_index, edge_attr)

    x1_rotated = x1 @ R.T

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
