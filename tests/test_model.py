"""
Tests for model components: GaussianRBF, AttentionPool, EGNNLayer, EGNN, SiameseEGNN.

These tests verify shapes, dtypes, and basic properties without requiring
PDB files or ESM-2 — they use synthetic random data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from models.egnn import GaussianRBF, EGNNLayer, EGNN, AttentionPool, SiameseEGNN

def test_rbf_output_shape():
    """RBF should map scalar distances to num_rbf-dim vectors."""
    rbf = GaussianRBF(num_rbf=16, cutoff=10.0)
    dist = torch.rand(50)
    out = rbf(dist)
    assert out.shape == (50, 16), f"Expected (50, 16), got {out.shape}"

def test_rbf_output_range():
    """RBF outputs should be in (0, 1] (Gaussian values)."""
    rbf = GaussianRBF(num_rbf=8)
    dist = torch.linspace(0, 10, 100)
    out = rbf(dist)
    assert out.min() >= 0, f"Negative RBF value: {out.min()}"
    assert out.max() <= 1 + 1e-6, f"RBF value > 1: {out.max()}"

def test_rbf_different_sizes():
    """RBF should work with different num_rbf values."""
    for n in [1, 4, 16, 32]:
        rbf = GaussianRBF(num_rbf=n)
        out = rbf(torch.rand(10))
        assert out.shape == (10, n)

def test_attention_pool_single_graph():
    """Pools N nodes into 1 embedding without batch index."""
    pool = AttentionPool(node_dim=32, hidden_dim=16)
    h = torch.randn(20, 32)
    pooled = pool(h)
    assert pooled.shape == (1, 32), f"Expected (1, 32), got {pooled.shape}"

def test_attention_pool_batched():
    """Pools batched graphs into B embeddings."""
    pool = AttentionPool(node_dim=64, hidden_dim=32)
    h = torch.randn(30, 64)
    batch = torch.tensor([0]*10 + [1]*10 + [2]*10)
    pooled = pool(h, batch)
    assert pooled.shape == (3, 64), f"Expected (3, 64), got {pooled.shape}"

def test_attention_pool_different_sizes():
    """Each graph in batch can have different number of nodes."""
    pool = AttentionPool(node_dim=32)
    h = torch.randn(25, 32)
    batch = torch.tensor([0]*5 + [1]*15 + [2]*5)
    pooled = pool(h, batch)
    assert pooled.shape == (3, 32)

def test_egnn_output_shape():
    """EGNN should preserve node feature dimension and coordinate dimension."""
    egnn = EGNN(node_dim=32, edge_dim=0, hidden_dim=64, num_layers=2, num_rbf=8)
    n_nodes = 15
    h = torch.randn(n_nodes, 32)
    x = torch.randn(n_nodes, 3)

    edge_index = torch.randint(0, n_nodes, (2, 40))

    h_out, x_out = egnn(h, x, edge_index)
    assert h_out.shape == (n_nodes, 32), f"Expected ({n_nodes}, 32), got {h_out.shape}"
    assert x_out.shape == (n_nodes, 3), f"Expected ({n_nodes}, 3), got {x_out.shape}"

def test_egnn_no_nan():
    """EGNN should not produce NaNs with random input."""
    egnn = EGNN(node_dim=16, edge_dim=0, hidden_dim=32, num_layers=2)
    h = torch.randn(10, 16)
    x = torch.randn(10, 3)
    edge_index = torch.randint(0, 10, (2, 20))

    h_out, x_out = egnn(h, x, edge_index)
    assert not torch.isnan(h_out).any(), "NaN in node features"
    assert not torch.isnan(x_out).any(), "NaN in coordinates"

def _make_dummy_graph(n_nodes=20, feat_dim=320):
    """Create a dummy PyG-like graph for testing."""
    from types import SimpleNamespace
    data = SimpleNamespace()
    data.x = torch.randn(n_nodes, feat_dim)
    data.pos = torch.randn(n_nodes, 3)
    data.edge_index = torch.randint(0, n_nodes, (2, n_nodes * 3))
    data.batch = None
    return data

def test_siamese_forward_one_shape():
    """forward_one should produce a normalized embedding vector."""
    model = SiameseEGNN(
        node_dim=32, edge_dim=0, hidden_dim=64,
        embed_dim=128, num_layers=2, geom_dim=32,
        num_rbf=8, dropout=0.0
    )
    model.eval()

    data = _make_dummy_graph(n_nodes=15, feat_dim=320)
    with torch.no_grad():
        emb = model.forward_one(data)

    assert emb.shape == (1, 128), f"Expected (1, 128), got {emb.shape}"

def test_siamese_embedding_normalized():
    """Embeddings should be L2-normalized (norm ≈ 1)."""
    model = SiameseEGNN(
        node_dim=32, edge_dim=0, hidden_dim=64,
        embed_dim=128, num_layers=2, geom_dim=32,
    )
    model.eval()

    data = _make_dummy_graph()
    with torch.no_grad():
        emb = model.forward_one(data)

    norm = torch.norm(emb, p=2, dim=-1).item()
    assert abs(norm - 1.0) < 1e-4, f"Embedding norm is {norm}, expected ~1.0"

def test_siamese_same_input_high_similarity():
    """The same graph fed twice should produce similarity ≈ 1.0."""
    model = SiameseEGNN(
        node_dim=32, edge_dim=0, hidden_dim=64,
        embed_dim=128, num_layers=2, geom_dim=32,
    )
    model.eval()

    data = _make_dummy_graph()
    with torch.no_grad():
        emb1 = model.forward_one(data)
        emb2 = model.forward_one(data)

    sim = F.cosine_similarity(emb1, emb2).item()
    assert sim > 0.99, f"Self-similarity = {sim}, expected ~1.0"

def test_siamese_different_inputs():
    """Different graphs should produce different embeddings."""
    model = SiameseEGNN(
        node_dim=32, edge_dim=0, hidden_dim=64,
        embed_dim=128, num_layers=2, geom_dim=32,
    )
    model.eval()

    data1 = _make_dummy_graph(n_nodes=15)
    data2 = _make_dummy_graph(n_nodes=25)

    with torch.no_grad():
        emb1 = model.forward_one(data1)
        emb2 = model.forward_one(data2)

    diff = torch.max(torch.abs(emb1 - emb2)).item()
    assert diff > 1e-6, "Different inputs produced identical embeddings"

def test_siamese_parameter_count():
    """Model should have a reasonable number of parameters."""
    model = SiameseEGNN(
        node_dim=64, edge_dim=0, hidden_dim=128,
        embed_dim=256, num_layers=4, geom_dim=64,
        num_rbf=16, dropout=0.1
    )
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params > 50_000, f"Too few params: {total_params}"
    assert total_params < 5_000_000, f"Too many params: {total_params}"
    print(f"  Model has {total_params:,} parameters")

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
