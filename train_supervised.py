"""
Phase 15: Targeted Alignment (Supervised Fine-Tuning)

This script implements Supervised Contrastive Learning using Triplet Loss.
It fine-tunes the model to specifically distinguish known true pairs from hard decoys.

Strategy:
- Anchor: Viral protein patch (e.g., 2V5I)
- Positive: Known Human Homolog patch (e.g., 1LB5)
- Negative: Known Decoy patch (e.g., 1UBQ)
- Loss: TripletMarginLoss
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, Batch

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

# ============================================================================
# Configuration
# ============================================================================

POSITIVE_DIR = 'data/benchmark/positive'
NEGATIVE_DIR = 'data/benchmark/negative'
PRETRAINED_WEIGHTS = 'models/geomimic_net_weights_final.pth'
SAVE_PATH = 'models/geomimic_net_weights_supervised.pth'

# Known True Pairs (Virus, Human)
TRUE_PAIRS = [
    ('1Q59', '1G5M'),  # EBV BHRF1 -> Bcl-2
    ('2V5I', '1LB5'),  # Vaccinia A52 -> TRAF6
    ('3CL3', '3H11'),  # KSHV vFLIP -> FLIP
    ('2GX9', '1KX5'),  # Flu NS1 -> Histone H3
]

# Negative Decoys
NEGATIVE_PDBS = ['1A3N', '1TRZ', '1MBN', '1UBQ']


# ============================================================================
# Utilities (Copied from train.py for self-containment)
# ============================================================================

def random_rotation_matrix():
    random_matrix = torch.randn(3, 3)
    q, r = torch.linalg.qr(random_matrix)
    d = torch.diag(torch.sign(torch.diag(r)))
    q = q @ d
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q

def rotate_coordinates(pos, rotation_matrix=None):
    if rotation_matrix is None:
        rotation_matrix = random_rotation_matrix()
    center = pos.mean(dim=0)
    pos_centered = pos - center
    pos_rotated = pos_centered @ rotation_matrix.T
    return pos_rotated + center

def extract_patch(data, center_idx, k_hops=2):
    edge_index = data.edge_index
    visited = {center_idx}
    frontier = {center_idx}
    for _ in range(k_hops):
        new_frontier = set()
        for node in frontier:
            mask_src = edge_index[0] == node
            mask_tgt = edge_index[1] == node
            neighbors = torch.cat([edge_index[1, mask_src], edge_index[0, mask_tgt]])
            for n in neighbors.tolist():
                if n not in visited:
                    new_frontier.add(n)
                    visited.add(n)
        frontier = new_frontier
        if not frontier: break
    
    patch_nodes = sorted(list(visited))
    if len(patch_nodes) < 5: return None
    
    node_map = {old: new for new, old in enumerate(patch_nodes)}
    patch_x = data.x[patch_nodes]
    patch_pos = data.pos[patch_nodes]
    
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        if src in visited and tgt in visited:
            edge_mask[i] = True
            
    patch_edges = edge_index[:, edge_mask]
    new_src = torch.tensor([node_map[s.item()] for s in patch_edges[0]])
    new_tgt = torch.tensor([node_map[t.item()] for t in patch_edges[1]])
    patch_edge_index = torch.stack([new_src, new_tgt])
    
    return Data(x=patch_x, pos=patch_pos, edge_index=patch_edge_index)

def create_augmented_pair(patch):
    """Returns original patch and rotated patch."""
    # This implementation is simplified for Triplet Loss augmentation
    # We return the original and a rotated version
    rotated_pos = rotate_coordinates(patch.pos)
    rotated_patch = Data(x=patch.x, pos=rotated_pos, edge_index=patch.edge_index)
    return patch, rotated_patch


# ============================================================================
# Data Loading
# ============================================================================

def load_pdb_data(pdb_id, directory):
    """Load PDB file and return PyG Data object."""
    path = os.path.join(directory, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        print(f"  [ERROR] {pdb_id}.pdb not found in {directory}")
        return None
    try:
        data = parse_pdb_to_pyg(path, use_esm=True)
        return data
    except Exception as e:
        print(f"  [ERROR] Loading {pdb_id}: {e}")
        return None

def load_all_graphs():
    """Load all necessary protein graphs into memory."""
    print("Loading protein graphs...")
    graphs = {}
    
    # Load Viral and True Human
    for viral_id, human_id in TRUE_PAIRS:
        # Load Viral
        if viral_id not in graphs:
            g = load_pdb_data(viral_id, POSITIVE_DIR)
            if g: graphs[viral_id] = g
            
        # Load Human
        if human_id not in graphs:
            g = load_pdb_data(human_id, POSITIVE_DIR)
            if g: graphs[human_id] = g
            
    # Load Negatives
    for neg_id in NEGATIVE_PDBS:
        if neg_id not in graphs:
            g = load_pdb_data(neg_id, NEGATIVE_DIR)
            if g: graphs[neg_id] = g
            
    print(f"  Loaded {len(graphs)} proteins.")
    return graphs


# ============================================================================
# Triplet Batch Generation
# ============================================================================

def create_triplet_batch(graphs, batch_size=32, k_hops=2):
    """
    Create a batch of triplets (Anchor, Positive, Negative).
    """
    anchors = []
    positives = []
    negatives = []
    
    # We want to create 'batch_size' triplets
    count = 0
    attempts = 0
    max_attempts = batch_size * 5
    
    while count < batch_size and attempts < max_attempts:
        attempts += 1
        
        # 1. Select a random True Pair
        viral_id, human_id = random.choice(TRUE_PAIRS)
        
        # 2. Select a random Negative protein
        neg_id = random.choice(NEGATIVE_PDBS)
        
        # Ensure graphs exist
        if viral_id not in graphs or human_id not in graphs or neg_id not in graphs:
            continue
            
        viral_data = graphs[viral_id]
        human_data = graphs[human_id]
        neg_data = graphs[neg_id]
        
        # 3. Extract random patches
        # Anchor (Virus)
        center_v = random.randint(0, viral_data.x.size(0) - 1)
        anchor = extract_patch(viral_data, center_v, k_hops)
        
        # Positive (Human)
        center_h = random.randint(0, human_data.x.size(0) - 1)
        positive = extract_patch(human_data, center_h, k_hops)
        
        # Negative (Decoy)
        center_n = random.randint(0, neg_data.x.size(0) - 1)
        negative = extract_patch(neg_data, center_n, k_hops)
        
        if anchor is not None and positive is not None and negative is not None:
            # Augment: Apply random rotation to all of them to prevent overfitting
            # We use the rotated version as the sample for robustness
            _, anchor_aug = create_augmented_pair(anchor)
            _, positive_aug = create_augmented_pair(positive)
            _, negative_aug = create_augmented_pair(negative)
            
            anchors.append(anchor_aug)
            positives.append(positive_aug)
            negatives.append(negative_aug)
            count += 1
    
    if not anchors:
        return None, None, None
        
    # Collate
    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    negative_batch = Batch.from_data_list(negatives)
    
    return anchor_batch, positive_batch, negative_batch


# ============================================================================
# Training
# ============================================================================

def train_supervised(num_epochs=50, batch_size=32, learning_rate=1e-4, 
                     margin=1.0, device='cpu'):
    print("=" * 60)
    print("Phase 15: Supervised Fine-Tuning (Triplet Loss)")
    print("=" * 60)
    
    # Load Data
    graphs = load_all_graphs()
    
    # Initialize Model
    print("\nInitializing model...")
    model = SiameseEGNN(
        node_dim=32,
        edge_dim=0,
        hidden_dim=64,
        embed_dim=128,
        num_layers=4,
        geom_dim=32
    ).to(device)
    
    # Load Pretrained Weights
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"  Loading pretrained weights: {PRETRAINED_WEIGHTS}")
        state_dict = torch.load(PRETRAINED_WEIGHTS, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"  [WARNING] Pretrained weights not found at {PRETRAINED_WEIGHTS}")
        print("  Proceeding with random weights (not recommended for fine-tuning).")
        
    # Optimizer & Loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    model.train()
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 10  # Fixed number of batches per epoch
        
        for _ in range(num_batches):
            a_batch, p_batch, n_batch = create_triplet_batch(graphs, batch_size=batch_size)
            
            if a_batch is None: continue
            
            a_batch = a_batch.to(device)
            p_batch = p_batch.to(device)
            n_batch = n_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Note: We use forward_one to get embeddings
            z_a = model.forward_one(a_batch)
            z_p = model.forward_one(p_batch)
            z_n = model.forward_one(n_batch)
            
            # Triplet Loss: max(d(a,p) - d(a,n) + margin, 0)
            loss = triplet_loss(z_a, z_p, z_n)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = " *"
            # Save best model immediately
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            marker = ""
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Triplet Loss: {avg_loss:.4f}{marker}")
            
    print("-" * 60)
    print(f"Training complete! Best Loss: {best_loss:.4f}")
    if best_loss == float('inf'):
         print("  Warning: Loss never converged.")
    else:
         print(f"Saved weights to {SAVE_PATH}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train_supervised(num_epochs=50, device=device)
