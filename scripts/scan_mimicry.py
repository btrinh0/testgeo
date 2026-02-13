"""
Mimicry Scanning Script
Detects structural mimicry between viral and human proteins using a Siamese EGNN.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import EGNN

# === Configuration ===
VIRAL_PT = 'data/processed/6H3X.pt'
HUMAN_PT = 'data/processed/6P5R.pt'
PATCH_RADIUS = 16.0  # Angstroms (Phase 14: increased from 10.0 for wider context)
NODE_DIM = 16
EDGE_DIM = 0
HIDDEN_DIM = 32
NUM_LAYERS = 2
OUTPUT_FILE = 'results/mimicry_site.txt'


def load_data(path):
    """Load a processed .pt file."""
    return torch.load(path, weights_only=False)


def extract_patches(data, radius):
    """
    Extract overlapping patches from the protein.
    Each patch is centered on an atom and includes all atoms within `radius` Angstroms.
    
    Returns:
        patches: List of (sub_data, center_idx) tuples
    """
    pos = data.pos  # [N, 3]
    x = data.x      # [N, 1]
    
    patches = []
    num_atoms = pos.size(0)
    
    # For each atom, find neighbors within radius
    for i in range(num_atoms):
        center = pos[i]
        dists = torch.norm(pos - center, dim=1)
        mask = dists <= radius
        
        if mask.sum() < 3:
            # Skip patches with too few atoms
            continue
        
        # Extract local patch
        local_indices = torch.where(mask)[0]
        local_pos = pos[local_indices]
        local_x = x[local_indices]
        
        # Center the coordinates (translation invariance)
        local_pos = local_pos - center
        
        # Build local edges (k-NN within patch)
        k = min(10, len(local_indices) - 1)
        if k <= 0:
            continue
            
        dist_matrix = torch.cdist(local_pos, local_pos)
        _, neighbors = dist_matrix.topk(k + 1, largest=False)
        neighbors = neighbors[:, 1:]  # Remove self
        
        num_local = local_pos.size(0)
        target = torch.arange(num_local).repeat_interleave(k)
        source = neighbors.flatten()
        edge_index = torch.stack([source, target], dim=0)
        
        patches.append({
            'pos': local_pos,
            'x': local_x,
            'edge_index': edge_index,
            'center_idx': i,
            'local_indices': local_indices
        })
    
    return patches


class SiameseEGNN(nn.Module):
    """Siamese EGNN for generating geometric embeddings."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(100, node_dim)
        self.egnn = EGNN(node_dim, edge_dim, hidden_dim, num_layers)
        self.pool = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, pos, edge_index):
        """
        Args:
            x: Atomic numbers [N, 1]
            pos: Coordinates [N, 3]
            edge_index: Graph connectivity [2, E]
        
        Returns:
            embedding: Geometric embedding vector [hidden_dim]
        """
        h = self.embedding(x.squeeze(-1).long())
        h_out, _ = self.egnn(h, pos, edge_index)
        
        # Global mean pooling
        patch_embedding = h_out.mean(dim=0)  # [node_dim]
        
        # Project to embedding space
        embedding = self.pool(patch_embedding)
        
        return embedding


def embed_patches(patches, model):
    """Generate embeddings for all patches using the model."""
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for patch in patches:
            emb = model(patch['x'], patch['pos'], patch['edge_index'])
            embeddings.append(emb)
    
    return torch.stack(embeddings)  # [num_patches, hidden_dim]


def find_top_match(viral_embeddings, human_embeddings, viral_patches, human_patches):
    """
    Find the best matching pair based on cosine similarity.
    
    Returns:
        best_viral_idx, best_human_idx, best_similarity
    """
    # Normalize embeddings for cosine similarity
    viral_norm = F.normalize(viral_embeddings, dim=1)
    human_norm = F.normalize(human_embeddings, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(viral_norm, human_norm.T)  # [V, H]
    
    # Find maximum
    max_val, max_idx = similarity_matrix.flatten().max(dim=0)
    viral_idx = max_idx // similarity_matrix.size(1)
    human_idx = max_idx % similarity_matrix.size(1)
    
    return viral_idx.item(), human_idx.item(), max_val.item()


def get_residue_ids(patch, original_data):
    """
    Extract residue IDs from a patch.
    Since we don't have residue info in .pt files, we return atom indices.
    """
    return patch['local_indices'].tolist()


def main():
    print("=== Mimicry Scanning ===\n")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("Loading processed data...")
    viral_data = load_data(VIRAL_PT)
    human_data = load_data(HUMAN_PT)
    print(f"  Viral: {viral_data.pos.size(0)} atoms")
    print(f"  Human: {human_data.pos.size(0)} atoms")
    
    # Extract patches
    print("\nExtracting patches (radius=10 Å)...")
    viral_patches = extract_patches(viral_data, PATCH_RADIUS)
    human_patches = extract_patches(human_data, PATCH_RADIUS)
    print(f"  Viral patches: {len(viral_patches)}")
    print(f"  Human patches: {len(human_patches)}")
    
    # Initialize model
    print("\nInitializing Siamese EGNN...")
    model = SiameseEGNN(NODE_DIM, EDGE_DIM, HIDDEN_DIM, NUM_LAYERS)
    
    # Load trained weights
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'geomimic_net_weights.pth')
    try:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"  Loaded trained weights from {weights_path}")
    except FileNotFoundError:
        print("  WARNING: Using random weights! Run train.py first.")
    except Exception as e:
        print(f"  WARNING: Could not load weights ({e}). Using random weights!")
    
    # Set model to evaluation mode before scanning
    model.eval()

    # Generate embeddings
    print("Generating geometric embeddings...")
    viral_embeddings = embed_patches(viral_patches, model)
    human_embeddings = embed_patches(human_patches, model)
    print(f"  Viral embeddings: {viral_embeddings.shape}")
    print(f"  Human embeddings: {human_embeddings.shape}")
    
    # Find top match
    print("\nFinding top mimicry match...")
    viral_idx, human_idx, similarity = find_top_match(
        viral_embeddings, human_embeddings, viral_patches, human_patches
    )
    
    # Get residue/atom IDs
    viral_residues = get_residue_ids(viral_patches[viral_idx], viral_data)
    
    print(f"\n=== TOP MATCH FOUND ===")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Viral Patch Center Atom: {viral_patches[viral_idx]['center_idx']}")
    print(f"Human Patch Center Atom: {human_patches[human_idx]['center_idx']}")
    print(f"Viral Patch Atom Indices: {viral_residues[:10]}..." if len(viral_residues) > 10 else f"Viral Patch Atom Indices: {viral_residues}")
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"# Mimicry Site Detection Results\n")
        f.write(f"# Viral: 6H3X, Human: 6P5R\n")
        f.write(f"# Patch Radius: {PATCH_RADIUS} Angstroms\n")
        f.write(f"# Cosine Similarity: {similarity:.4f}\n\n")
        f.write(f"# Viral Patch Center Atom Index: {viral_patches[viral_idx]['center_idx']}\n")
        f.write(f"# Human Patch Center Atom Index: {human_patches[human_idx]['center_idx']}\n\n")
        f.write("# Viral Patch Atom Indices (potential mimicry site):\n")
        for idx in viral_residues:
            f.write(f"{idx}\n")
    
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
