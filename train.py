"""
Phase 7: Training Implementation
Contrastive training script for geometric mimicry detection using SiameseEGNN.

This script:
1. Loads protein graphs from processed data
2. Extracts patches (local subgraphs) from each protein
3. Creates positive pairs via 3D rotation augmentation
4. Creates negative pairs from different proteins/patches
5. Trains using InfoNCE contrastive loss
6. Saves final weights to models/geomimic_net_weights.pth
"""

import os
import sys
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, Batch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from models.egnn import SiameseEGNN


# ============================================================================
# Utility Functions
# ============================================================================

def random_rotation_matrix():
    """
    Generate a random 3D rotation matrix using QR decomposition.
    This ensures uniform distribution over SO(3).
    """
    # Random matrix
    random_matrix = torch.randn(3, 3)
    # QR decomposition gives orthogonal Q
    q, r = torch.linalg.qr(random_matrix)
    # Ensure proper rotation (det = 1, not -1)
    d = torch.diag(torch.sign(torch.diag(r)))
    q = q @ d
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotate_coordinates(pos, rotation_matrix=None):
    """
    Apply a random 3D rotation to coordinates.
    
    Args:
        pos: Coordinates tensor [N, 3]
        rotation_matrix: Optional rotation matrix [3, 3]
        
    Returns:
        Rotated coordinates [N, 3]
    """
    if rotation_matrix is None:
        rotation_matrix = random_rotation_matrix()
    
    # Center coordinates
    center = pos.mean(dim=0)
    pos_centered = pos - center
    
    # Apply rotation
    pos_rotated = pos_centered @ rotation_matrix.T
    
    # Restore center
    pos_rotated = pos_rotated + center
    
    return pos_rotated


def extract_patch(data, center_idx, k_hops=2):
    """
    Extract a local patch (k-hop neighborhood) around a center node.
    
    Args:
        data: PyG Data object with x, pos, edge_index
        center_idx: Index of the center node
        k_hops: Number of hops to include
        
    Returns:
        PyG Data object representing the patch
    """
    edge_index = data.edge_index
    num_nodes = data.x.size(0)
    
    # Find k-hop neighbors using BFS
    visited = {center_idx}
    frontier = {center_idx}
    
    for _ in range(k_hops):
        new_frontier = set()
        for node in frontier:
            # Find neighbors (both directions in edge_index)
            mask_src = edge_index[0] == node
            mask_tgt = edge_index[1] == node
            neighbors = torch.cat([edge_index[1, mask_src], edge_index[0, mask_tgt]])
            for n in neighbors.tolist():
                if n not in visited:
                    new_frontier.add(n)
                    visited.add(n)
        frontier = new_frontier
        if not frontier:
            break
    
    # Get node indices in the patch
    patch_nodes = sorted(list(visited))
    if len(patch_nodes) < 5:  # Skip very small patches
        return None
    
    # Create mapping from old to new indices
    node_map = {old: new for new, old in enumerate(patch_nodes)}
    
    # Extract node features and positions
    patch_x = data.x[patch_nodes]
    patch_pos = data.pos[patch_nodes]
    
    # Extract and remap edges within patch
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        if src in visited and tgt in visited:
            edge_mask[i] = True
    
    patch_edges = edge_index[:, edge_mask]
    # Remap indices
    new_src = torch.tensor([node_map[s.item()] for s in patch_edges[0]])
    new_tgt = torch.tensor([node_map[t.item()] for t in patch_edges[1]])
    patch_edge_index = torch.stack([new_src, new_tgt])
    
    return Data(x=patch_x, pos=patch_pos, edge_index=patch_edge_index)


def create_augmented_pair(patch):
    """
    Create a positive pair by augmenting a patch with random rotation.
    
    Args:
        patch: PyG Data object
        
    Returns:
        (original_patch, rotated_patch) - a positive pair
    """
    # Clone the patch
    rotated_patch = Data(
        x=patch.x.clone(),
        pos=rotate_coordinates(patch.pos.clone()),
        edge_index=patch.edge_index.clone()
    )
    
    return patch, rotated_patch


# ============================================================================
# InfoNCE Loss
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss.
    
    Given embeddings of positive pairs and negatives, computes:
    L = -log(exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ))
    
    where z_i, z_j are positive pairs and z_k includes all negatives.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_anchor, z_positive, z_negatives=None):
        """
        Compute InfoNCE loss.
        
        Args:
            z_anchor: Anchor embeddings [B, D]
            z_positive: Positive embeddings [B, D]
            z_negatives: Negative embeddings [B, N, D] or None
                        If None, uses other samples in batch as negatives
                        
        Returns:
            Scalar loss value
        """
        batch_size = z_anchor.size(0)
        
        # Compute positive similarity
        pos_sim = F.cosine_similarity(z_anchor, z_positive, dim=-1)  # [B]
        pos_sim = pos_sim / self.temperature
        
        if z_negatives is None:
            # Use other samples in batch as negatives (NT-Xent style)
            # Similarity matrix between all anchors and all positives
            # z_anchor @ z_positive.T gives [B, B] similarities
            all_sim = torch.mm(z_anchor, z_positive.T) / self.temperature  # [B, B]
            
            # For each anchor i, positive is at index i, rest are negatives
            # LogSumExp over all comparisons
            labels = torch.arange(batch_size, device=z_anchor.device)
            loss = F.cross_entropy(all_sim, labels)
        else:
            # Explicit negatives provided
            num_negatives = z_negatives.size(1)
            
            # Compute negative similarities
            # z_anchor: [B, D], z_negatives: [B, N, D]
            neg_sim = torch.bmm(z_negatives, z_anchor.unsqueeze(-1)).squeeze(-1)  # [B, N]
            neg_sim = neg_sim / self.temperature
            
            # Concatenate positive and negative scores
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
            
            # Labels: positive is always at index 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=z_anchor.device)
            loss = F.cross_entropy(logits, labels)
        
        return loss


# ============================================================================
# Training Functions
# ============================================================================

def load_protein_data(processed_dir):
    """Load processed protein graphs."""
    virus_path = os.path.join(processed_dir, '6H3X.pt')
    human_path = os.path.join(processed_dir, '6P5R.pt')
    
    print(f"Loading virus protein from {virus_path}...")
    virus_data = torch.load(virus_path, weights_only=False)
    print(f"  Nodes: {virus_data.x.size(0)}, Edges: {virus_data.edge_index.size(1)}")
    
    print(f"Loading human protein from {human_path}...")
    human_data = torch.load(human_path, weights_only=False)
    print(f"  Nodes: {human_data.x.size(0)}, Edges: {human_data.edge_index.size(1)}")
    
    return virus_data, human_data


def extract_all_patches(data, num_patches=50, k_hops=2, min_nodes=10):
    """
    Extract multiple patches from a protein graph.
    
    Args:
        data: PyG Data object
        num_patches: Number of patches to extract
        k_hops: Number of hops for each patch
        min_nodes: Minimum nodes in a valid patch
        
    Returns:
        List of patch Data objects
    """
    num_nodes = data.x.size(0)
    patches = []
    
    # Randomly sample center nodes
    center_indices = random.sample(range(num_nodes), min(num_patches * 2, num_nodes))
    
    for center_idx in center_indices:
        if len(patches) >= num_patches:
            break
            
        patch = extract_patch(data, center_idx, k_hops)
        if patch is not None and patch.x.size(0) >= min_nodes:
            patches.append(patch)
    
    return patches


def create_training_batch(virus_patches, human_patches, batch_size=32):
    """
    Create a training batch with positive and negative pairs.
    
    Strategy:
    - Positive pairs: Same patch with different rotations
    - Negative pairs: Use in-batch negatives (other samples in batch)
    
    Args:
        virus_patches: List of virus patches
        human_patches: List of human patches
        batch_size: Number of pairs per batch
        
    Returns:
        (anchor_batch, positive_batch) as PyG Batch objects
    """
    all_patches = virus_patches + human_patches
    
    if len(all_patches) < batch_size:
        batch_size = len(all_patches)
    
    # Sample patches for this batch
    selected = random.sample(all_patches, batch_size)
    
    anchors = []
    positives = []
    
    for patch in selected:
        anchor, positive = create_augmented_pair(patch)
        anchors.append(anchor)
        positives.append(positive)
    
    # Batch the graphs
    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    
    return anchor_batch, positive_batch


def train_epoch(model, virus_patches, human_patches, optimizer, criterion, 
                batch_size=32, num_batches=10, device='cpu'):
    """
    Train for one epoch.
    
    Args:
        model: SiameseEGNN model
        virus_patches: List of virus protein patches
        human_patches: List of human protein patches
        optimizer: PyTorch optimizer
        criterion: InfoNCE loss
        batch_size: Pairs per batch
        num_batches: Number of batches per epoch
        device: Training device
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx in range(num_batches):
        # Create batch
        anchor_batch, positive_batch = create_training_batch(
            virus_patches, human_patches, batch_size
        )
        
        # Move to device
        anchor_batch = anchor_batch.to(device)
        positive_batch = positive_batch.to(device)
        
        # Forward pass
        z_anchor = model.forward_one(anchor_batch)
        z_positive = model.forward_one(positive_batch)
        
        # Compute loss (using in-batch negatives)
        loss = criterion(z_anchor, z_positive)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def train(num_epochs=100, batch_size=32, learning_rate=1e-3, 
          hidden_dim=64, embed_dim=128, num_layers=4,
          device='cpu'):
    """
    Main training function.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Pairs per batch
        learning_rate: Adam learning rate
        hidden_dim: EGNN hidden dimension
        embed_dim: Output embedding dimension
        num_layers: Number of EGNN layers
        device: Training device
    """
    print("=" * 60)
    print("Phase 7: Contrastive Training for Geometric Mimicry Detection")
    print("=" * 60)
    
    # Load data
    processed_dir = os.path.join('data', 'processed')
    virus_data, human_data = load_protein_data(processed_dir)
    
    # Extract patches
    print("\nExtracting patches from proteins...")
    virus_patches = extract_all_patches(virus_data, num_patches=100, k_hops=2)
    human_patches = extract_all_patches(human_data, num_patches=100, k_hops=2)
    print(f"  Virus patches: {len(virus_patches)}")
    print(f"  Human patches: {len(human_patches)}")
    
    if len(virus_patches) < 10 or len(human_patches) < 10:
        print("WARNING: Too few patches extracted. Reducing k_hops...")
        virus_patches = extract_all_patches(virus_data, num_patches=100, k_hops=3, min_nodes=5)
        human_patches = extract_all_patches(human_data, num_patches=100, k_hops=3, min_nodes=5)
        print(f"  Virus patches: {len(virus_patches)}")
        print(f"  Human patches: {len(human_patches)}")
    
    # Initialize model
    print("\nInitializing SiameseEGNN model...")
    model = SiameseEGNN(
        node_dim=32,   # Must match geom_dim for cross-attention
        edge_dim=0,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        geom_dim=32    # Cross-attention dimension
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    num_batches_per_epoch = max(10, (len(virus_patches) + len(human_patches)) // batch_size)
    
    for epoch in range(1, num_epochs + 1):
        # Re-extract patches every 20 epochs for variety
        if epoch > 1 and epoch % 20 == 0:
            virus_patches = extract_all_patches(virus_data, num_patches=100, k_hops=2)
            human_patches = extract_all_patches(human_data, num_patches=100, k_hops=2)
        
        avg_loss = train_epoch(
            model, virus_patches, human_patches,
            optimizer, criterion,
            batch_size=batch_size,
            num_batches=num_batches_per_epoch,
            device=device
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_marker = " *"
        else:
            best_marker = ""
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f}{best_marker}")
    
    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    
    # Save model weights
    os.makedirs('models', exist_ok=True)
    save_path = os.path.join('models', 'geomimic_net_weights.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to: {save_path}")
    
    # Verify save
    file_size = os.path.getsize(save_path) / 1024
    print(f"  File size: {file_size:.1f} KB")
    
    return model


# ============================================================================
# Phase 13: Hard Negative Mining Training
# ============================================================================

def load_hard_negatives(negatives_dir='data/processed_negatives', num_patches=50, k_hops=2, min_nodes=10):
    """
    Load hard negative samples from processed decoy proteins.
    
    Hard negatives are patches from unrelated proteins (Ubiquitin, Myoglobin, etc.)
    that produce high similarity scores but should be classified as non-matches.
    
    Args:
        negatives_dir: Directory containing processed negative .pt files
        num_patches: Number of patches to extract per protein
        k_hops: Number of hops for patch extraction
        min_nodes: Minimum nodes in a valid patch
        
    Returns:
        List of patch Data objects from hard negative proteins
    """
    hard_negative_patches = []
    
    if not os.path.exists(negatives_dir):
        print(f"  Warning: {negatives_dir} does not exist. No hard negatives loaded.")
        return hard_negative_patches
    
    pt_files = [f for f in os.listdir(negatives_dir) if f.endswith('.pt')]
    
    if not pt_files:
        print(f"  Warning: No .pt files found in {negatives_dir}")
        return hard_negative_patches
    
    print(f"Loading hard negatives from {negatives_dir}...")
    
    for pt_file in pt_files:
        pt_path = os.path.join(negatives_dir, pt_file)
        try:
            data = torch.load(pt_path, weights_only=False)
            patches = extract_all_patches(data, num_patches=num_patches, k_hops=k_hops, min_nodes=min_nodes)
            hard_negative_patches.extend(patches)
            print(f"  {pt_file}: {len(patches)} patches")
        except Exception as e:
            print(f"  Error loading {pt_file}: {e}")
    
    print(f"  Total hard negative patches: {len(hard_negative_patches)}")
    return hard_negative_patches


def create_training_batch_with_hard_negatives(virus_patches, human_patches, hard_negatives, 
                                               batch_size=32, hard_negative_ratio=0.5):
    """
    Create a training batch with positive pairs and a mix of random/hard negatives.
    
    Args:
        virus_patches: List of virus patches
        human_patches: List of human patches
        hard_negatives: List of hard negative patches
        batch_size: Number of pairs per batch
        hard_negative_ratio: Fraction of negatives that should be "hard"
        
    Returns:
        (anchor_batch, positive_batch) as PyG Batch objects
    """
    all_patches = virus_patches + human_patches
    
    if len(all_patches) < batch_size:
        batch_size = len(all_patches)
    
    # Sample patches for this batch
    selected = random.sample(all_patches, batch_size)
    
    anchors = []
    positives = []
    
    for patch in selected:
        r = random.random()
        
        if r < hard_negative_ratio and hard_negatives:
            # Use a hard negative as the "positive" (will have high similarity but wrong class)
            # The InfoNCE loss will see high similarity for wrong pairs and generate large gradients
            # Actually, we keep anchor/positive as rotation pairs for the loss
            anchor, positive = create_augmented_pair(patch)
        else:
            # Standard: same patch with different rotations
            anchor, positive = create_augmented_pair(patch)
        
        anchors.append(anchor)
        positives.append(positive)
    
    # For hard negative mining, we also need to include hard negatives in the batch
    # so they become in-batch negatives in the InfoNCE loss
    if hard_negatives and hard_negative_ratio > 0:
        num_hard = int(batch_size * hard_negative_ratio)
        if num_hard > 0 and len(hard_negatives) > 0:
            hard_samples = random.sample(hard_negatives, min(num_hard, len(hard_negatives)))
            for hard_patch in hard_samples:
                # Add hard negative as an anchor with its rotated version
                anchor, positive = create_augmented_pair(hard_patch)
                anchors.append(anchor)
                positives.append(positive)
    
    # Batch the graphs
    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    
    return anchor_batch, positive_batch


def train_epoch_with_hard_negatives(model, virus_patches, human_patches, hard_negatives,
                                     optimizer, criterion, batch_size=32, num_batches=10, 
                                     device='cpu', hard_negative_ratio=0.5, penalty_scale=1.0):
    """
    Train for one epoch with hard negative mining.
    
    Args:
        penalty_scale: Multiplier for loss when hard negatives are present (default 1.0).
                       Set to 2.0 to make model "feel more pain" on hard negative mistakes.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx in range(num_batches):
        # Create batch with hard negatives mixed in
        anchor_batch, positive_batch = create_training_batch_with_hard_negatives(
            virus_patches, human_patches, hard_negatives, 
            batch_size, hard_negative_ratio
        )
        
        # Move to device
        anchor_batch = anchor_batch.to(device)
        positive_batch = positive_batch.to(device)
        
        # Forward pass
        z_anchor = model.forward_one(anchor_batch)
        z_positive = model.forward_one(positive_batch)
        
        # Compute loss (using in-batch negatives including hard negatives)
        loss = criterion(z_anchor, z_positive)
        
        # Apply penalty scaling when hard negatives are present
        # This makes the model "feel more pain" for confusing virus with Ubiquitin
        if hard_negatives and hard_negative_ratio > 0:
            loss = loss * penalty_scale
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def train_with_hard_negatives(num_epochs=30, batch_size=32, learning_rate=1e-3,
                               hidden_dim=64, embed_dim=128, num_layers=4,
                               device='cpu', hard_negative_ratio=0.5,
                               penalty_scale=1.0,
                               save_path='models/geomimic_net_weights_v3.pth'):
    """
    Phase 13B: Training with Hard Negative Mining.
    
    This training uses adversarial examples (Ubiquitin, Myoglobin, etc.) that 
    produce high similarity scores but should be classified as negatives.
    The InfoNCE loss sees high similarity for these hard negatives and generates
    large gradients, forcing the model to learn subtle distinguishing features.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Pairs per batch
        learning_rate: Adam learning rate
        hidden_dim: EGNN hidden dimension
        embed_dim: Output embedding dimension
        num_layers: Number of EGNN layers
        device: Training device
        hard_negative_ratio: Fraction of negatives that are "hard"
        save_path: Where to save the trained weights
    """
    print("=" * 60)
    print("Phase 13B: Hard Negative Mining Training")
    print("=" * 60)
    print(f"Hard Negative Ratio: {hard_negative_ratio:.0%}")
    print(f"Penalty Scale: {penalty_scale}x")
    
    # Load regular training data
    processed_dir = os.path.join('data', 'processed')
    virus_data, human_data = load_protein_data(processed_dir)
    
    # Extract patches from regular data
    print("\nExtracting patches from proteins...")
    virus_patches = extract_all_patches(virus_data, num_patches=100, k_hops=2)
    human_patches = extract_all_patches(human_data, num_patches=100, k_hops=2)
    print(f"  Virus patches: {len(virus_patches)}")
    print(f"  Human patches: {len(human_patches)}")
    
    # Load hard negatives
    hard_negatives = load_hard_negatives('data/processed_negatives', num_patches=50, k_hops=2)
    
    if not hard_negatives:
        print("WARNING: No hard negatives loaded! Running standard training.")
    
    # Initialize model
    print("\nInitializing SiameseEGNN model...")
    model = SiameseEGNN(
        node_dim=32,
        edge_dim=0,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        geom_dim=32
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    num_batches_per_epoch = max(10, (len(virus_patches) + len(human_patches)) // batch_size)
    
    for epoch in range(1, num_epochs + 1):
        # Re-extract patches every 10 epochs for variety
        if epoch > 1 and epoch % 10 == 0:
            virus_patches = extract_all_patches(virus_data, num_patches=100, k_hops=2)
            human_patches = extract_all_patches(human_data, num_patches=100, k_hops=2)
            hard_negatives = load_hard_negatives('data/processed_negatives', num_patches=50, k_hops=2)
        
        avg_loss = train_epoch_with_hard_negatives(
            model, virus_patches, human_patches, hard_negatives,
            optimizer, criterion,
            batch_size=batch_size,
            num_batches=num_batches_per_epoch,
            device=device,
            hard_negative_ratio=hard_negative_ratio,
            penalty_scale=penalty_scale
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_marker = " *"
        else:
            best_marker = ""
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f}{best_marker}")
    
    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    
    # Save model weights
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to: {save_path}")
    
    # Verify save
    file_size = os.path.getsize(save_path) / 1024
    print(f"  File size: {file_size:.1f} KB")
    
    return model


# ============================================================================
# Phase 14: Context Expansion Training
# ============================================================================

# Phase 14 Configuration
K_HOPS_WIDE = 3  # Increased from 2 for wider context (~16Å vs ~10Å)

def train_wide_context(num_epochs=30, batch_size=32, learning_rate=1e-4,
                       hidden_dim=64, embed_dim=128, num_layers=4,
                       device='cpu', 
                       pretrained_weights='models/geomimic_net_weights_final.pth',
                       save_path='models/geomimic_net_weights_wide.pth'):
    """
    Phase 14: Context Expansion Training with Transfer Learning.
    
    Loads pretrained weights and fine-tunes with wider patch context (k_hops=3).
    The wider context helps the model see more of the protein fold, potentially
    reducing false positives that occur due to small similar patches.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Pairs per batch
        learning_rate: Lower LR for fine-tuning (1e-4 instead of 1e-3)
        pretrained_weights: Path to pretrained weights for transfer learning
        save_path: Where to save the fine-tuned weights
    """
    print("=" * 60)
    print("Phase 14: Context Expansion Training")
    print("=" * 60)
    print(f"K-hops: {K_HOPS_WIDE} (wider context)")
    print(f"Transfer Learning from: {pretrained_weights}")
    
    # Load regular training data
    processed_dir = os.path.join('data', 'processed')
    virus_data, human_data = load_protein_data(processed_dir)
    
    # Extract patches with WIDER CONTEXT (k_hops=3)
    print(f"\nExtracting patches with k_hops={K_HOPS_WIDE}...")
    virus_patches = extract_all_patches(virus_data, num_patches=100, k_hops=K_HOPS_WIDE)
    human_patches = extract_all_patches(human_data, num_patches=100, k_hops=K_HOPS_WIDE)
    print(f"  Virus patches: {len(virus_patches)}")
    print(f"  Human patches: {len(human_patches)}")
    
    # Load hard negatives too (for continued discrimination)
    hard_negatives = load_hard_negatives('data/processed_negatives', num_patches=50, k_hops=K_HOPS_WIDE)
    
    # Initialize model
    print("\nInitializing SiameseEGNN model...")
    model = SiameseEGNN(
        node_dim=32,
        edge_dim=0,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        geom_dim=32
    ).to(device)
    
    # Transfer Learning: Load pretrained weights
    if os.path.exists(pretrained_weights):
        try:
            state_dict = torch.load(pretrained_weights, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"  [OK] Loaded pretrained weights from {pretrained_weights}")
        except Exception as e:
            print(f"  [ERROR] Loading weights: {e}")
            print("  Starting from scratch...")
    else:
        print(f"  [WARNING] {pretrained_weights} not found. Starting from scratch.")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer (lower learning rate for fine-tuning)
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    num_batches_per_epoch = max(10, (len(virus_patches) + len(human_patches)) // batch_size)
    
    for epoch in range(1, num_epochs + 1):
        # Re-extract patches every 10 epochs for variety
        if epoch > 1 and epoch % 10 == 0:
            virus_patches = extract_all_patches(virus_data, num_patches=100, k_hops=K_HOPS_WIDE)
            human_patches = extract_all_patches(human_data, num_patches=100, k_hops=K_HOPS_WIDE)
            hard_negatives = load_hard_negatives('data/processed_negatives', num_patches=50, k_hops=K_HOPS_WIDE)
        
        avg_loss = train_epoch_with_hard_negatives(
            model, virus_patches, human_patches, hard_negatives,
            optimizer, criterion,
            batch_size=batch_size,
            num_batches=num_batches_per_epoch,
            device=device,
            hard_negative_ratio=0.5,  # Moderate ratio for fine-tuning
            penalty_scale=1.5  # Moderate penalty
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_marker = " *"
        else:
            best_marker = ""
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f}{best_marker}")
    
    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    
    # Save model weights
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to: {save_path}")
    
    # Verify save
    file_size = os.path.getsize(save_path) / 1024
    print(f"  File size: {file_size:.1f} KB")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train the model
    model = train(
        num_epochs=50,  # Phase 10: 50 epochs for cross-attention learning
        batch_size=32,
        learning_rate=1e-3,
        hidden_dim=64,
        embed_dim=128,
        num_layers=4,
        device=device
    )
    
    print("\n" + "=" * 60)
    print("Phase 7 Complete!")
    print("=" * 60)
