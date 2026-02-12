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
RAW_DIR = 'data/raw'
PRETRAINED_WEIGHTS = 'models/geomimic_net_weights_final.pth'
SAVE_PATH = 'models/geomimic_net_weights_supervised.pth'

# Known True Pairs (Virus, Human) - 16 validated mimicry pairs
TRUE_PAIRS = [
    # Original 4
    ('1Q59', '1G5M'),  # EBV BHRF1 -> Bcl-2
    ('2V5I', '1LB5'),  # Vaccinia A52 -> TRAF6
    ('3CL3', '3H11'),  # KSHV vFLIP -> FLIP
    ('2GX9', '1KX5'),  # Flu NS1 -> Histone H3
    # Expanded 12
    ('2JBY', '1G5M'),  # Myxoma M11L -> Bcl-2
    ('1B4C', '1ITB'),  # Vaccinia B15 -> IL-1R
    ('1FV1', '1CDF'),  # EBV LMP1 -> CD40
    ('1H26', '1CF7'),  # Adenovirus E1A -> E2F
    ('1GUX', '1CF7'),  # HPV E7 -> E2F
    ('1EFN', '1SHF'),  # HIV Nef -> Fyn SH3
    ('3D2U', '1HHK'),  # CMV UL18 -> MHC-I HLA-A
    ('2UWI', '1EXT'),  # Cowpox CrmE -> TNFR1
    ('2BZR', '1MAZ'),  # KSHV vBcl-2 -> Bcl-xL
    ('2VGA', '1CA9'),  # Variola CrmB -> TNFR2
    ('1F5Q', '1B7T'),  # KSHV vCyclin -> Cyclin D2
    ('2BBR', '1A1W'),  # Molluscum MC159 -> FADD DED
]

# Negative Decoys - 10 unrelated proteins
NEGATIVE_PDBS = ['1A3N', '1TRZ', '1MBN', '1UBQ', '1LYZ', '1EMA', '4INS', '1CLL', '7RSA', '1HRC']

# All directories to search for PDB files
PDB_DIRS = [RAW_DIR, POSITIVE_DIR, NEGATIVE_DIR]


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

def load_pdb_data(pdb_id, directories=None):
    """Load PDB file from any of the given directories."""
    if directories is None:
        directories = PDB_DIRS
    
    for directory in directories:
        path = os.path.join(directory, f"{pdb_id}.pdb")
        if os.path.exists(path):
            try:
                data = parse_pdb_to_pyg(path, use_esm=True)
                return data
            except Exception as e:
                print(f"  [ERROR] Loading {pdb_id} from {directory}: {e}")
                return None
    
    print(f"  [ERROR] {pdb_id}.pdb not found in any directory")
    return None

def load_all_graphs():
    """Load all necessary protein graphs into memory."""
    print(f"Loading protein graphs ({len(TRUE_PAIRS)} pairs + {len(NEGATIVE_PDBS)} negatives)...")
    graphs = {}
    
    # Load Viral and True Human
    for viral_id, human_id in TRUE_PAIRS:
        if viral_id not in graphs:
            g = load_pdb_data(viral_id)
            if g: graphs[viral_id] = g
            
        if human_id not in graphs:
            g = load_pdb_data(human_id)
            if g: graphs[human_id] = g
            
    # Load Negatives
    for neg_id in NEGATIVE_PDBS:
        if neg_id not in graphs:
            g = load_pdb_data(neg_id)
            if g: graphs[neg_id] = g
            
    print(f"  Loaded {len(graphs)} proteins.")
    return graphs


# ============================================================================
# Pre-extract Patches (FAST - done once at startup)
# ============================================================================

def preextract_all_patches(graphs, num_patches_per_protein=50, k_hops=2):
    """
    Pre-extract patches from all proteins at startup.
    This is done ONCE and cached for fast batch sampling during training.
    """
    print("Pre-extracting patches (one-time operation)...")
    
    viral_patches = {}  # viral_id -> list of patches
    human_patches = {}  # human_id -> list of patches
    negative_patches = {}  # neg_id -> list of patches
    
    # Extract from viral and human proteins
    for viral_id, human_id in TRUE_PAIRS:
        if viral_id in graphs and viral_id not in viral_patches:
            patches = []
            data = graphs[viral_id]
            centers = random.sample(range(data.x.size(0)), min(num_patches_per_protein * 2, data.x.size(0)))
            for c in centers:
                p = extract_patch(data, c, k_hops)
                if p is not None and p.x.size(0) >= 5:
                    patches.append(p)
                if len(patches) >= num_patches_per_protein:
                    break
            viral_patches[viral_id] = patches
            print(f"  {viral_id}: {len(patches)} patches")
            
        if human_id in graphs and human_id not in human_patches:
            patches = []
            data = graphs[human_id]
            centers = random.sample(range(data.x.size(0)), min(num_patches_per_protein * 2, data.x.size(0)))
            for c in centers:
                p = extract_patch(data, c, k_hops)
                if p is not None and p.x.size(0) >= 5:
                    patches.append(p)
                if len(patches) >= num_patches_per_protein:
                    break
            human_patches[human_id] = patches
            print(f"  {human_id}: {len(patches)} patches")
    
    # Extract from negative proteins
    for neg_id in NEGATIVE_PDBS:
        if neg_id in graphs and neg_id not in negative_patches:
            patches = []
            data = graphs[neg_id]
            centers = random.sample(range(data.x.size(0)), min(num_patches_per_protein * 2, data.x.size(0)))
            for c in centers:
                p = extract_patch(data, c, k_hops)
                if p is not None and p.x.size(0) >= 5:
                    patches.append(p)
                if len(patches) >= num_patches_per_protein:
                    break
            negative_patches[neg_id] = patches
            print(f"  {neg_id}: {len(patches)} patches")
    
    return viral_patches, human_patches, negative_patches


# ============================================================================
# Fast Triplet Batch Generation (samples from pre-extracted patches)
# ============================================================================

def create_triplet_batch_fast(viral_patches, human_patches, negative_patches, batch_size=32):
    """
    Create a batch of triplets by sampling from PRE-EXTRACTED patches.
    
    Uses TWO types of negatives (50/50 mix):
    1. Random negatives: unrelated proteins (hemoglobin, ubiquitin, etc.)
    2. Cross-pair hard negatives: WRONG human targets for each viral protein
       e.g., anchor=EBV BHRF1, positive=Bcl-2, negative=TRAF6 (wrong target)
    
    This teaches the model to distinguish true mimicry from structural similarity
    to other human proteins.
    """
    anchors = []
    positives = []
    negatives = []
    
    # Build lookup: viral_id -> correct human_id
    pair_lookup = {}
    for v, h in TRUE_PAIRS:
        pair_lookup.setdefault(v, set()).add(h)
    
    # All human IDs for cross-pair negatives
    all_human_ids = list(human_patches.keys())
    
    for i in range(batch_size):
        # 1. Select a random True Pair
        viral_id, human_id = random.choice(TRUE_PAIRS)
        
        # Check patches exist
        if viral_id not in viral_patches or human_id not in human_patches:
            continue
        if not viral_patches[viral_id] or not human_patches[human_id]:
            continue
        
        # 2. Select negative: 50% cross-pair hard negative, 50% random negative
        use_hard_negative = (random.random() < 0.5) and len(all_human_ids) > 1
        
        if use_hard_negative:
            # Pick a WRONG human target (not the correct one for this viral)
            correct_humans = pair_lookup.get(viral_id, set())
            wrong_humans = [h for h in all_human_ids if h not in correct_humans and human_patches.get(h)]
            if wrong_humans:
                neg_id = random.choice(wrong_humans)
                neg_patch = random.choice(human_patches[neg_id])
            else:
                use_hard_negative = False
        
        if not use_hard_negative:
            # Random negative from unrelated proteins
            neg_id = random.choice(NEGATIVE_PDBS)
            if neg_id not in negative_patches or not negative_patches[neg_id]:
                continue
            neg_patch = random.choice(negative_patches[neg_id])
        
        # 3. Sample random patches (FAST - just indexing)
        anchor = random.choice(viral_patches[viral_id])
        positive = random.choice(human_patches[human_id])
        
        # 4. Apply random rotation for augmentation
        _, anchor_aug = create_augmented_pair(anchor)
        _, positive_aug = create_augmented_pair(positive)
        _, neg_aug = create_augmented_pair(neg_patch)
        
        anchors.append(anchor_aug)
        positives.append(positive_aug)
        negatives.append(neg_aug)
    
    if not anchors:
        return None, None, None
        
    # Collate into batches
    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    negative_batch = Batch.from_data_list(negatives)
    
    return anchor_batch, positive_batch, negative_batch


# ============================================================================
# Training
# ============================================================================

def train_supervised(num_epochs=100, batch_size=32, learning_rate=1e-4, 
                     margin=1.0, device='cpu'):
    print("=" * 60)
    print("Phase 22: Supervised Fine-Tuning (Cross-Pair Hard Negatives)")
    print("=" * 60)
    
    # Load Data
    graphs = load_all_graphs()
    
    # PRE-EXTRACT patches once (this is the slow part, done only once)
    viral_patches, human_patches, negative_patches = preextract_all_patches(graphs, num_patches_per_protein=100)
    
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
        
    # Optimizer with weight decay (L2 regularization) to prevent overfitting
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    model.train()
    print(f"\nStarting training for {num_epochs} epochs (with early stopping)...")
    print(f"  Using 50% cross-pair hard negatives + 50% random negatives")
    print("-" * 60)
    
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10  # Stop if no improvement for 10 epochs
    min_loss_threshold = 0.001  # More aggressive training before stopping
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 20  # More batches per epoch for better coverage
        
        for _ in range(num_batches):
            # Use FAST batch generation (samples from pre-extracted patches)
            a_batch, p_batch, n_batch = create_triplet_batch_fast(
                viral_patches, human_patches, negative_patches, batch_size=batch_size
            )
            
            if a_batch is None: continue
            
            a_batch = a_batch.to(device)
            p_batch = p_batch.to(device)
            n_batch = n_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
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
            patience_counter = 0
            # Save best model immediately
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            marker = ""
            patience_counter += 1
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Triplet Loss: {avg_loss:.4f}{marker}")
        
        # Early stopping checks
        if avg_loss < min_loss_threshold:
            print(f"\n[EARLY STOP] Loss {avg_loss:.4f} < {min_loss_threshold} threshold.")
            print("Stopping to prevent overfitting (memorization).")
            break
            
        if patience_counter >= early_stop_patience:
            print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs.")
            break
            
    print("-" * 60)
    print(f"Training complete! Best Loss: {best_loss:.4f}")
    if best_loss == float('inf'):
         print("  Warning: Loss never converged.")
    else:
         print(f"Saved weights to {SAVE_PATH}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Train with cross-pair hard negatives for 100 epochs
    train_supervised(num_epochs=100, device=device)


