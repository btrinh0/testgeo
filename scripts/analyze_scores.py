"""
Phase 12: Threshold Calibration - Score Distribution Analysis

This script:
1. Loads the Phase 10 trained model (with Cross-Attention)
2. Computes cosine similarity for all True Pairs and Negative Pairs
3. Prints RAW similarity scores for every pair
4. Creates histogram visualization of score distributions
5. Saves plot to results/score_distribution.png
"""

import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg


# ============================================================================
# Configuration
# ============================================================================

POSITIVE_DIR = 'data/benchmark/positive'
NEGATIVE_DIR = 'data/benchmark/negative'
RAW_DIR = 'data/raw'
WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
OUTPUT_PLOT = 'results/score_distribution.png'

# Directories to search for PDB files
PDB_DIRS = [RAW_DIR, POSITIVE_DIR, NEGATIVE_DIR]

# Ground Truth: 16 validated mimicry pairs (viral -> human target)
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

# Viral proteins (known to mimic human proteins)
VIRAL_PDBS = sorted(set(v for v, h in TRUE_PAIRS))

# Negative controls (10 unrelated proteins)
NEGATIVE_PDBS = ['1A3N', '1TRZ', '1MBN', '1UBQ', '1LYZ', '1EMA', '4INS', '1CLL', '7RSA', '1HRC']


# ============================================================================
# Helper Functions
# ============================================================================

def load_model():
    """Load the trained SiameseEGNN model with Phase 10 Cross-Attention."""
    print("Loading SiameseEGNN model (Phase 10 Cross-Attention)...")
    
    model = SiameseEGNN(
        input_dim=320,   # ESM-2 embeddings
        node_dim=64,
        edge_dim=0,
        hidden_dim=64,
        embed_dim=128,
        num_layers=4,
        geom_dim=32      # Cross-attention dimension
    )
    
    if os.path.exists(WEIGHTS_PATH):
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"  [OK] Loaded weights from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"  [ERROR] Loading weights: {e}")
            return None
    else:
        print(f"  [ERROR] {WEIGHTS_PATH} not found!")
        print("  Run train.py first to train the model.")
        return None
    
    model.eval()
    return model


def find_pdb(pdb_id):
    """Search multiple directories for a PDB file."""
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None


def load_pdb(pdb_id):
    """Find and load a PDB file from any search directory."""
    path = find_pdb(pdb_id)
    if path is None:
        print(f"  [ERROR] {pdb_id}.pdb not found in any directory")
        return None
    try:
        data = parse_pdb_to_pyg(path, use_esm=True)
        return data
    except Exception as e:
        print(f"  [ERROR] Loading {pdb_id}: {e}")
        return None


def compute_similarity(model, data1, data2):
    """Compute cosine similarity between two protein graphs."""
    with torch.no_grad():
        z1 = model.forward_one(data1)
        z2 = model.forward_one(data2)
        similarity = F.cosine_similarity(z1, z2, dim=-1)
        return similarity.item()


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_scores():
    print("=" * 70)
    print("Phase 12: Threshold Calibration - Score Distribution Analysis")
    print("=" * 70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Load all proteins
    print("\n" + "-" * 70)
    print("Loading Proteins...")
    print("-" * 70)
    
    viral_graphs = {}
    for pdb_id in VIRAL_PDBS:
        graph = load_pdb(pdb_id)
        if graph is not None:
            viral_graphs[pdb_id] = graph
            print(f"  [OK] Viral {pdb_id}: {graph.x.size(0)} atoms")
    
    # Load human targets
    human_graphs = {}
    for viral_id, human_id in TRUE_PAIRS:
        if human_id not in human_graphs:
            graph = load_pdb(human_id)
            if graph is not None:
                human_graphs[human_id] = graph
                print(f"  [OK] Human {human_id}: {graph.x.size(0)} atoms")
    
    negative_graphs = {}
    for pdb_id in NEGATIVE_PDBS:
        graph = load_pdb(pdb_id)
        if graph is not None:
            negative_graphs[pdb_id] = graph
            print(f"  [OK] Negative {pdb_id}: {graph.x.size(0)} atoms")
    
    # Compute scores for TRUE PAIRS
    print("\n" + "=" * 70)
    print("TRUE POSITIVE PAIRS (Known Mimicry)")
    print("=" * 70)
    print(f"{'Pair':<25} {'Similarity':>12}")
    print("-" * 40)
    
    true_positive_scores = []
    for viral_id, human_id in TRUE_PAIRS:
        if viral_id in viral_graphs and human_id in human_graphs:
            score = compute_similarity(model, viral_graphs[viral_id], human_graphs[human_id])
            true_positive_scores.append(score)
            print(f"{viral_id} vs {human_id:<10} {score:>12.4f}")
        else:
            print(f"{viral_id} vs {human_id:<10} {'MISSING':>12}")
    
    # Compute scores for NEGATIVE PAIRS (viral vs unrelated)
    print("\n" + "=" * 70)
    print("NEGATIVE PAIRS (Viral vs Unrelated Human Proteins)")
    print("=" * 70)
    print(f"{'Pair':<25} {'Similarity':>12}")
    print("-" * 40)
    
    negative_scores = []
    for viral_id in VIRAL_PDBS:
        for neg_id in NEGATIVE_PDBS:
            if viral_id in viral_graphs and neg_id in negative_graphs:
                score = compute_similarity(model, viral_graphs[viral_id], negative_graphs[neg_id])
                negative_scores.append(score)
                print(f"{viral_id} vs {neg_id:<10} {score:>12.4f}")
    
    # Summary Statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    if true_positive_scores:
        tp_mean = sum(true_positive_scores) / len(true_positive_scores)
        tp_min = min(true_positive_scores)
        tp_max = max(true_positive_scores)
        print(f"True Positives ({len(true_positive_scores)} pairs):")
        print(f"  Mean: {tp_mean:.4f}  Min: {tp_min:.4f}  Max: {tp_max:.4f}")
    
    if negative_scores:
        neg_mean = sum(negative_scores) / len(negative_scores)
        neg_min = min(negative_scores)
        neg_max = max(negative_scores)
        print(f"\nNegatives ({len(negative_scores)} pairs):")
        print(f"  Mean: {neg_mean:.4f}  Min: {neg_min:.4f}  Max: {neg_max:.4f}")
    
    if true_positive_scores and negative_scores:
        gap = tp_min - neg_max
        print(f"\nSeparation Gap (TP min - Neg max): {gap:.4f}")
        
        # Suggest threshold
        suggested_threshold = (tp_min + neg_max) / 2
        print(f"Suggested Threshold: {suggested_threshold:.4f}")
    
    # Create Visualization
    print("\n" + "-" * 70)
    print("Creating Histogram Visualization...")
    print("-" * 70)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    if true_positive_scores:
        plt.hist(true_positive_scores, bins=15, alpha=0.7, color='green', 
                 label=f'True Positives (n={len(true_positive_scores)})', edgecolor='darkgreen')
    
    if negative_scores:
        plt.hist(negative_scores, bins=15, alpha=0.7, color='red', 
                 label=f'Negatives (n={len(negative_scores)})', edgecolor='darkred')
    
    # Add labels and title
    plt.xlabel('Cosine Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Phase 12: Score Distribution Analysis\n(True Positives vs Negatives)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add vertical line for suggested threshold
    if true_positive_scores and negative_scores:
        plt.axvline(x=suggested_threshold, color='blue', linestyle='--', 
                    linewidth=2, label=f'Suggested Threshold: {suggested_threshold:.3f}')
        plt.legend(loc='upper left', fontsize=10)
    
    # Set x-axis limits
    plt.xlim(-1.0, 1.0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"  [OK] Saved histogram to {OUTPUT_PLOT}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("Phase 12 Complete!")
    print("=" * 70)
    
    return {
        'true_positive_scores': true_positive_scores,
        'negative_scores': negative_scores
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    analyze_scores()
