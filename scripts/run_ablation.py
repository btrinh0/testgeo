"""
Phase 18: Ablation Study
Compare the contribution of Geometry vs Sequence features.

Experiments:
1. Geometry Only: Zero out ESM-2 sequence features
2. Sequence Only: Zero out EGNN geometric features (randomize positions)
3. Full Model: Standard model with both features

Hypothesis: Geometry ~0.45, Sequence ~0.60, GeoMimic-Net ~1.0
"""

import os
import sys
import copy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
POSITIVE_DIR = 'data/benchmark/positive'
NEGATIVE_DIR = 'data/benchmark/negative'
OUTPUT_PATH = 'results/ablation_chart.png'
THRESHOLD = 0.85

# Ground Truth
TRUE_PAIRS = [
    ('1Q59', '1G5M'),
    ('2V5I', '1LB5'),
    ('3CL3', '3H11'),
    ('2GX9', '1KX5'),
]

VIRAL_PDBS = ['1Q59', '2V5I', '3CL3', '2GX9']
HUMAN_PDBS = ['1G5M', '1LB5', '3H11', '1KX5']
NEGATIVE_PDBS = ['1A3N', '1TRZ', '1MBN', '1UBQ']


# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    model = SiameseEGNN(
        node_dim=32,
        edge_dim=0,
        hidden_dim=64,
        embed_dim=128,
        num_layers=4,
        geom_dim=32
    )
    
    if os.path.exists(WEIGHTS_PATH):
        state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model


# ============================================================================
# Data Loading with Ablation
# ============================================================================

def load_proteins():
    """Load all proteins for benchmark."""
    proteins = {}
    
    # Load viral proteins
    for pdb_id in VIRAL_PDBS:
        path = os.path.join(POSITIVE_DIR, f"{pdb_id}.pdb")
        if os.path.exists(path):
            proteins[pdb_id] = parse_pdb_to_pyg(path, use_esm=True)
    
    # Load human proteins
    for pdb_id in HUMAN_PDBS:
        path = os.path.join(POSITIVE_DIR, f"{pdb_id}.pdb")
        if os.path.exists(path):
            proteins[pdb_id] = parse_pdb_to_pyg(path, use_esm=True)
    
    # Load negative proteins
    for pdb_id in NEGATIVE_PDBS:
        path = os.path.join(NEGATIVE_DIR, f"{pdb_id}.pdb")
        if os.path.exists(path):
            proteins[pdb_id] = parse_pdb_to_pyg(path, use_esm=True)
    
    return proteins


def apply_ablation(data, mode):
    """
    Apply ablation to data based on mode.
    
    Args:
        data: PyG Data object
        mode: 'full', 'geometry_only', or 'sequence_only'
    
    Returns:
        Modified data object
    """
    ablated = copy.deepcopy(data)
    
    if mode == 'geometry_only':
        # Zero out sequence features (ESM-2 embeddings)
        ablated.x = torch.zeros_like(ablated.x)
        
    elif mode == 'sequence_only':
        # Randomize positions to destroy geometric information
        # Keep the center of mass but randomize relative positions
        center = ablated.pos.mean(dim=0)
        ablated.pos = torch.randn_like(ablated.pos) * 10 + center
        
    # 'full' mode keeps everything as-is
    return ablated


# ============================================================================
# Evaluation
# ============================================================================

def compute_similarity(model, data1, data2):
    """Compute cosine similarity between two proteins."""
    with torch.no_grad():
        emb1 = model.forward_one(data1)
        emb2 = model.forward_one(data2)
        similarity = F.cosine_similarity(emb1, emb2).item()
    return similarity


def run_experiment(model, proteins, mode):
    """
    Run benchmark with specified ablation mode.
    
    Returns:
        dict with TP, FP, TN, FN counts and F1 score
    """
    TP, FP, TN, FN = 0, 0, 0, 0
    
    # Test viral vs human (positive folder)
    for viral_id in VIRAL_PDBS:
        for human_id in HUMAN_PDBS:
            if viral_id not in proteins or human_id not in proteins:
                continue
            
            data1 = apply_ablation(proteins[viral_id], mode)
            data2 = apply_ablation(proteins[human_id], mode)
            
            sim = compute_similarity(model, data1, data2)
            predicted_positive = sim >= THRESHOLD
            is_true_pair = (viral_id, human_id) in TRUE_PAIRS
            
            if is_true_pair:
                if predicted_positive:
                    TP += 1
                else:
                    FN += 1
            else:
                if predicted_positive:
                    FP += 1
                else:
                    TN += 1
    
    # Test viral vs negatives
    for viral_id in VIRAL_PDBS:
        for neg_id in NEGATIVE_PDBS:
            if viral_id not in proteins or neg_id not in proteins:
                continue
            
            data1 = apply_ablation(proteins[viral_id], mode)
            data2 = apply_ablation(proteins[neg_id], mode)
            
            sim = compute_similarity(model, data1, data2)
            predicted_positive = sim >= THRESHOLD
            
            # All viral vs negative should be negative
            if predicted_positive:
                FP += 1
            else:
                TN += 1
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================================
# Visualization
# ============================================================================

def create_ablation_chart(results, output_path):
    """Create bar chart comparing F1 scores."""
    modes = ['Geometry Only', 'Sequence Only', 'Full Model\n(GeoMimic-Net)']
    f1_scores = [
        results['geometry_only']['f1'],
        results['sequence_only']['f1'],
        results['full']['f1']
    ]
    
    # Colors
    colors = ['#e74c3c', '#3498db', '#27ae60']  # Red, Blue, Green
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(modes, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=14, fontweight='bold')
    
    # Styling
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Ablation Study: Feature Contribution Analysis', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.85)')
    
    # Add hypothesis annotations
    ax.text(0, 0.48, 'Hypothesis:\n~0.45', ha='center', va='bottom', fontsize=9, color='gray')
    ax.text(1, 0.63, 'Hypothesis:\n~0.60', ha='center', va='bottom', fontsize=9, color='gray')
    ax.text(2, 1.03, 'Hypothesis:\n~1.0', ha='center', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved ablation chart to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 18: Ablation Study")
    print("=" * 60)
    
    print("\nLoading model and data...")
    model = load_model()
    proteins = load_proteins()
    print(f"  Loaded {len(proteins)} proteins")
    
    results = {}
    
    # Experiment 1: Geometry Only
    print("\n--- Experiment 1: Geometry Only ---")
    print("  (Zeroing out ESM-2 sequence features)")
    results['geometry_only'] = run_experiment(model, proteins, 'geometry_only')
    print(f"  F1 Score: {results['geometry_only']['f1']:.4f}")
    
    # Experiment 2: Sequence Only
    print("\n--- Experiment 2: Sequence Only ---")
    print("  (Randomizing positions to destroy geometry)")
    results['sequence_only'] = run_experiment(model, proteins, 'sequence_only')
    print(f"  F1 Score: {results['sequence_only']['f1']:.4f}")
    
    # Experiment 3: Full Model
    print("\n--- Experiment 3: Full Model (GeoMimic-Net) ---")
    print("  (Both sequence and geometry features)")
    results['full'] = run_experiment(model, proteins, 'full')
    print(f"  F1 Score: {results['full']['f1']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Mode':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for mode, name in [('geometry_only', 'Geometry Only'), 
                       ('sequence_only', 'Sequence Only'),
                       ('full', 'Full Model')]:
        r = results[mode]
        print(f"{name:<25} {r['precision']:.4f}       {r['recall']:.4f}       {r['f1']:.4f}")
    
    # Create visualization
    print("\n--- Creating Ablation Chart ---")
    create_ablation_chart(results, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("Phase 18 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
