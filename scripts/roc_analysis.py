"""
Phase 21: ROC Curve & Threshold Optimization

Generates an ROC curve from the expanded benchmark, calculates AUC,
and finds the optimal threshold using Youden's Index.

Load Data: True Pairs + Decoys + Cross-Folder Negatives
Sweep: TPR/FPR for thresholds 0.0..1.0
Output: results/roc_curve.png + recommended threshold
"""

import os
import sys
import numpy as np
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
OUTPUT_PATH = 'results/roc_curve.png'

# Directories to find PDBs
PDB_DIRS = ['data/raw', 'data/benchmark/positive', 'data/benchmark/negative']

# Expanded TRUE_PAIRS (16 validated mimicry pairs)
TRUE_PAIRS = [
    ('1Q59', '1G5M'),  # EBV BHRF1 -> Bcl-2
    ('2V5I', '1LB5'),  # Vaccinia A52 -> TRAF6
    ('3CL3', '3H11'),  # KSHV vFLIP -> FLIP
    ('2GX9', '1KX5'),  # Flu NS1 -> Histone H3
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

# Negative controls (unrelated proteins)
NEGATIVE_IDS = ['1A3N', '1TRZ', '1MBN', '1UBQ', '1LYZ', '1EMA', '4INS', '1CLL', '7RSA', '1HRC']


# ============================================================================
# Helpers
# ============================================================================

def find_pdb(pdb_id):
    """Search multiple directories for a PDB file."""
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None


def load_model():
    model = SiameseEGNN(
        node_dim=32, edge_dim=0, hidden_dim=64,
        embed_dim=128, num_layers=4, geom_dim=32
    )
    state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_similarity(model, data1, data2):
    with torch.no_grad():
        emb1 = model.forward_one(data1)
        emb2 = model.forward_one(data2)
        return F.cosine_similarity(emb1, emb2).item()


# ============================================================================
# Data Collection
# ============================================================================

def collect_scores(model):
    """Compute all pairwise scores and labels."""
    print("Loading proteins...")
    
    # Get all unique PDB IDs
    all_viral = sorted(set(v for v, h in TRUE_PAIRS))
    all_human = sorted(set(h for v, h in TRUE_PAIRS))
    true_pair_set = set(TRUE_PAIRS)
    
    # Load all proteins
    proteins = {}
    for pdb_id in set(all_viral + all_human + NEGATIVE_IDS):
        path = find_pdb(pdb_id)
        if path:
            try:
                proteins[pdb_id] = parse_pdb_to_pyg(path, use_esm=True)
            except Exception as e:
                print(f"  [ERROR] {pdb_id}: {e}")
    
    print(f"  Loaded {len(proteins)} proteins")
    
    scores = []
    labels = []  # 1 = true pair, 0 = not a pair
    pair_names = []
    
    # Part 1: Viral vs Human (true pairs + cross-folder)
    print("\nScoring viral vs human pairs...")
    for viral_id in all_viral:
        for human_id in all_human:
            if viral_id not in proteins or human_id not in proteins:
                continue
            sim = compute_similarity(model, proteins[viral_id], proteins[human_id])
            is_true = (viral_id, human_id) in true_pair_set
            scores.append(sim)
            labels.append(1 if is_true else 0)
            pair_names.append(f"{viral_id}-{human_id}")
            tag = "TRUE" if is_true else "neg"
            print(f"  {viral_id} vs {human_id}: {sim:.4f} [{tag}]")
    
    # Part 2: Viral vs Negatives (all should be negative)
    print("\nScoring viral vs negative controls...")
    for viral_id in all_viral:
        for neg_id in NEGATIVE_IDS:
            if viral_id not in proteins or neg_id not in proteins:
                continue
            sim = compute_similarity(model, proteins[viral_id], proteins[neg_id])
            scores.append(sim)
            labels.append(0)
            pair_names.append(f"{viral_id}-{neg_id}")
            print(f"  {viral_id} vs {neg_id}: {sim:.4f} [neg]")
    
    return np.array(scores), np.array(labels), pair_names


# ============================================================================
# ROC Analysis
# ============================================================================

def compute_roc(scores, labels):
    """Compute ROC curve by sweeping thresholds."""
    thresholds = np.linspace(-1.0, 1.0, 2001)
    
    tprs = []
    fprs = []
    
    total_pos = np.sum(labels == 1)
    total_neg = np.sum(labels == 0)
    
    for thresh in thresholds:
        predicted_pos = scores >= thresh
        
        tp = np.sum(predicted_pos & (labels == 1))
        fp = np.sum(predicted_pos & (labels == 0))
        
        tpr = tp / total_pos if total_pos > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    return np.array(fprs), np.array(tprs), thresholds


def compute_auc(fprs, tprs):
    """Compute AUC using trapezoidal rule."""
    # Sort by FPR
    sorted_indices = np.argsort(fprs)
    sorted_fprs = fprs[sorted_indices]
    sorted_tprs = tprs[sorted_indices]
    
    # np.trapezoid (NumPy 2.0+) or np.trapz (older)
    trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
    auc = trapz_fn(sorted_tprs, sorted_fprs)
    return auc


def find_optimal_threshold(fprs, tprs, thresholds):
    """Find optimal threshold using Youden's Index (J = Sensitivity + Specificity - 1)."""
    specificities = 1 - fprs
    youdens_j = tprs + specificities - 1
    
    best_idx = np.argmax(youdens_j)
    return thresholds[best_idx], youdens_j[best_idx], tprs[best_idx], fprs[best_idx]


# ============================================================================
# Visualization
# ============================================================================

def plot_roc(fprs, tprs, auc, opt_thresh, opt_tpr, opt_fpr, output_path):
    """Plot ROC curve with AUC and optimal threshold."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # ROC curve
    sorted_indices = np.argsort(fprs)
    ax.plot(fprs[sorted_indices], tprs[sorted_indices],
            color='#e74c3c', linewidth=2.5, label=f'GeoMimic-Net (AUC = {auc:.4f})')
    
    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC = 0.50)')
    
    # Optimal threshold point
    ax.scatter([opt_fpr], [opt_tpr], s=150, c='#27ae60', zorder=5, edgecolors='black', linewidths=2)
    ax.annotate(f'Optimal Threshold = {opt_thresh:.3f}\n(TPR={opt_tpr:.2f}, FPR={opt_fpr:.2f})',
               xy=(opt_fpr, opt_tpr),
               xytext=(opt_fpr + 0.15, opt_tpr - 0.15),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#27ae60'))
    
    # AUC label
    ax.text(0.60, 0.15, f'AUC = {auc:.4f}', fontsize=18, fontweight='bold',
            color='#e74c3c', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#e74c3c', alpha=0.9))
    
    # Styling
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve: GeoMimic-Net Mimicry Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved ROC curve to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 21: ROC Curve & Threshold Optimization")
    print("=" * 60)
    
    model = load_model()
    scores, labels, pair_names = collect_scores(model)
    
    print(f"\n--- Score Summary ---")
    print(f"  Total pairs:    {len(scores)}")
    print(f"  True positives: {np.sum(labels == 1)}")
    print(f"  True negatives: {np.sum(labels == 0)}")
    
    # Compute ROC
    print("\n--- Computing ROC Curve ---")
    fprs, tprs, thresholds = compute_roc(scores, labels)
    auc = compute_auc(fprs, tprs)
    print(f"  AUC: {auc:.4f}")
    
    # Find optimal threshold
    opt_thresh, opt_j, opt_tpr, opt_fpr = find_optimal_threshold(fprs, tprs, thresholds)
    print(f"\n--- Optimal Threshold (Youden's Index) ---")
    print(f"  Threshold:   {opt_thresh:.4f}")
    print(f"  Youden's J:  {opt_j:.4f}")
    print(f"  Sensitivity: {opt_tpr:.4f}")
    print(f"  Specificity: {1 - opt_fpr:.4f}")
    print(f"  FPR:         {opt_fpr:.4f}")
    
    # Calculate F1 at optimal threshold
    predicted_pos = scores >= opt_thresh
    tp = np.sum(predicted_pos & (labels == 1))
    fp = np.sum(predicted_pos & (labels == 0))
    fn = np.sum(~predicted_pos & (labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n--- F1 at Optimal Threshold ---")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    
    # Plot
    print("\n--- Generating ROC Plot ---")
    plot_roc(fprs, tprs, auc, opt_thresh, opt_tpr, opt_fpr, OUTPUT_PATH)
    
    # Final recommendation
    print("\n" + "=" * 60)
    print(f"  RECOMMENDED THRESHOLD: {opt_thresh:.4f}")
    print(f"  AUC: {auc:.4f}")
    print("=" * 60)
    print("Phase 21 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
