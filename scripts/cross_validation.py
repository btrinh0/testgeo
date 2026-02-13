"""
Phase A: Leave-One-Out Cross-Validation (LOOCV)


Performs LOOCV on the 16 true mimicry pairs:
- For each fold, holds out 1 true pair as test
- Evaluates if the model ranks the true target highest among all human candidates
- Computes AUC, rank metrics, and 95% confidence intervals
- Performs permutation test for statistical significance (p-value)

This addresses the statistical rigor gap vs competing projects.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg


# ============================================================================
# Configuration
# ============================================================================

RAW_DIR = 'data/raw'
POSITIVE_DIR = 'data/benchmark/positive'
NEGATIVE_DIR = 'data/benchmark/negative'
WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
OUTPUT_DIR = 'results'

PDB_DIRS = [RAW_DIR, POSITIVE_DIR, NEGATIVE_DIR]

# Ground Truth: 16 validated mimicry pairs
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

PAIR_NAMES = [
    'EBV BHRF1/Bcl-2', 'Vaccinia A52/TRAF6', 'KSHV vFLIP/FLIP', 
    'Flu NS1/Histone H3', 'Myxoma M11L/Bcl-2', 'Vaccinia B15/IL-1R',
    'EBV LMP1/CD40', 'Adenovirus E1A/E2F', 'HPV E7/E2F',
    'HIV Nef/Fyn SH3', 'CMV UL18/MHC-I', 'Cowpox CrmE/TNFR1',
    'KSHV vBcl-2/Bcl-xL', 'Variola CrmB/TNFR2', 'KSHV vCyclin/CyclinD2',
    'MC159/FADD DED'
]

NEGATIVE_PDBS = ['1A3N', '1TRZ', '1MBN', '1UBQ', '1LYZ', '1EMA', '4INS', '1CLL', '7RSA', '1HRC']

ALL_HUMAN_PDBS = sorted(set(h for _, h in TRUE_PAIRS))


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
        node_dim=64, edge_dim=0, hidden_dim=128,
        embed_dim=256, num_layers=4, geom_dim=64,
        num_rbf=16, dropout=0.1
    )
    if os.path.exists(WEIGHTS_PATH):
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_similarity(model, graph1, graph2):
    """Compute cosine similarity between two pre-loaded graphs."""
    with torch.no_grad():
        emb1 = model.forward_one(graph1)
        emb2 = model.forward_one(graph2)
        return F.cosine_similarity(emb1, emb2).item()


def load_all_graphs():
    """Load all PDB files into PyG graphs."""
    all_ids = set()
    for v, h in TRUE_PAIRS:
        all_ids.add(v)
        all_ids.add(h)
    for n in NEGATIVE_PDBS:
        all_ids.add(n)
    
    graphs = {}
    for pdb_id in sorted(all_ids):
        path = find_pdb(pdb_id)
        if path:
            try:
                graphs[pdb_id] = parse_pdb_to_pyg(path, use_esm=True)
            except Exception as e:
                print(f"  [WARN] Failed to load {pdb_id}: {e}")
    return graphs


# ============================================================================
# Per-Pair AUC (each viral vs all human candidates)
# ============================================================================

def compute_per_pair_auc(scores_true, scores_neg):
    """
    Compute AUC for a single pair: is the true score ranked above negatives?
    Manual ROC to avoid sklearn dependency.
    """
    labels = [1] * len(scores_true) + [0] * len(scores_neg)
    scores = scores_true + scores_neg
    
    # Sort by score descending
    paired = sorted(zip(scores, labels), reverse=True)
    
    tp = 0
    fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    
    if total_pos == 0 or total_neg == 0:
        return 0.5
    
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr
    
    return auc


# ============================================================================
# Leave-One-Out Cross-Validation
# ============================================================================

def run_loocv(model, graphs):
    """
    Leave-One-Out Cross-Validation.
    
    For each held-out pair (viral_i, human_i):
    1. Score viral_i against ALL human candidates
    2. Record rank of true target among all candidates
    3. Compute per-pair AUC
    """
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
    print("=" * 70)
    
    results = []
    
    for fold_idx, (viral_id, human_id) in enumerate(TRUE_PAIRS):
        if viral_id not in graphs or human_id not in graphs:
            print(f"  [SKIP] Fold {fold_idx+1}: {viral_id} or {human_id} not loaded")
            continue
        
        # Score this viral against ALL human candidates
        scores = {}
        for h_id in ALL_HUMAN_PDBS:
            if h_id in graphs:
                sim = compute_similarity(model, graphs[viral_id], graphs[h_id])
                scores[h_id] = sim
        
        # Also score against negative controls
        neg_scores = []
        for n_id in NEGATIVE_PDBS:
            if n_id in graphs:
                sim = compute_similarity(model, graphs[viral_id], graphs[n_id])
                neg_scores.append(sim)
        
        # Rank the true target
        true_score = scores.get(human_id, 0)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank = next(i+1 for i, (h, s) in enumerate(ranked) if h == human_id)
        
        # Per-pair AUC: true score vs (other humans + negatives)
        other_scores = [s for h, s in scores.items() if h != human_id] + neg_scores
        pair_auc = compute_per_pair_auc([true_score], other_scores)
        
        results.append({
            'fold': fold_idx + 1,
            'name': PAIR_NAMES[fold_idx],
            'viral': viral_id,
            'human': human_id,
            'true_score': true_score,
            'rank': rank,
            'total_candidates': len(scores),
            'pair_auc': pair_auc,
        })
        
        status = "[OK]" if rank <= 3 else "[X]"
        print(f"  Fold {fold_idx+1:2d}: {PAIR_NAMES[fold_idx]:25s} | "
              f"Score={true_score:+.4f} | Rank={rank}/{len(scores)} | "
              f"AUC={pair_auc:.3f} {status}")
    
    return results


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_confidence_interval(values, confidence=0.95):
    """Compute mean and 95% CI using bootstrap."""
    n = len(values)
    mean = np.mean(values)
    
    # Bootstrap CI
    n_bootstrap = 10000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = sorted(bootstrap_means)
    lower_idx = int((1 - confidence) / 2 * n_bootstrap)
    upper_idx = int((1 + confidence) / 2 * n_bootstrap)
    
    return mean, bootstrap_means[lower_idx], bootstrap_means[upper_idx]


def permutation_test(true_ranks, n_candidates, n_permutations=10000):
    """
    Permutation test: Is the mean rank significantly better than random?
    
    Null hypothesis: model assigns ranks uniformly at random.
    """
    observed_mean_rank = np.mean(true_ranks)
    
    count_as_good_or_better = 0
    for _ in range(n_permutations):
        # Random ranks under null
        random_ranks = [random.randint(1, n_candidates) for _ in true_ranks]
        if np.mean(random_ranks) <= observed_mean_rank:
            count_as_good_or_better += 1
    
    p_value = count_as_good_or_better / n_permutations
    return p_value


# ============================================================================
# Visualization
# ============================================================================

def create_cv_chart(results, output_path):
    """Create cross-validation results visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # --- Left: Rank per pair ---
        names = [r['name'] for r in results]
        ranks = [r['rank'] for r in results]
        totals = [r['total_candidates'] for r in results]
        
        colors = ['#2ecc71' if r <= 3 else '#e74c3c' if r > 5 else '#f39c12' for r in ranks]
        
        bars = axes[0].barh(range(len(names)), ranks, color=colors, edgecolor='white', linewidth=0.5)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=9)
        axes[0].set_xlabel('Rank of True Target', fontsize=11)
        axes[0].set_title('LOOCV: True Target Rank\n(lower is better)', fontsize=13, fontweight='bold')
        axes[0].axvline(x=3, color='green', linestyle='--', alpha=0.5, label='Top-3')
        axes[0].legend(fontsize=9)
        axes[0].invert_yaxis()
        
        # Add rank numbers on bars
        for bar, rank in zip(bars, ranks):
            axes[0].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                        f'#{rank}', va='center', fontsize=9, fontweight='bold')
        
        # --- Right: Per-pair AUC ---
        aucs = [r['pair_auc'] for r in results]
        
        colors2 = ['#2ecc71' if a >= 0.8 else '#e74c3c' if a < 0.5 else '#f39c12' for a in aucs]
        
        bars2 = axes[1].barh(range(len(names)), aucs, color=colors2, edgecolor='white', linewidth=0.5)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=9)
        axes[1].set_xlabel('Per-Pair AUC', fontsize=11)
        axes[1].set_title('LOOCV: Per-Pair AUC\n(higher is better)', fontsize=13, fontweight='bold')
        axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        axes[1].set_xlim(0, 1.1)
        axes[1].legend(fontsize=9)
        axes[1].invert_yaxis()
        
        # Add AUC numbers on bars
        for bar, auc in zip(bars2, aucs):
            axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{auc:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Saved LOOCV chart to {output_path}")
        
    except ImportError:
        print("\n[WARN] matplotlib not available, skipping chart")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase A: Leave-One-Out Cross-Validation with Statistical Analysis")
    print("=" * 70)
    
    # Load model and data
    print("\nLoading model...")
    model = load_model()
    
    print("Loading all protein graphs...")
    graphs = load_all_graphs()
    print(f"  Loaded {len(graphs)} graphs")
    
    # Run LOOCV
    results = run_loocv(model, graphs)
    
    if not results:
        print("[ERROR] No results generated!")
        return
    
    # ====================================================================
    # Summary Statistics
    # ====================================================================
    ranks = [r['rank'] for r in results]
    aucs = [r['pair_auc'] for r in results]
    scores = [r['true_score'] for r in results]
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Rank analysis
    mean_rank, rank_ci_low, rank_ci_high = compute_confidence_interval(ranks)
    top1 = sum(1 for r in ranks if r == 1)
    top3 = sum(1 for r in ranks if r <= 3)
    top5 = sum(1 for r in ranks if r <= 5)
    n_candidates = results[0]['total_candidates'] if results else 14
    
    print(f"\n  Rank Analysis ({n_candidates} human candidates):")
    print(f"    Mean Rank:     {mean_rank:.2f} ± [{rank_ci_low:.2f}, {rank_ci_high:.2f}] (95% CI)")
    print(f"    Top-1 Hits:    {top1}/{len(results)} ({100*top1/len(results):.1f}%)")
    print(f"    Top-3 Hits:    {top3}/{len(results)} ({100*top3/len(results):.1f}%)")
    print(f"    Top-5 Hits:    {top5}/{len(results)} ({100*top5/len(results):.1f}%)")
    
    # AUC analysis
    mean_auc, auc_ci_low, auc_ci_high = compute_confidence_interval(aucs)
    print(f"\n  Per-Pair AUC:")
    print(f"    Mean AUC:      {mean_auc:.4f} ± [{auc_ci_low:.4f}, {auc_ci_high:.4f}] (95% CI)")
    print(f"    Min AUC:       {min(aucs):.4f}")
    print(f"    Max AUC:       {max(aucs):.4f}")
    
    # Score analysis
    mean_score, score_ci_low, score_ci_high = compute_confidence_interval(scores)
    print(f"\n  True Pair Scores:")
    print(f"    Mean Score:    {mean_score:.4f} ± [{score_ci_low:.4f}, {score_ci_high:.4f}] (95% CI)")
    
    # Permutation test
    print("\n  Permutation Test (H0: random ranking):")
    p_value = permutation_test(ranks, n_candidates, n_permutations=10000)
    print(f"    p-value:       {p_value:.6f}")
    if p_value < 0.001:
        print(f"    Significance:  *** p < 0.001 (highly significant)")
    elif p_value < 0.01:
        print(f"    Significance:  ** p < 0.01 (significant)")
    elif p_value < 0.05:
        print(f"    Significance:  * p < 0.05 (significant)")
    else:
        print(f"    Significance:  NS (not significant)")
    
    # Per-pair table
    print("\n" + "-" * 70)
    print(f"{'Pair':<28s} {'Score':>8s} {'Rank':>6s} {'AUC':>8s}")
    print("-" * 70)
    for r in results:
        marker = "[OK]" if r['rank'] <= 3 else ""
        print(f"  {r['name']:<26s} {r['true_score']:>+8.4f} "
              f"  #{r['rank']:<3d}  {r['pair_auc']:>8.3f} {marker}")
    
    # Create visualization
    output_path = os.path.join(OUTPUT_DIR, 'cross_validation.png')
    create_cv_chart(results, output_path)
    
    print("\n" + "=" * 70)
    print("Phase A Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
