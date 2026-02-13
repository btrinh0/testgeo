"""
Phase E: Publication-Ready Figures & Rank Analysis

Generates key publication-quality metrics and figures:
1. Per-pair rank table (where does true target rank?)
2. Cumulative rank distribution plot
3. Summary statistics table for paper
"""

import os
import sys
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

TRUE_PAIRS = [
    ('1Q59', '1G5M', 'EBV BHRF1 / Bcl-2'),
    ('2V5I', '1LB5', 'Vaccinia A52 / TRAF6'),
    ('3CL3', '3H11', 'KSHV vFLIP / FLIP'),
    ('2GX9', '1KX5', 'Flu NS1 / Histone H3'),
    ('2JBY', '1G5M', 'Myxoma M11L / Bcl-2'),
    ('1B4C', '1ITB', 'Vaccinia B15 / IL-1R'),
    ('1FV1', '1CDF', 'EBV LMP1 / CD40'),
    ('1H26', '1CF7', 'Adenovirus E1A / E2F'),
    ('1GUX', '1CF7', 'HPV E7 / E2F'),
    ('1EFN', '1SHF', 'HIV Nef / Fyn SH3'),
    ('3D2U', '1HHK', 'CMV UL18 / MHC-I HLA-A'),
    ('2UWI', '1EXT', 'Cowpox CrmE / TNFR1'),
    ('2BZR', '1MAZ', 'KSHV vBcl-2 / Bcl-xL'),
    ('2VGA', '1CA9', 'Variola CrmB / TNFR2'),
    ('1F5Q', '1B7T', 'KSHV vCyclin / Cyclin D2'),
    ('2BBR', '1A1W', 'MC159 / FADD DED'),
]

NEGATIVE_PDBS = ['1A3N', '1TRZ', '1MBN', '1UBQ', '1LYZ', '1EMA', '4INS', '1CLL', '7RSA', '1HRC']

ALL_HUMAN_PDBS = sorted(set(h for _, h, _ in TRUE_PAIRS))


# ============================================================================
# Helpers
# ============================================================================

def find_pdb(pdb_id):
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
    with torch.no_grad():
        emb1 = model.forward_one(graph1)
        emb2 = model.forward_one(graph2)
        return F.cosine_similarity(emb1, emb2).item()

def load_all_graphs():
    all_ids = set()
    for v, h, _ in TRUE_PAIRS:
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
                print(f"  [WARN] {pdb_id}: {e}")
    return graphs


# ============================================================================
# Per-Pair Rank Analysis
# ============================================================================

def compute_ranks(model, graphs):
    """For each viral protein, rank all human candidates by similarity."""
    results = []
    
    for viral_id, true_human_id, name in TRUE_PAIRS:
        if viral_id not in graphs:
            continue
        
        # Score against all human candidates
        all_scores = {}
        for h_id in ALL_HUMAN_PDBS:
            if h_id in graphs:
                sim = compute_similarity(model, graphs[viral_id], graphs[h_id])
                all_scores[h_id] = sim
        
        # Score against negatives
        neg_scores = {}
        for n_id in NEGATIVE_PDBS:
            if n_id in graphs:
                sim = compute_similarity(model, graphs[viral_id], graphs[n_id])
                neg_scores[n_id] = sim
        
        # Rank true target
        ranked_humans = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        true_score = all_scores.get(true_human_id, 0)
        rank = next(i+1 for i, (h, s) in enumerate(ranked_humans) if h == true_human_id)
        
        # Combined ranking (humans + negatives)
        all_combined = {**all_scores, **neg_scores}
        ranked_all = sorted(all_combined.items(), key=lambda x: x[1], reverse=True)
        rank_overall = next(i+1 for i, (h, s) in enumerate(ranked_all) if h == true_human_id)
        
        # Separation: how much higher is true score vs 2nd human?
        if len(ranked_humans) > 1 and rank == 1:
            second_score = ranked_humans[1][1]
            separation = true_score - second_score
        else:
            separation = 0
        
        results.append({
            'name': name,
            'viral_id': viral_id,
            'human_id': true_human_id,
            'true_score': true_score,
            'rank_human': rank,
            'rank_overall': rank_overall,
            'total_humans': len(ranked_humans),
            'total_overall': len(ranked_all),
            'separation': separation,
            'top_3': [(h, s) for h, s in ranked_humans[:3]],
        })
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def create_publication_figure(results, output_path):
    """Create publication-quality cumulative rank distribution + rank table."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.5]})
        
        # --- Left: Cumulative Rank Distribution ---
        ranks = [r['rank_human'] for r in results]
        max_rank = max(ranks) + 1
        
        cum_ranks = []
        for k in range(1, max_rank + 1):
            frac = sum(1 for r in ranks if r <= k) / len(ranks) * 100
            cum_ranks.append(frac)
        
        axes[0].plot(range(1, max_rank + 1), cum_ranks, 'o-', color='#2ecc71', 
                    linewidth=2.5, markersize=8, label='GeoMimic-Net')
        
        # Random baseline
        random_cum = [k / results[0]['total_humans'] * 100 for k in range(1, max_rank + 1)]
        axes[0].plot(range(1, max_rank + 1), random_cum, '--', color='#e74c3c', 
                    linewidth=1.5, alpha=0.7, label='Random')
        
        axes[0].set_xlabel('Rank Cutoff (k)', fontsize=12)
        axes[0].set_ylabel('% True Targets in Top-k', fontsize=12)
        axes[0].set_title('Cumulative Rank Distribution', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].set_ylim(0, 105)
        axes[0].set_xlim(0.5, max_rank + 0.5)
        axes[0].grid(True, alpha=0.3)
        
        # Highlight points
        for k in [1, 3, 5]:
            if k <= len(cum_ranks):
                axes[0].annotate(f'{cum_ranks[k-1]:.0f}%', 
                               (k, cum_ranks[k-1]), 
                               textcoords="offset points", xytext=(10, 5),
                               fontsize=10, fontweight='bold', color='#27ae60')
        
        # --- Right: Per-Pair Score Table ---
        axes[1].axis('off')
        
        # Table data
        cell_text = []
        cell_colors = []
        for r in results:
            rank_color = '#d5f5e3' if r['rank_human'] <= 3 else '#fadbd8' if r['rank_human'] > 5 else '#fdebd0'
            cell_text.append([
                r['name'],
                f"{r['true_score']:+.3f}",
                f"#{r['rank_human']}/{r['total_humans']}",
                f"#{r['rank_overall']}/{r['total_overall']}",
            ])
            cell_colors.append([rank_color] * 4)
        
        table = axes[1].table(
            cellText=cell_text,
            colLabels=['Mimicry Pair', 'Score', 'Rank (Human)', 'Rank (All)'],
            cellColours=cell_colors,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header
        for j in range(4):
            table[0, j].set_facecolor('#2c3e50')
            table[0, j].set_text_props(color='white', fontweight='bold')
        
        axes[1].set_title('Per-Pair Rank Analysis', fontsize=13, fontweight='bold')
        
        plt.suptitle('GeoMimic-Net: Publication Metrics', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Saved publication figures to {output_path}")
        
    except ImportError:
        print("\n[WARN] matplotlib not available, skipping figures")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase E: Publication-Ready Metrics & Rank Analysis")
    print("=" * 70)
    
    model = load_model()
    print("\nLoading graphs...")
    graphs = load_all_graphs()
    print(f"  Loaded {len(graphs)} graphs")
    
    results = compute_ranks(model, graphs)
    
    # Print detailed table
    print("\n" + "=" * 80)
    print(f"{'Pair':<28s} {'Score':>8s} {'Rank(H)':>8s} {'Rank(All)':>10s} {'Top-3 Candidates'}")
    print("-" * 80)
    
    for r in results:
        top3_str = ', '.join(f"{h}({s:.2f})" for h, s in r['top_3'])
        hr = f"#{r['rank_human']}/{r['total_humans']}"
        ar = f"#{r['rank_overall']}/{r['total_overall']}"
        print(f"  {r['name']:<26s} {r['true_score']:>+8.4f} {hr:>8s} {ar:>10s}  {top3_str}")
    
    # Summary stats
    ranks = [r['rank_human'] for r in results]
    overall_ranks = [r['rank_overall'] for r in results]
    scores = [r['true_score'] for r in results]
    
    top1 = sum(1 for r in ranks if r == 1)
    top3 = sum(1 for r in ranks if r <= 3)
    top5 = sum(1 for r in ranks if r <= 5)
    
    print("\n" + "=" * 70)
    print("PUBLICATION SUMMARY METRICS")
    print("=" * 70)
    print(f"  Dataset:          16 validated viral-human mimicry pairs")
    print(f"  Human candidates: {results[0]['total_humans']} proteins")
    print(f"  Negative controls: {len(NEGATIVE_PDBS)} unrelated proteins")
    print(f"")
    print(f"  Mean True-Pair Score:  {np.mean(scores):.4f}")  
    print(f"  Mean Rank (human):     {np.mean(ranks):.2f} / {results[0]['total_humans']}")
    print(f"  Mean Rank (overall):   {np.mean(overall_ranks):.2f} / {results[0]['total_overall']}")
    print(f"  Median Rank (human):   {np.median(ranks):.1f}")
    print(f"")
    print(f"  Top-1 Accuracy:  {top1}/{len(results)} ({100*top1/len(results):.1f}%)")
    print(f"  Top-3 Accuracy:  {top3}/{len(results)} ({100*top3/len(results):.1f}%)")
    print(f"  Top-5 Accuracy:  {top5}/{len(results)} ({100*top5/len(results):.1f}%)")
    print(f"")
    print(f"  MRR (Mean Reciprocal Rank): {np.mean([1/r for r in ranks]):.4f}")
    
    # Create figure
    output_path = os.path.join(OUTPUT_DIR, 'publication_figures.png')
    create_publication_figure(results, output_path)
    
    print("\n" + "=" * 70)
    print("Phase E Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
