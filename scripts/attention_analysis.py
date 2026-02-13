"""
Phase C: Attention Interpretability Analysis

Extracts cross-attention weights from the GeoMimic-Net model to identify
which residues drive mimicry detection. Generates per-pair attention heatmaps
showing the top-attended residues — providing biological interpretability
comparable to CLIGAT's influence trees or NeuroPlasmaNet's gene panels.
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
    ('3D2U', '1HHK', 'CMV UL18 / MHC-I'),
    ('2UWI', '1EXT', 'Cowpox CrmE / TNFR1'),
    ('2BZR', '1MAZ', 'KSHV vBcl-2 / Bcl-xL'),
    ('2VGA', '1CA9', 'Variola CrmB / TNFR2'),
    ('1F5Q', '1B7T', 'KSHV vCyclin / CyclinD2'),
    ('2BBR', '1A1W', 'MC159 / FADD DED'),
]


# ============================================================================
# Helpers
# ============================================================================

def find_pdb(pdb_id):
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None


def extract_residues_from_pdb(pdb_path):
    """Extract residue numbers and names from CA atoms."""
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    residues = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                chain = line[21].strip()
                aa = three_to_one.get(resname, 'X')
                residues.append((resnum, aa, chain))
    return residues


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


# ============================================================================
# Attention Extraction
# ============================================================================

def extract_attention_weights(model, data):
    """
    Extract cross-attention weights from the model for a single protein.
    
    Returns attention weights showing which residues are most attended to
    during the geometric-semantic fusion step.
    """
    with torch.no_grad():
        # Step 1: Input projection
        if data.x.dim() == 2 and data.x.size(1) == model.input_dim:
            seq_emb_raw = data.x.float()
            h = model.input_projector(seq_emb_raw)
        else:
            return None, None
        
        # Step 2: Run through EGNN
        h_out, _ = model.egnn(h, data.pos, data.edge_index)
        
        # Step 3: Extract cross-attention weights
        geom_emb = h_out.unsqueeze(0)           # [1, N, 32]
        seq_proj = model.seq_proj(seq_emb_raw.unsqueeze(0))  # [1, N, 32]
        
        # Run cross-attention with output weights
        attn_out, attn_weights = model.cross_attention(
            query=geom_emb,
            key=seq_proj,
            value=seq_proj,
            need_weights=True,
            average_attn_weights=True  # Average across heads
        )
        # attn_weights: [1, N, N] - attention from each residue to each residue
        
        # Per-residue importance: mean attention received (how much each residue is attended to)
        per_residue_importance = attn_weights.squeeze(0).mean(dim=0).numpy()  # [N]
        
        return per_residue_importance, attn_weights.squeeze(0).numpy()


# ============================================================================
# Analysis
# ============================================================================

def analyze_pair(model, viral_id, human_id, name):
    """Analyze attention for a single mimicry pair."""
    viral_path = find_pdb(viral_id)
    human_path = find_pdb(human_id)
    
    if not viral_path or not human_path:
        return None
    
    viral_data = parse_pdb_to_pyg(viral_path, use_esm=True)
    human_data = parse_pdb_to_pyg(human_path, use_esm=True)
    
    viral_residues = extract_residues_from_pdb(viral_path)
    human_residues = extract_residues_from_pdb(human_path)
    
    # Extract attention weights
    viral_importance, viral_attn = extract_attention_weights(model, viral_data)
    human_importance, human_attn = extract_attention_weights(model, human_data)
    
    if viral_importance is None or human_importance is None:
        return None
    
    # Normalize importance to [0, 1]
    viral_imp_norm = (viral_importance - viral_importance.min()) / (viral_importance.max() - viral_importance.min() + 1e-8)
    human_imp_norm = (human_importance - human_importance.min()) / (human_importance.max() - human_importance.min() + 1e-8)
    
    # Find top-5 attended residues
    n_top = min(5, len(viral_residues), len(human_residues))
    
    viral_top_idx = np.argsort(viral_imp_norm)[-n_top:][::-1]
    human_top_idx = np.argsort(human_imp_norm)[-n_top:][::-1]
    
    return {
        'name': name,
        'viral_id': viral_id,
        'human_id': human_id,
        'viral_residues': viral_residues,
        'human_residues': human_residues,
        'viral_importance': viral_imp_norm,
        'human_importance': human_imp_norm,
        'viral_top_idx': viral_top_idx,
        'human_top_idx': human_top_idx,
    }


# ============================================================================
# Visualization 
# ============================================================================

def create_attention_heatmap(all_results, output_path):
    """Create summary attention heatmap for all pairs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_pairs = len(all_results)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(18, 3 * n_pairs))
        
        if n_pairs == 1:
            axes = [axes]
        
        for idx, result in enumerate(all_results):
            # Viral attention
            ax_v = axes[idx][0] if n_pairs > 1 else axes[0]
            imp_v = result['viral_importance']
            ax_v.bar(range(len(imp_v)), imp_v, color='#e74c3c', alpha=0.7, width=1.0)
            ax_v.set_ylabel('Attention', fontsize=8)
            ax_v.set_title(f"{result['viral_id']} (Viral)", fontsize=9, fontweight='bold')
            
            # Mark top residues
            for i in result['viral_top_idx']:
                if i < len(result['viral_residues']):
                    res = result['viral_residues'][i]
                    ax_v.annotate(f'{res[1]}{res[0]}', (i, imp_v[i]),
                                fontsize=6, ha='center', va='bottom',
                                fontweight='bold', color='darkred')
            
            # Human attention
            ax_h = axes[idx][1] if n_pairs > 1 else axes[1]
            imp_h = result['human_importance']
            ax_h.bar(range(len(imp_h)), imp_h, color='#2ecc71', alpha=0.7, width=1.0)
            ax_h.set_ylabel('Attention', fontsize=8)
            ax_h.set_title(f"{result['human_id']} (Human Target)", fontsize=9, fontweight='bold')
            
            for i in result['human_top_idx']:
                if i < len(result['human_residues']):
                    res = result['human_residues'][i]
                    ax_h.annotate(f'{res[1]}{res[0]}', (i, imp_h[i]),
                                fontsize=6, ha='center', va='bottom',
                                fontweight='bold', color='darkgreen')
            
            # Add pair name
            ax_v.text(-0.1, 0.5, result['name'], transform=ax_v.transAxes,
                     fontsize=8, va='center', ha='right', rotation=0, fontweight='bold')
        
        plt.suptitle('GeoMimic-Net Cross-Attention: Residue-Level Importance',
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Saved attention heatmap to {output_path}")
        
    except ImportError:
        print("\n[WARN] matplotlib not available, skipping heatmap")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase C: Cross-Attention Interpretability Analysis")
    print("=" * 70)
    
    model = load_model()
    
    all_results = []
    
    print(f"\n{'Pair':<28s}  {'Top Viral Residues':<30s}  {'Top Human Residues'}")
    print("-" * 95)
    
    for viral_id, human_id, name in TRUE_PAIRS:
        result = analyze_pair(model, viral_id, human_id, name)
        
        if result is None:
            print(f"  {name:<26s}  [SKIPPED]")
            continue
        
        all_results.append(result)
        
        # Format top residues
        viral_top_str = ', '.join(
            f"{result['viral_residues'][i][1]}{result['viral_residues'][i][0]}"
            for i in result['viral_top_idx']
            if i < len(result['viral_residues'])
        )
        human_top_str = ', '.join(
            f"{result['human_residues'][i][1]}{result['human_residues'][i][0]}"
            for i in result['human_top_idx']
            if i < len(result['human_residues'])
        )
        
        print(f"  {name:<26s}  {viral_top_str:<30s}  {human_top_str}")
    
    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETABILITY SUMMARY")
    print("=" * 70)
    print(f"  Analyzed {len(all_results)} mimicry pairs")
    print(f"  Extracted top-5 attended residues per protein per pair")
    print(f"  These residues highlight the structural features driving")
    print(f"  the model's mimicry predictions, providing biological")
    print(f"  insight into the molecular mimicry mechanism.")
    
    # Create visualization (top 6 pairs for readability)
    output_path = os.path.join(OUTPUT_DIR, 'attention_analysis.png')
    create_attention_heatmap(all_results[:6], output_path)
    
    print("\n" + "=" * 70)
    print("Phase C Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
