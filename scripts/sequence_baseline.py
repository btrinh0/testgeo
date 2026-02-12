"""
Phase 23: Sequence Baseline Comparison

Compares traditional sequence alignment (Smith-Waterman) against 
GeoMimic-Net structural similarity for all 16 mimicry pairs.

Hypothesis: Sequence identity ~15-20%, GeoMimic-Net ~95%+
This demonstrates that molecular mimicry is primarily structural,
not sequence-based, validating our geometric approach.
"""

import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Bio import Align
from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
OUTPUT_PATH = 'results/baseline_comparison.png'
PDB_DIRS = ['data/raw', 'data/benchmark/positive', 'data/benchmark/negative']

# 16 validated mimicry pairs
TRUE_PAIRS = [
    ('1Q59', 'EBV BHRF1',        '1G5M', 'Bcl-2',         'Anti-apoptosis'),
    ('2V5I', 'Vaccinia A52',     '1LB5', 'TRAF6',          'Signaling hijack'),
    ('3CL3', 'KSHV vFLIP',      '3H11', 'FLIP',            'Death domain'),
    ('2GX9', 'Flu NS1',          '1KX5', 'Histone H3',     'Epigenetic'),
    ('2JBY', 'Myxoma M11L',     '1G5M', 'Bcl-2',           'Anti-apoptosis'),
    ('1B4C', 'Vaccinia B15',    '1ITB', 'IL-1R',            'Decoy Receptor'),
    ('1FV1', 'EBV LMP1',        '1CDF', 'CD40',             'Receptor Mimic'),
    ('1H26', 'Adenovirus E1A',  '1CF7', 'E2F',              'Cell Cycle'),
    ('1GUX', 'HPV E7',          '1CF7', 'E2F',              'Cell Cycle'),
    ('1EFN', 'HIV Nef',         '1SHF', 'Fyn SH3',          'Kinase Mimic'),
    ('3D2U', 'CMV UL18',        '1HHK', 'MHC-I HLA-A',     'Immune Evasion'),
    ('2UWI', 'Cowpox CrmE',     '1EXT', 'TNFR1',            'Decoy Receptor'),
    ('2BZR', 'KSHV vBcl-2',    '1MAZ', 'Bcl-xL',           'Anti-apoptosis'),
    ('2VGA', 'Variola CrmB',   '1CA9', 'TNFR2',             'Decoy Receptor'),
    ('1F5Q', 'KSHV vCyclin',   '1B7T', 'Cyclin D2',         'Cell Cycle'),
    ('2BBR', 'Molluscum MC159', '1A1W', 'FADD DED',         'Death Domain'),
]

# Standard amino acid 3-to-1 letter mapping
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
}


# ============================================================================
# Helpers
# ============================================================================

def find_pdb(pdb_id):
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None


def extract_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB file (first chain, CA atoms)."""
    residues = []
    seen = set()
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                resname = line[17:20].strip()
                chain = line[21]
                resnum = line[22:26].strip()
                key = (chain, resnum)
                
                if key not in seen and resname in AA_MAP:
                    seen.add(key)
                    residues.append((chain, int(resnum), AA_MAP[resname]))
    
    if not residues:
        return ""
    
    # Use first chain only
    first_chain = residues[0][0]
    seq = ''.join(r[2] for r in residues if r[0] == first_chain)
    return seq


def compute_sequence_identity(seq1, seq2):
    """Compute sequence identity using Smith-Waterman local alignment."""
    if not seq1 or not seq2:
        return 0.0
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    alignments = aligner.align(seq1, seq2)
    
    if not alignments:
        return 0.0
    
    best = alignments[0]
    
    # Count identical positions in the alignment
    aligned_seq1 = str(best).split('\n')[0]
    aligned_seq2 = str(best).split('\n')[2]
    
    # Calculate identity over aligned region
    matches = 0
    aligned_len = 0
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a != '-' and b != '-':
            aligned_len += 1
            if a == b:
                matches += 1
    
    if aligned_len == 0:
        return 0.0
    
    # Identity = matches / length of shorter sequence (standard definition)
    shorter = min(len(seq1), len(seq2))
    identity = min(matches / shorter * 100, 100.0)
    
    return identity


def load_model():
    model = SiameseEGNN(
        node_dim=32, edge_dim=0, hidden_dim=64,
        embed_dim=128, num_layers=4, geom_dim=32
    )
    state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_geomimic_score(model, data1, data2):
    with torch.no_grad():
        emb1 = model.forward_one(data1)
        emb2 = model.forward_one(data2)
        return F.cosine_similarity(emb1, emb2).item()


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_chart(pair_labels, seq_ids, geomimic_scores, output_path):
    """Create side-by-side bar chart."""
    n = len(pair_labels)
    x = np.arange(n)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Bars
    bars1 = ax.bar(x - width/2, seq_ids, width, label='Sequence Identity (%)',
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [s * 100 for s in geomimic_scores], width,
                   label='GeoMimic-Net Similarity (%)',
                   color='#27ae60', edgecolor='black', linewidth=0.5)
    
    # Value labels on bars
    for bar, val in zip(bars1, seq_ids):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold', color='#2c3e50')
    
    for bar, val in zip(bars2, geomimic_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 100 + 1,
               f'{val:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold', color='#2c3e50')
    
    # Averages
    avg_seq = np.mean(seq_ids)
    avg_geo = np.mean(geomimic_scores) * 100
    ax.axhline(y=avg_seq, color='#3498db', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=avg_geo, color='#27ae60', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(n - 0.5, avg_seq + 1, f'Avg Seq: {avg_seq:.1f}%', color='#3498db', fontsize=9, fontweight='bold')
    ax.text(n - 0.5, avg_geo + 1, f'Avg Geo: {avg_geo:.1f}%', color='#27ae60', fontsize=9, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Mimicry Pair', fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Sequence Identity vs GeoMimic-Net: Structural Mimicry is Invisible to BLAST',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved comparison chart to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 23: Sequence Baseline Comparison")
    print("=" * 60)
    
    model = load_model()
    
    pair_labels = []
    seq_identities = []
    geomimic_scores = []
    
    print(f"\n{'Pair':<35} {'Seq ID%':>8}  {'GeoMimic':>9}  {'Winner'}")
    print("-" * 70)
    
    for viral_id, viral_name, human_id, human_name, mimicry_type in TRUE_PAIRS:
        viral_path = find_pdb(viral_id)
        human_path = find_pdb(human_id)
        
        if not viral_path or not human_path:
            print(f"  [SKIP] {viral_id} or {human_id} not found")
            continue
        
        # Sequence identity
        seq1 = extract_sequence_from_pdb(viral_path)
        seq2 = extract_sequence_from_pdb(human_path)
        seq_id = compute_sequence_identity(seq1, seq2)
        
        # GeoMimic-Net score
        data1 = parse_pdb_to_pyg(viral_path, use_esm=True)
        data2 = parse_pdb_to_pyg(human_path, use_esm=True)
        geo_score = compute_geomimic_score(model, data1, data2)
        
        label = f"{viral_name} / {human_name}"
        pair_labels.append(label)
        seq_identities.append(seq_id)
        geomimic_scores.append(geo_score)
        
        winner = "GeoMimic" if geo_score * 100 > seq_id else "Sequence"
        print(f"  {label:<33} {seq_id:>7.1f}%  {geo_score:>8.4f}  {winner}")
    
    # Summary
    avg_seq = np.mean(seq_identities)
    avg_geo = np.mean(geomimic_scores)
    
    print("\n" + "=" * 70)
    print(f"{'AVERAGES':<35} {avg_seq:>7.1f}%  {avg_geo:>8.4f}")
    print("=" * 70)
    print(f"\n  Sequence alignment detects:   {avg_seq:.1f}% average identity")
    print(f"  GeoMimic-Net detects:         {avg_geo*100:.1f}% average similarity")
    print(f"  GeoMimic advantage:           {avg_geo*100 - avg_seq:.1f} percentage points")
    
    # Create chart
    print("\n--- Creating Comparison Chart ---")
    create_comparison_chart(pair_labels, seq_identities, geomimic_scores, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("Phase 23 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
