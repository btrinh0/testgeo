"""
Tier 2B: Functional Site Mapping - Attention Enrichment Analysis

Cross-references attention hotspot residues with known functional annotations
to prove the model learns biologically meaningful features.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
PDB_DIRS = ['data/raw', 'data/benchmark/positive', 'data/benchmark/negative']

FUNCTIONAL_ANNOTATIONS = {
    '1G5M': [(97, 107, 'BH3 binding groove'), (136, 155, 'BH1 domain'), (187, 202, 'BH2 domain')],
    '1CF7': [(11, 28, 'DNA binding domain'), (47, 62, 'Cyclin A binding')],
    '1SHF': [(85, 142, 'SH3 domain (PxxP binding)')],
    '1HHK': [(1, 90, 'Alpha-1 (peptide binding)'), (91, 182, 'Alpha-2 (peptide binding)'), (183, 275, 'Alpha-3 (CD8 binding)')],
    '1MAZ': [(85, 98, 'BH3 binding groove'), (130, 147, 'BH1 domain')],
    '1LB5': [(346, 504, 'TRAF-C domain')],
    '1A1W': [(1, 83, 'Death effector domain')],
    '1B7T': [(20, 150, 'Cyclin box fold'), (151, 280, 'C-terminal cyclin fold')],
    '1TSR': [(94, 293, 'DNA-binding domain'), (100, 104, 'L1 loop'), (236, 251, 'L3 loop')],
}

TRUE_PAIRS = [
    ('1Q59', '1G5M', 'EBV BHRF1 / Bcl-2'),
    ('1EFN', '1SHF', 'HIV Nef / Fyn SH3'),
    ('3D2U', '1HHK', 'CMV UL18 / MHC-I'),
    ('2BZR', '1MAZ', 'KSHV vBcl-2 / Bcl-xL'),
    ('1H26', '1CF7', 'Adeno E1A / E2F'),
    ('2V5I', '1LB5', 'Vaccinia A52 / TRAF6'),
    ('2BBR', '1A1W', 'MC159 / FADD DED'),
    ('1F5Q', '1B7T', 'KSHV vCyclin / CyclinD2'),
    ('4GIZ', '1TSR', 'HPV E6 / p53'),
]

def find_pdb(pdb_id):
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None

def extract_residue_numbers(pdb_path):
    residues = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                residues.append(int(line[22:26].strip()))
    return residues

def extract_attention(model, data):
    with torch.no_grad():
        if data.x.dim() == 2 and data.x.size(1) == model.input_dim:
            seq_emb_raw = data.x.float()
            h = model.input_projector(seq_emb_raw)
        else:
            return None
        h_out, _ = model.egnn(h, data.pos, data.edge_index)
        geom_emb = h_out.unsqueeze(0)
        seq_proj = model.seq_proj(seq_emb_raw.unsqueeze(0))
        _, attn_weights = model.cross_attention(
            query=geom_emb, key=seq_proj, value=seq_proj,
            need_weights=True, average_attn_weights=True
        )
        return attn_weights.squeeze(0).mean(dim=0).numpy()

def main():
    print("=" * 70)
    print("Tier 2B: Functional Site Mapping (Attention Enrichment)")
    print("=" * 70)
    
    model = SiameseEGNN(
        node_dim=64, edge_dim=0, hidden_dim=128,
        embed_dim=256, num_layers=4, geom_dim=64,
        num_rbf=16, dropout=0.1
    )
    if os.path.exists(WEIGHTS_PATH):
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"\n{'Pair':<28s} {'Enrichment':>12s} {'Func Attn':>10s} {'Non-Func':>10s} {'Result'}")
    print("-" * 80)
    
    enrichments = []
    
    for viral_id, human_id, name in TRUE_PAIRS:
        if human_id not in FUNCTIONAL_ANNOTATIONS:
            continue
        human_path = find_pdb(human_id)
        if not human_path:
            continue
        try:
            data = parse_pdb_to_pyg(human_path, use_esm=True)
            attention = extract_attention(model, data)
            if attention is None:
                continue
            
            residue_numbers = extract_residue_numbers(human_path)
            annotations = FUNCTIONAL_ANNOTATIONS[human_id]
            
            func_indices = set()
            for start, end, _ in annotations:
                for i, resnum in enumerate(residue_numbers):
                    if start <= resnum <= end:
                        func_indices.add(i)
            
            if not func_indices or len(func_indices) >= len(attention):
                continue
            
            func_attn = np.mean([attention[i] for i in func_indices if i < len(attention)])
            non_func_attn = np.mean([attention[i] for i in range(len(attention)) if i not in func_indices])
            enrichment = func_attn / (non_func_attn + 1e-8)
            
            enrichments.append(enrichment)
            verdict = "ENRICHED" if enrichment > 1.0 else "not enriched"
            print(f"  {name:<26s} {enrichment:>10.2f}x  {func_attn:>10.4f} {non_func_attn:>10.4f}  {verdict}")
        except Exception as e:
            print(f"  {name:<26s} [ERROR: {e}]")
    
    if enrichments:
        print("\n" + "=" * 70)
        print("FUNCTIONAL ENRICHMENT SUMMARY")
        print("=" * 70)
        enriched = sum(1 for e in enrichments if e > 1.0)
        print(f"  Pairs analyzed:   {len(enrichments)}")
        print(f"  Enriched (>1.0x): {enriched}/{len(enrichments)} ({100*enriched/len(enrichments):.0f}%)")
        print(f"  Mean enrichment:  {np.mean(enrichments):.2f}x")
        if np.mean(enrichments) > 1.0:
            print(f"\n  [SIGNIFICANT] Model pays {np.mean(enrichments):.1f}x MORE attention")
            print(f"  to known functional sites than non-functional regions.")
    
    print("\n" + "=" * 70)
    print("Tier 2B Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
