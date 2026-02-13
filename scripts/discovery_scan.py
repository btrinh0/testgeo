"""
Tier 2A: Novel Mimicry Discovery Scan

Scans viral proteins against human proteome representatives to discover
previously unknown molecular mimicry candidates.
"""

import os
import sys
import torch
import torch.nn.functional as F
import urllib.request
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
SCAN_DIR = 'data/discovery_scan'
PDB_DIRS = ['data/raw', 'data/benchmark/positive', 'data/benchmark/negative', 
            'data/blind_validation', SCAN_DIR]
PDB_BASE_URL = 'https://files.rcsb.org/download/'

NOVEL_VIRAL = [
    ('6VXX', 'SARS-CoV-2 Spike (full)'),
    ('7BZ5', 'SARS-CoV-2 Nucleocapsid'),
    ('5IRE', 'Zika NS5 methyltransferase'),
    ('2GTP', 'Dengue NS3 helicase'),
    ('3BSZ', 'Influenza PA endonuclease'),
    ('1YVB', 'Rotavirus VP7 glycoprotein'),
    ('2HIN', 'Hantavirus nucleoprotein'),
    ('3PHF', 'RSV fusion protein'),
]

HUMAN_REPRESENTATIVES = [
    ('1A3N', 'Hemoglobin'), ('1HHK', 'MHC-I HLA-A'), ('1BF5', 'STAT1'),
    ('1TSR', 'p53'), ('1UBQ', 'Ubiquitin'), ('1G5M', 'Bcl-2'),
    ('1CF7', 'E2F'), ('1SHF', 'Fyn SH3'), ('1LB5', 'TRAF6'),
    ('1CDF', 'CD40'), ('2ILK', 'IL-10'), ('1MAZ', 'Bcl-xL'),
    ('1B7T', 'Cyclin D2'), ('1EXT', 'TNFR1'), ('1ITB', 'IL-1R'),
]

def download_pdb(pdb_id, output_dir):
    filepath = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        return filepath
    try:
        urllib.request.urlretrieve(f"{PDB_BASE_URL}{pdb_id}.pdb", filepath)
        time.sleep(0.3)
        return filepath
    except Exception as e:
        print(f"  [ERROR] {pdb_id}: {e}")
        return None

def find_pdb(pdb_id):
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None

def main():
    print("=" * 70)
    print("Tier 2A: Novel Mimicry Discovery Scan")
    print("=" * 70)
    
    os.makedirs(SCAN_DIR, exist_ok=True)
    
    print("\nDownloading novel viral proteins...")
    for pdb_id, name in NOVEL_VIRAL:
        if not find_pdb(pdb_id):
            download_pdb(pdb_id, SCAN_DIR)
    
    print("\nLoading model...")
    model = SiameseEGNN(
        node_dim=64, edge_dim=0, hidden_dim=128,
        embed_dim=256, num_layers=4, geom_dim=64,
        num_rbf=16, dropout=0.1
    )
    if os.path.exists(WEIGHTS_PATH):
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("Loading protein structures...")
    graphs = {}
    for pdb_id, name in NOVEL_VIRAL + HUMAN_REPRESENTATIVES:
        path = find_pdb(pdb_id)
        if path:
            try:
                graphs[pdb_id] = parse_pdb_to_pyg(path, use_esm=True)
            except Exception as e:
                print(f"  [WARN] {pdb_id}: {e}")
    print(f"  Loaded {len(graphs)} structures")
    
    print("\n" + "=" * 70)
    print("SCANNING FOR NOVEL MIMICRY CANDIDATES")
    print("=" * 70)
    
    discoveries = []
    
    for viral_id, viral_name in NOVEL_VIRAL:
        if viral_id not in graphs:
            continue
        print(f"\n  Scanning {viral_name} ({viral_id}):")
        pair_scores = []
        for human_id, human_name in HUMAN_REPRESENTATIVES:
            if human_id not in graphs:
                continue
            with torch.no_grad():
                emb_v = model.forward_one(graphs[viral_id])
                emb_h = model.forward_one(graphs[human_id])
                sim = F.cosine_similarity(emb_v, emb_h).item()
            pair_scores.append((human_id, human_name, sim))
        
        pair_scores.sort(key=lambda x: x[2], reverse=True)
        for rank, (h_id, h_name, score) in enumerate(pair_scores[:3], 1):
            flag = " *** NOVEL CANDIDATE" if score > 0.8 else ""
            print(f"    #{rank}: {h_name} ({h_id}) = {score:+.4f}{flag}")
            if score > 0.8:
                discoveries.append({
                    'viral_name': viral_name, 'human_name': h_name, 'score': score
                })
    
    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)
    if discoveries:
        print(f"\n  Found {len(discoveries)} novel mimicry candidates (score > 0.8):")
        for d in sorted(discoveries, key=lambda x: x['score'], reverse=True):
            print(f"  {d['viral_name']:<35s} {d['human_name']:<20s} {d['score']:>+8.4f}")
    else:
        print("\n  No strong novel candidates found (all scores < 0.8).")
    
    print("\n" + "=" * 70)
    print("Tier 2A Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
