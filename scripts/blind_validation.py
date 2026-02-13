"""
Phase D: Expanded Blind Validation
Tests model on UNSEEN virus-human pairs not in training data.

6 Test Pairs (held-out from training):
1. Zika Envelope (5JHM) vs Human AXL (5U6B) - Zika entry mimicry
2. Chikungunya E1 (2XFB) vs Human MxRA8 (6NK3) - Alphavirus entry
3. SARS-CoV-2 Spike RBD (6M0J) vs Human ACE2 (1R42) - COVID receptor mimicry
4. HIV gp120 (3JWD) vs Human CD4 (1WIO) - HIV entry
5. Hepatitis C NS3 (1A1V) vs Human TBK1 (4IWO) - Immune evasion
6. Epstein-Barr EBNA1 (1B3T) vs Human HMGB1 (1CKT) - Nuclear mimicry

Negative Controls (should NOT match):
- All viral proteins vs 3 random unrelated proteins
"""

import os
import sys
import urllib.request
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
BLIND_DIR = 'data/blind_validation'

# 6 Held-Out Test Pairs
TEST_PAIRS = [
    ('5JHM', '5U6B', 'Zika Envelope vs Human AXL'),
    ('2XFB', '6NK3', 'Chikungunya E1 vs Human MxRA8'),
    ('6M0J', '1R42', 'SARS-CoV-2 RBD vs ACE2'),
    ('3JWD', '1WIO', 'HIV gp120 vs CD4'),
    ('1A1V', '4IWO', 'HCV NS3 vs TBK1'),
    ('1B3T', '1CKT', 'EBV EBNA1 vs HMGB1'),
]

# Negative control proteins for blind set
BLIND_NEGATIVES = ['1UBQ', '1LYZ', '1MBN']

PDB_BASE_URL = 'https://files.rcsb.org/download/'


# ============================================================================
# PDB Download
# ============================================================================

def download_pdb(pdb_id, output_dir):
    """Download PDB file if not already present."""
    filepath = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        return filepath
    
    url = f"{PDB_BASE_URL}{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"  [DL] {pdb_id}: {size:,} bytes")
        return filepath
    except Exception as e:
        print(f"  [ERROR] {pdb_id}: {e}")
        return None


# ============================================================================
# Model
# ============================================================================

def load_model():
    model = SiameseEGNN(
        node_dim=32, edge_dim=0, hidden_dim=64,
        embed_dim=128, num_layers=4, geom_dim=32
    )
    if os.path.exists(WEIGHTS_PATH):
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"[ERROR] {WEIGHTS_PATH} not found!")
        return None
    model.eval()
    return model


def compute_similarity(model, pdb1_path, pdb2_path):
    """Compute whole-protein similarity."""
    data1 = parse_pdb_to_pyg(pdb1_path, use_esm=True)
    data2 = parse_pdb_to_pyg(pdb2_path, use_esm=True)
    with torch.no_grad():
        emb1 = model.forward_one(data1)
        emb2 = model.forward_one(data2)
        similarity = F.cosine_similarity(emb1, emb2).item()
    return similarity


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase D: Expanded Blind Validation (6 Pairs + Negative Controls)")
    print("=" * 70)
    
    # Ensure directory exists
    os.makedirs(BLIND_DIR, exist_ok=True)
    
    # Download all required PDBs
    print("\nDownloading PDB files...")
    all_ids = set()
    for v, h, _ in TEST_PAIRS:
        all_ids.add(v)
        all_ids.add(h)
    for n in BLIND_NEGATIVES:
        all_ids.add(n)
    
    pdb_paths = {}
    for pdb_id in sorted(all_ids):
        # First check existing directories
        for d in ['data/raw', 'data/benchmark/positive', 'data/benchmark/negative', BLIND_DIR]:
            p = os.path.join(d, f"{pdb_id}.pdb")
            if os.path.exists(p) and os.path.getsize(p) > 100:
                pdb_paths[pdb_id] = p
                break
        
        if pdb_id not in pdb_paths:
            path = download_pdb(pdb_id, BLIND_DIR)
            if path:
                pdb_paths[pdb_id] = path
    
    print(f"  Available: {len(pdb_paths)}/{len(all_ids)} PDBs")
    
    # Load model
    print("\nLoading model...")
    model = load_model()
    if model is None:
        return
    
    # ====================================================================
    # Test True Pairs
    # ====================================================================
    print("\n" + "-" * 70)
    print("BLIND VALIDATION: TRUE PAIRS")
    print("-" * 70)
    
    pair_results = []
    for viral_id, human_id, description in TEST_PAIRS:
        viral_path = pdb_paths.get(viral_id)
        human_path = pdb_paths.get(human_id)
        
        if not viral_path or not human_path:
            print(f"  [SKIP] {description}: PDB missing")
            continue
        
        try:
            similarity = compute_similarity(model, viral_path, human_path)
            
            if similarity > 0.85:
                status = "PASS (Mimicry Detected)"
            elif similarity > 0.5:
                status = "MARGINAL"
            elif similarity > 0.0:
                status = "WEAK"
            else:
                status = "FAIL"
            
            pair_results.append({
                'name': description,
                'score': similarity,
                'detected': similarity > 0.5,
            })
            
            print(f"  {description:<35s} | {similarity:+.4f} | {status}")
            
        except Exception as e:
            print(f"  [ERROR] {description}: {e}")
    
    # ====================================================================
    # Test Negative Controls
    # ====================================================================
    print("\n" + "-" * 70)
    print("BLIND VALIDATION: NEGATIVE CONTROLS")
    print("-" * 70)
    
    neg_results = []
    for viral_id, _, description in TEST_PAIRS:
        viral_path = pdb_paths.get(viral_id)
        if not viral_path:
            continue
        
        for neg_id in BLIND_NEGATIVES:
            neg_path = pdb_paths.get(neg_id)
            if not neg_path:
                continue
            
            try:
                similarity = compute_similarity(model, viral_path, neg_path)
                rejected = similarity < 0.5
                
                neg_results.append({
                    'viral': viral_id,
                    'neg': neg_id,
                    'score': similarity,
                    'rejected': rejected,
                })
                
                status = "TN (Rejected)" if rejected else "FP (False Match)"
                print(f"  {viral_id} vs {neg_id}: {similarity:+.4f} | {status}")
                
            except Exception as e:
                print(f"  [ERROR] {viral_id} vs {neg_id}: {e}")
    
    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("BLIND VALIDATION SUMMARY")
    print("=" * 70)
    
    if pair_results:
        detected = sum(1 for r in pair_results if r['detected'])
        avg_score = sum(r['score'] for r in pair_results) / len(pair_results)
        print(f"  True Pairs Detected: {detected}/{len(pair_results)}")
        print(f"  Average True Score:  {avg_score:+.4f}")
    
    if neg_results:
        rejected = sum(1 for r in neg_results if r['rejected'])
        avg_neg = sum(r['score'] for r in neg_results) / len(neg_results)
        print(f"  Negatives Rejected:  {rejected}/{len(neg_results)}")
        print(f"  Average Neg Score:   {avg_neg:+.4f}")
    
    if pair_results and neg_results:
        sensitivity = detected / len(pair_results) * 100
        specificity = rejected / len(neg_results) * 100
        print(f"\n  Blind Sensitivity:   {sensitivity:.1f}%")
        print(f"  Blind Specificity:   {specificity:.1f}%")
    
    print("\n" + "=" * 70)
    print("Phase D Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
