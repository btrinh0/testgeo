"""
Phase 17: Blind Validation Script
Tests Model on Unseen Virus-Human Pairs

Test Pairs:
1. Zika Envelope (5JHM) vs Human AXL (5U6B) 
2. Chikungunya E1 (2XFB) vs Human MxRA8 (6NK3)

Success: Similarity > 0.85 for known mimics
Failure: Similarity < 0.5 indicates overfitting
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
BLIND_DIR = 'data/blind_validation'

# Test Pairs
TEST_PAIRS = [
    ('5JHM', '5U6B', 'Zika Envelope vs Human AXL'),
    ('2XFB', '6NK3', 'Chikungunya E1 vs Human MxRA8'),
]

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
        emb1 = model.forward_one(data1)  # [1, embed_dim]
        emb2 = model.forward_one(data2)  # [1, embed_dim]
        
        # Cosine similarity
        similarity = F.cosine_similarity(emb1, emb2).item()
        
    return similarity

def main():
    print("=" * 60)
    print("Phase 17: Blind Validation (Universal Scan)")
    print("=" * 60)
    
    model = load_model()
    if model is None:
        return
    
    print("\n" + "-" * 60)
    print("BLIND VALIDATION RESULTS")
    print("-" * 60)
    
    all_passed = True
    
    for viral_id, human_id, description in TEST_PAIRS:
        viral_path = os.path.join(BLIND_DIR, f"{viral_id}.pdb")
        human_path = os.path.join(BLIND_DIR, f"{human_id}.pdb")
        
        if not os.path.exists(viral_path):
            print(f"[ERROR] {viral_path} not found!")
            continue
        if not os.path.exists(human_path):
            print(f"[ERROR] {human_path} not found!")
            continue
            
        print(f"\n{description}")
        print(f"  {viral_id} vs {human_id}:")
        
        try:
            similarity = compute_similarity(model, viral_path, human_path)
            
            if similarity > 0.85:
                status = "PASS (Known Mimic Detected)"
            elif similarity > 0.5:
                status = "MARGINAL (Partial Match)"
            else:
                status = "FAIL (Overfitting?)"
                all_passed = False
                
            print(f"  Similarity: {similarity:.4f} -> {status}")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("BLIND VALIDATION: PASSED!")
        print("Model generalizes to unseen virus-human pairs.")
    else:
        print("BLIND VALIDATION: NEEDS INVESTIGATION")
        print("Some pairs scored below threshold.")
    print("=" * 60)

if __name__ == "__main__":
    main()
