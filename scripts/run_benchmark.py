"""
Phase 11: Benchmark Script (Fixed)
Evaluates the SiameseEGNN model on a benchmark dataset of known mimicry pairs.

This script:
1. Loads the trained SiameseEGNN model with Cross-Attention weights
2. Processes PDB files from benchmark folders
3. Uses correct ground truth labels for mimicry pairs
4. Computes accuracy and outputs a confusion matrix
"""

import os
import sys
import random
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg


# ============================================================================
# Configuration
# ============================================================================

POSITIVE_DIR = 'data/benchmark/positive'
NEGATIVE_DIR = 'data/benchmark/negative'
RAW_DIR = 'data/raw'
WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'  # Phase 15: Supervised Fine-Tuning
SIMILARITY_THRESHOLD = 0.85  # Above this = Match (Predicted Positive)

# Directories to search for PDB files
PDB_DIRS = [RAW_DIR, POSITIVE_DIR, NEGATIVE_DIR]

# ============================================================================
# GROUND TRUTH: 16 validated mimicry pairs
# ============================================================================
TRUE_PAIRS = [
    # Original 4
    ('1Q59', '1G5M'),  # EBV BHRF1 -> Bcl-2
    ('2V5I', '1LB5'),  # Vaccinia A52 -> TRAF6
    ('3CL3', '3H11'),  # KSHV vFLIP -> FLIP
    ('2GX9', '1KX5'),  # Flu NS1 -> Histone H3
    # Expanded 12
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

# Viral proteins (known to mimic human proteins)
VIRAL_PDBS = sorted(set(v for v, h in TRUE_PAIRS))

# Human proteins (mimicry targets)
HUMAN_PDBS = sorted(set(h for v, h in TRUE_PAIRS))

# Negative controls (10 unrelated proteins)
NEGATIVE_PDBS = ['1A3N', '1TRZ', '1MBN', '1UBQ', '1LYZ', '1EMA', '4INS', '1CLL', '7RSA', '1HRC']


# ============================================================================
# Helper Functions
# ============================================================================

def load_model():
    """Load the trained SiameseEGNN model."""
    print("Loading SiameseEGNN model...")
    
    # Use same parameters as Phase 15 trained model
    model = SiameseEGNN(
        node_dim=64,
        edge_dim=0,
        hidden_dim=64,
        embed_dim=128,
        num_layers=4,
        geom_dim=32
    )
    
    # Load trained weights
    if os.path.exists(WEIGHTS_PATH):
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded weights from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"  WARNING: Could not load weights ({e})")
            print("  Using random weights - results will be meaningless!")
    else:
        print(f"  WARNING: {WEIGHTS_PATH} not found!")
        print("  Using random weights - results will be meaningless!")
    
    model.eval()
    return model


def find_pdb(pdb_id):
    """Search multiple directories for a PDB file."""
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None


def load_pdb_to_graph(pdb_id):
    """Find and load a PDB file from any search directory."""
    path = find_pdb(pdb_id)
    if path is None:
        print(f"  ERROR: {pdb_id}.pdb not found in any directory")
        return None
    try:
        data = parse_pdb_to_pyg(path, use_esm=True)
        return data
    except Exception as e:
        print(f"  ERROR loading {pdb_id}: {e}")
        return None


def compute_similarity(model, data1, data2):
    """
    Compute cosine similarity between two protein graphs.
    
    Returns:
        float: Cosine similarity score in range [-1, 1]
    """
    with torch.no_grad():
        # Get embeddings
        z1 = model.forward_one(data1)  # [1, embed_dim]
        z2 = model.forward_one(data2)  # [1, embed_dim]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(z1, z2, dim=-1)
        
        return similarity.item()


def is_true_pair(viral_id, human_id):
    """Check if (viral_id, human_id) is a known true mimicry pair."""
    return (viral_id, human_id) in TRUE_PAIRS


# ============================================================================
# Benchmark Function
# ============================================================================

def run_benchmark(model):
    """
    Run comprehensive benchmark with correct ground truth labels.
    
    Ground Truth Logic:
    - If (viral_id, human_id) is in TRUE_PAIRS -> label = 1 (Positive)
    - If they are NOT in TRUE_PAIRS -> label = 0 (Negative)
    - Any comparison against negative/ folder -> label = 0 (Negative)
    
    Returns:
        List of (pair_desc, similarity, ground_truth, prediction) tuples
    """
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"\nGround Truth Pairs (TRUE mimicry):")
    for viral, human in TRUE_PAIRS:
        print(f"  {viral} -> {human}")
    
    results = []
    
    # ====== Load all graphs ======
    print("\n" + "-" * 60)
    print("Loading Viral proteins...")
    viral_graphs = {}
    for pdb_id in VIRAL_PDBS:
        graph = load_pdb_to_graph(pdb_id)
        if graph is not None:
            viral_graphs[pdb_id] = graph
            print(f"  {pdb_id}: {graph.x.size(0)} atoms")
    
    print("\nLoading Human proteins...")
    human_graphs = {}
    for pdb_id in HUMAN_PDBS:
        graph = load_pdb_to_graph(pdb_id)
        if graph is not None:
            human_graphs[pdb_id] = graph
            print(f"  {pdb_id}: {graph.x.size(0)} atoms")
    
    print("\nLoading Negative controls...")
    negative_graphs = {}
    for pdb_id in NEGATIVE_PDBS:
        graph = load_pdb_to_graph(pdb_id)
        if graph is not None:
            negative_graphs[pdb_id] = graph
            print(f"  {pdb_id}: {graph.x.size(0)} atoms")
    
    # ====== Compare ALL Viral vs Human pairs ======
    print("\n" + "=" * 60)
    print("PART 1: Viral vs Human (positive/ folder)")
    print("=" * 60)
    print(f"Threshold: {SIMILARITY_THRESHOLD}")
    print("-" * 60)
    
    for viral_id, viral_graph in viral_graphs.items():
        for human_id, human_graph in human_graphs.items():
            similarity = compute_similarity(model, viral_graph, human_graph)
            
            # Ground Truth: Only TRUE_PAIRS are positives
            ground_truth = 1 if is_true_pair(viral_id, human_id) else 0
            
            # Prediction: Based on threshold
            prediction = 1 if similarity > SIMILARITY_THRESHOLD else 0
            
            # Classify result
            if ground_truth == 1 and prediction == 1:
                result_type = "TP"
            elif ground_truth == 1 and prediction == 0:
                result_type = "FN"
            elif ground_truth == 0 and prediction == 1:
                result_type = "FP"
            else:
                result_type = "TN"
            
            pair_desc = f"{viral_id} vs {human_id}"
            results.append((pair_desc, similarity, ground_truth, prediction, result_type))
            
            gt_str = "[TRUE PAIR]" if ground_truth == 1 else "[not a pair]"
            print(f"  {pair_desc}: {similarity:.4f} -> {result_type} {gt_str}")
    
    # ====== Compare Viral vs Negative controls ======
    print("\n" + "=" * 60)
    print("PART 2: Viral vs Negative Controls")
    print("=" * 60)
    print("-" * 60)
    
    for viral_id, viral_graph in viral_graphs.items():
        for neg_id, neg_graph in negative_graphs.items():
            similarity = compute_similarity(model, viral_graph, neg_graph)
            
            # Ground Truth: Always negative (no mimicry with random proteins)
            ground_truth = 0
            
            # Prediction: Based on threshold
            prediction = 1 if similarity > SIMILARITY_THRESHOLD else 0
            
            # Classify result
            result_type = "FP" if prediction == 1 else "TN"
            
            pair_desc = f"{viral_id} vs {neg_id}"
            results.append((pair_desc, similarity, ground_truth, prediction, result_type))
            
            print(f"  {pair_desc}: {similarity:.4f} -> {result_type}")
    
    return results


def compute_confusion_matrix(results):
    """
    Compute and display the confusion matrix from results.
    """
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    
    # Count each type
    tp = sum(1 for _, _, _, _, rt in results if rt == "TP")
    fn = sum(1 for _, _, _, _, rt in results if rt == "FN")
    fp = sum(1 for _, _, _, _, rt in results if rt == "FP")
    tn = sum(1 for _, _, _, _, rt in results if rt == "TN")
    
    # Calculate metrics
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print matrix
    print("""
                        Predicted
                    Positive  Negative
    Actual Positive   TP={tp:3d}    FN={fn:3d}
    Actual Negative   FP={fp:3d}    TN={tn:3d}
    """.format(tp=tp, fn=fn, fp=fp, tn=tn))
    
    print("-" * 60)
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Negatives (FN): {fn}")
    print("-" * 60)
    print(f"  Total Pairs Tested: {total}")
    print(f"  - True Pairs: {len(TRUE_PAIRS)}")
    print(f"  - Non-Pairs (viral vs wrong human): {len(VIRAL_PDBS) * len(HUMAN_PDBS) - len(TRUE_PAIRS)}")
    print(f"  - Negative Controls: {len(VIRAL_PDBS) * len(NEGATIVE_PDBS)}")
    print("-" * 60)
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print("=" * 60)
    
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 20: Expanded Benchmark Evaluation (16 pairs + 10 negatives)")
    print("=" * 60)
    print(f"PDB search dirs: {PDB_DIRS}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    
    # Load model
    model = load_model()
    
    # Run benchmark
    results = run_benchmark(model)
    
    # Compute and display confusion matrix
    metrics = compute_confusion_matrix(results)
    
    print("\nBenchmark complete!")
    
    return metrics


if __name__ == "__main__":
    main()
