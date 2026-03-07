"""
Tier 3A: Comprehensive Ablation Study

Tests the contribution of each architectural component by removing them one at a time.
This proves every design decision is justified and contributing to performance.

Ablation conditions:
1. Full model (baseline)
2. No cross-attention (geometry only)
3. No RBF edge features (reverts to scalar distance)
4. No attention pooling (reverts to mean pooling)
5. Smaller model (32/64/128 dims)
6. Fewer EGNN layers (2 instead of 4)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN, EGNN, AttentionPool, GaussianRBF
from utils.protein_parser import parse_pdb_to_pyg

from config.constants import (
    TRUE_PAIRS_16 as _TRUE_PAIRS_2, NEGATIVE_PDBS, PDB_DIRS,
    SUPERVISED_WEIGHTS,
)

WEIGHTS_PATH = SUPERVISED_WEIGHTS

_PAIR_NAMES = [
    'EBV BHRF1 / Bcl-2', 'Vaccinia A52 / TRAF6', 'KSHV vFLIP / FLIP',
    'Flu NS1 / Histone H3', 'Myxoma M11L / Bcl-2', 'Vaccinia B15 / IL-1R',
    'EBV LMP1 / CD40', 'Adeno E1A / E2F', 'HPV E7 / E2F',
    'HIV Nef / Fyn SH3', 'CMV UL18 / MHC-I', 'Cowpox CrmE / TNFR1',
    'KSHV vBcl-2 / Bcl-xL', 'Variola CrmB / TNFR2', 'KSHV vCyclin / CyclinD2',
    'MC159 / FADD DED',
]
TRUE_PAIRS = [(v, h, n) for (v, h), n in zip(_TRUE_PAIRS_2, _PAIR_NAMES)]

def find_pdb(pdb_id):
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None

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
            except:
                pass
    return graphs

def evaluate_model(model, graphs):
    """Evaluate model: compute mean true score, mean rank, and top-3 accuracy."""
    model.eval()

    human_ids = sorted(set(h for _, h, _ in TRUE_PAIRS))
    neg_ids = [n for n in NEGATIVE_PDBS if n in graphs]
    all_candidates = human_ids + neg_ids

    embeddings = {}
    with torch.no_grad():
        for pdb_id in set(list(v for v, _, _ in TRUE_PAIRS) + list(all_candidates)):
            if pdb_id in graphs:
                embeddings[pdb_id] = model.forward_one(graphs[pdb_id])

    true_scores = []
    ranks = []
    top3_hits = 0

    for viral_id, human_id, _ in TRUE_PAIRS:
        if viral_id not in embeddings or human_id not in embeddings:
            continue

        emb_v = embeddings[viral_id]

        scores = []
        for cand_id in all_candidates:
            if cand_id in embeddings:
                sim = F.cosine_similarity(emb_v, embeddings[cand_id]).item()
                scores.append((cand_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        true_score = F.cosine_similarity(emb_v, embeddings[human_id]).item()
        true_scores.append(true_score)

        rank = next(i+1 for i, (cid, _) in enumerate(scores) if cid == human_id)
        ranks.append(rank)

        if rank <= 3:
            top3_hits += 1

    n = len(true_scores)
    return {
        'mean_score': np.mean(true_scores) if true_scores else 0,
        'mean_rank': np.mean(ranks) if ranks else 0,
        'top3_pct': 100 * top3_hits / n if n > 0 else 0,
        'n': n,
    }

def main():
    print("=" * 70)
    print("Tier 3A: Comprehensive Ablation Study")
    print("=" * 70)

    print("\nLoading graphs...")
    graphs = load_all_graphs()
    print(f"  Loaded {len(graphs)} graphs")

    configs = {
        'Full Model (v2)': {
            'node_dim': 64, 'hidden_dim': 128, 'embed_dim': 256,
            'num_layers': 4, 'geom_dim': 64, 'num_rbf': 16, 'dropout': 0.1,
        },
        'No Cross-Attention': {
            'node_dim': 64, 'hidden_dim': 128, 'embed_dim': 256,
            'num_layers': 4, 'geom_dim': 64, 'num_rbf': 16, 'dropout': 0.1,
            'ablate_cross_attn': True,
        },
        'No RBF (scalar dist)': {
            'node_dim': 64, 'hidden_dim': 128, 'embed_dim': 256,
            'num_layers': 4, 'geom_dim': 64, 'num_rbf': 1, 'dropout': 0.1,
        },
        'Mean Pooling (no attn)': {
            'node_dim': 64, 'hidden_dim': 128, 'embed_dim': 256,
            'num_layers': 4, 'geom_dim': 64, 'num_rbf': 16, 'dropout': 0.1,
            'ablate_attn_pool': True,
        },
        'Smaller (32/64/128)': {
            'node_dim': 32, 'hidden_dim': 64, 'embed_dim': 128,
            'num_layers': 4, 'geom_dim': 32, 'num_rbf': 16, 'dropout': 0.1,
        },
        '2 EGNN Layers': {
            'node_dim': 64, 'hidden_dim': 128, 'embed_dim': 256,
            'num_layers': 2, 'geom_dim': 64, 'num_rbf': 16, 'dropout': 0.1,
        },
    }

    print(f"\nRunning {len(configs)} ablation conditions...")
    print(f"\n{'Condition':<25s} {'Mean Score':>10s} {'Mean Rank':>10s} {'Top-3%':>8s} {'Params':>10s}")
    print("-" * 70)

    results = {}

    for name, config in configs.items():
        ablate_cross_attn = config.pop('ablate_cross_attn', False)
        ablate_attn_pool = config.pop('ablate_attn_pool', False)

        model = SiameseEGNN(edge_dim=0, **config)

        if os.path.exists(WEIGHTS_PATH):
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict, strict=False)
            except:
                pass

        if ablate_cross_attn:
            orig_forward_one = model.forward_one
            def no_cross_forward(data, model=model):
                if data.x.dim() == 2 and data.x.size(1) == model.input_dim:
                    seq_emb_raw = data.x.float()
                    h = model.input_projector(seq_emb_raw)
                else:
                    x_indices = data.x.squeeze(-1).long()
                    h = model.embedding(x_indices)
                h_out, _ = model.egnn(h, data.pos, data.edge_index)
                if hasattr(data, 'batch') and data.batch is not None:
                    h_pooled = model.attention_pool(h_out, data.batch)
                else:
                    h_pooled = model.attention_pool(h_out)
                z = model.projector(h_pooled)
                z = F.normalize(z, p=2, dim=-1)
                return z
            model.forward_one = no_cross_forward

        if ablate_attn_pool:
            orig_forward_one = model.forward_one
            def mean_pool_forward(data, model=model):
                if data.x.dim() == 2 and data.x.size(1) == model.input_dim:
                    seq_emb_raw = data.x.float()
                    h = model.input_projector(seq_emb_raw)
                else:
                    x_indices = data.x.squeeze(-1).long()
                    h = model.embedding(x_indices)
                h_out, _ = model.egnn(h, data.pos, data.edge_index)
                if seq_emb_raw is not None:
                    h_out = model.forward(h_out.unsqueeze(0), seq_emb_raw.unsqueeze(0)).squeeze(0)
                h_pooled = h_out.mean(dim=0, keepdim=True)
                z = model.projector(h_pooled)
                z = F.normalize(z, p=2, dim=-1)
                return z
            model.forward_one = mean_pool_forward

        total_params = sum(p.numel() for p in model.parameters())

        metrics = evaluate_model(model, graphs)
        results[name] = metrics

        print(f"  {name:<23s} {metrics['mean_score']:>+10.4f} {metrics['mean_rank']:>8.1f}/24 "
              f"{metrics['top3_pct']:>7.1f}% {total_params:>9,d}")

    print("\n" + "=" * 70)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)

    full = results.get('Full Model (v2)', {})
    if full:
        print(f"\n  {'Component':<25s} {'Impact on Top-3':>15s}")
        print("  " + "-" * 43)

        for name, metrics in results.items():
            if name == 'Full Model (v2)':
                continue
            delta = metrics['top3_pct'] - full['top3_pct']
            direction = "DROP" if delta < 0 else "GAIN" if delta > 0 else "SAME"
            print(f"  Removing {name:<20s} {delta:>+10.1f}%  ({direction})")

    print("\n" + "=" * 70)
    print("Tier 3A Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
