"""
Oropouche Virus Validation - GeoMimic-Net Cross-Project Integration

Runs GeoMimic-Net's mimicry detection pipeline against the Oropouche virus
Gc glycoprotein (PDB: 6H3X) to validate the connection between the
GeoMimic-Net project and the Oropouche miniprotein binder project.

Three Tiers:
  Tier 1 - Discovery Scan:    Gc vs expanded human proteome panel
  Tier 2 - Attention Mapping:  Per-residue attention on top candidates
  Tier 3 - Cross-Project Link: Overlap with miniprotein binder target interface
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import urllib.request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.egnn import SiameseEGNN
from utils.protein_parser import parse_pdb_to_pyg

WEIGHTS_PATH = 'models/geomimic_net_weights_supervised.pth'
RESULTS_FILE = 'results/oropouche_validation_results.txt'
SCAN_DIR = 'data/discovery_scan'
PDB_DIRS = ['data/raw', 'data/benchmark/positive', 'data/benchmark/negative',
            'data/blind_validation', SCAN_DIR]
PDB_BASE_URL = 'https://files.rcsb.org/download/'

OROPOUCHE_PROTEINS = [
    ('6H3X', 'OROV Gc Head Domain'),
]

HUMAN_PANEL = [

    ('1EXT', 'TNFR1 (TNF Receptor)'),
    ('1ITB', 'IL-1R (Interleukin-1 Receptor)'),
    ('1CDF', 'CD40'),
    ('2ILK', 'IL-10'),
    ('1HHK', 'MHC-I HLA-A'),
    ('1LB5', 'TRAF6'),
    ('1A1W', 'FADD DED'),

    ('1G5M', 'Bcl-2'),
    ('1MAZ', 'Bcl-xL'),

    ('1CF7', 'E2F'),
    ('1TSR', 'p53'),
    ('1B7T', 'Cyclin D2'),

    ('1SHF', 'Fyn SH3 Domain'),
    ('1BF5', 'STAT1'),

    ('1A3N', 'Hemoglobin'),
    ('1UBQ', 'Ubiquitin'),

    ('1R42', 'ACE2'),
    ('1WIO', 'CD4'),
    ('5U6B', 'AXL'),

    ('1LYZ', 'Lysozyme (neg ctrl)'),
    ('1MBN', 'Myoglobin (neg ctrl)'),
]

GC_FUNCTIONAL_REGIONS = [
    (482, 540, 'N-terminal head region'),
    (541, 600, 'Central beta-sheet core'),
    (601, 650, 'Receptor-binding surface'),
    (651, 702, 'Head-stalk interface'),
]

def download_pdb(pdb_id, output_dir):
    """Download PDB file if not already present."""
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
    """Find PDB file in any data directory."""
    for d in PDB_DIRS:
        path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(path):
            return path
    return None

def load_model():
    """Load the trained SiameseEGNN model."""
    model = SiameseEGNN(
        node_dim=64, edge_dim=0, hidden_dim=128,
        embed_dim=256, num_layers=4, geom_dim=64,
        num_rbf=16, dropout=0.1
    )
    if os.path.exists(WEIGHTS_PATH):
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"  [OK] Loaded weights: {WEIGHTS_PATH}")
    else:
        print(f"  [ERROR] Weights not found: {WEIGHTS_PATH}")
        return None
    model.eval()
    return model

def extract_residue_numbers(pdb_path):
    """Extract CA residue numbers from PDB."""
    residues = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                residues.append(int(line[22:26].strip()))
    return residues

def extract_attention_weights(model, data):
    """
    Extract per-residue attention weights from the AttentionPool layer.
    Returns attention weights as numpy array [N_atoms].
    """
    with torch.no_grad():

        if data.x.dim() == 2 and data.x.size(1) == model.input_dim:
            h = model.input_projector(data.x.float())
        else:
            return None

        h_out, _ = model.egnn(h, data.pos, data.edge_index)

        attn_scores = model.attention_pool.attention(h_out)
        attn_weights = torch.softmax(attn_scores, dim=0).squeeze(-1)

        return attn_weights.numpy()

def tier1_discovery_scan(model, graphs, output_lines):
    """Scan Oropouche Gc against human proteome panel."""
    print("\n" + "=" * 70)
    print("TIER 1: OROPOUCHE Gc DISCOVERY SCAN")
    print("=" * 70)
    output_lines.append("=" * 70)
    output_lines.append("TIER 1: OROPOUCHE Gc DISCOVERY SCAN")
    output_lines.append("=" * 70)

    all_discoveries = []

    for viral_id, viral_name in OROPOUCHE_PROTEINS:
        if viral_id not in graphs:
            print(f"  [SKIP] {viral_name}: graph not loaded")
            continue

        print(f"\n  Scanning {viral_name} ({viral_id}) against {len(HUMAN_PANEL)} human proteins:")
        output_lines.append(f"\n  Scanning {viral_name} ({viral_id}):")

        pair_scores = []
        for human_id, human_name in HUMAN_PANEL:
            if human_id not in graphs:
                continue
            with torch.no_grad():
                emb_v = model.forward_one(graphs[viral_id])
                emb_h = model.forward_one(graphs[human_id])
                sim = F.cosine_similarity(emb_v, emb_h).item()
            pair_scores.append((human_id, human_name, sim))

        pair_scores.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  {'Rank':<5} {'Human Target':<30} {'PDB':<6} {'Similarity':>10} {'Status'}")
        print("  " + "-" * 65)
        output_lines.append(f"\n  {'Rank':<5} {'Human Target':<30} {'PDB':<6} {'Similarity':>10} {'Status'}")
        output_lines.append("  " + "-" * 65)

        for rank, (h_id, h_name, score) in enumerate(pair_scores, 1):
            if score > 0.85:
                status = "*** STRONG MIMICRY"
            elif score > 0.7:
                status = "** CANDIDATE"
            elif score > 0.5:
                status = "* WEAK"
            elif score < 0.0:
                status = "REJECTED"
            else:
                status = ""

            line = f"  #{rank:<4} {h_name:<30} {h_id:<6} {score:>+10.4f}   {status}"
            print(line)
            output_lines.append(line)

            if score > 0.7:
                all_discoveries.append({
                    'viral_id': viral_id, 'viral_name': viral_name,
                    'human_id': h_id, 'human_name': h_name,
                    'score': score
                })

    print(f"\n  {'-' * 50}")
    print(f"  Discovery Summary:")
    output_lines.append(f"\n  Discovery Summary:")

    candidates = [d for d in all_discoveries if d['score'] > 0.85]
    weak = [d for d in all_discoveries if 0.7 < d['score'] <= 0.85]

    line = f"  Strong mimicry candidates (>0.85): {len(candidates)}"
    print(line)
    output_lines.append(line)
    for d in candidates:
        line = f"    -> {d['human_name']} ({d['human_id']}): {d['score']:+.4f}"
        print(line)
        output_lines.append(line)

    line = f"  Weak candidates (0.70-0.85):       {len(weak)}"
    print(line)
    output_lines.append(line)

    return all_discoveries

def tier2_attention_mapping(model, graphs, pdb_paths, discoveries, output_lines):
    """Map attention weights on Oropouche Gc to identify mimicry-driving residues."""
    print("\n" + "=" * 70)
    print("TIER 2: ATTENTION-BASED RESIDUE MAPPING ON OROPOUCHE Gc")
    print("=" * 70)
    output_lines.append("\n" + "=" * 70)
    output_lines.append("TIER 2: ATTENTION-BASED RESIDUE MAPPING ON OROPOUCHE Gc")
    output_lines.append("=" * 70)

    for viral_id, viral_name in OROPOUCHE_PROTEINS:
        if viral_id not in graphs:
            continue

        data = graphs[viral_id]
        attention = extract_attention_weights(model, data)
        if attention is None:
            print(f"  [ERROR] Could not extract attention for {viral_id}")
            continue

        pdb_path = pdb_paths.get(viral_id)
        if not pdb_path:
            continue
        residue_numbers = extract_residue_numbers(pdb_path)

        print(f"\n  {viral_name} ({viral_id}): {len(attention)} atoms, {len(residue_numbers)} CA atoms")
        output_lines.append(f"\n  {viral_name} ({viral_id}): {len(attention)} atoms, {len(residue_numbers)} CA atoms")

        print(f"\n  Functional Region Enrichment Analysis:")
        print(f"  {'Region':<35} {'Enrichment':>12} {'Mean Attn':>10} {'Status'}")
        print("  " + "-" * 70)
        output_lines.append(f"\n  Functional Region Enrichment Analysis:")
        output_lines.append(f"  {'Region':<35} {'Enrichment':>12} {'Mean Attn':>10} {'Status'}")
        output_lines.append("  " + "-" * 70)

        global_mean_attn = np.mean(attention)
        enrichments = []

        for start, end, region_name in GC_FUNCTIONAL_REGIONS:

            region_indices = []
            for i, resnum in enumerate(residue_numbers):
                if start <= resnum <= end:

                    if i < len(attention):
                        region_indices.append(i)

            if not region_indices:
                continue

            region_attn = np.mean([attention[i] for i in region_indices])
            non_region_attn = np.mean([attention[i] for i in range(min(len(attention), len(residue_numbers)))
                                       if i not in region_indices])
            enrichment = region_attn / (non_region_attn + 1e-8)
            enrichments.append((region_name, enrichment, region_attn))

            status = "ENRICHED" if enrichment > 1.0 else "not enriched"
            line = f"  {region_name:<35} {enrichment:>10.2f}x  {region_attn:>10.6f}  {status}"
            print(line)
            output_lines.append(line)

        print(f"\n  Top-20 Attention Hotspot Residues:")
        output_lines.append(f"\n  Top-20 Attention Hotspot Residues:")

        ca_attention = attention[:min(len(attention), len(residue_numbers))]
        top_indices = np.argsort(ca_attention)[::-1][:20]

        hotspot_residues = []
        for idx in top_indices:
            if idx < len(residue_numbers):
                resnum = residue_numbers[idx]
                attn_val = ca_attention[idx]
                hotspot_residues.append(resnum)

                region = "-"
                for start, end, name in GC_FUNCTIONAL_REGIONS:
                    if start <= resnum <= end:
                        region = name
                        break

                line = f"    Residue {resnum:>4d}  attn={attn_val:.6f}  [{region}]"
                print(line)
                output_lines.append(line)

        return hotspot_residues

    return []

def tier3_cross_project(hotspot_residues, output_lines):
    """
    Compare GeoMimic-Net's attention hotspots with the miniprotein binder
    project's target interface on Oropouche Gc.
    """
    print("\n" + "=" * 70)
    print("TIER 3: CROSS-PROJECT CONNECTION - MINIPROTEIN BINDER INTERFACE")
    print("=" * 70)
    output_lines.append("\n" + "=" * 70)
    output_lines.append("TIER 3: CROSS-PROJECT CONNECTION - MINIPROTEIN BINDER INTERFACE")
    output_lines.append("=" * 70)

    binder_target_residues = list(range(601, 703))

    print(f"\n  Miniprotein Binder Target Interface: residues {binder_target_residues[0]}-{binder_target_residues[-1]}")
    print(f"  (Receptor-binding surface + Head-stalk interface)")
    print(f"\n  GeoMimic-Net Attention Hotspots: {len(hotspot_residues)} residues")
    output_lines.append(f"\n  Miniprotein Binder Target: residues {binder_target_residues[0]}-{binder_target_residues[-1]}")
    output_lines.append(f"  GeoMimic-Net Hotspots: {len(hotspot_residues)} residues")

    if not hotspot_residues:
        print("  [WARN] No hotspot residues available for comparison.")
        output_lines.append("  [WARN] No hotspot residues available.")
        return

    hotspot_set = set(hotspot_residues)
    binder_set = set(binder_target_residues)
    overlap = hotspot_set & binder_set
    overlap_pct = len(overlap) / len(hotspot_set) * 100 if hotspot_set else 0

    print(f"\n  {'-' * 50}")
    print(f"  OVERLAP ANALYSIS:")
    print(f"    Hotspot residues in binder target:  {len(overlap)}/{len(hotspot_set)}")
    print(f"    Overlap percentage:                 {overlap_pct:.1f}%")
    output_lines.append(f"\n  OVERLAP ANALYSIS:")
    output_lines.append(f"    Hotspot residues in binder target: {len(overlap)}/{len(hotspot_set)}")
    output_lines.append(f"    Overlap percentage: {overlap_pct:.1f}%")

    if overlap:
        overlap_sorted = sorted(overlap)
        line = f"    Overlapping residues: {overlap_sorted}"
        print(line)
        output_lines.append(line)

    print(f"\n  {'-' * 50}")
    print(f"  INTERPRETATION:")
    output_lines.append(f"\n  INTERPRETATION:")

    if overlap_pct > 50:
        msg = ("  [STRONG VALIDATION] GeoMimic-Net's attention hotspots show >50%\n"
               "  overlap with the miniprotein binder target interface.\n"
               "  This indicates GeoMimic-Net independently identified the same\n"
               "  biologically critical regions that the binder was designed to block.\n"
               "  The mimicry detection -> therapeutic targeting pipeline is validated.")
    elif overlap_pct > 25:
        msg = ("  [MODERATE VALIDATION] Partial overlap between GeoMimic-Net's\n"
               "  predicted mimicry hotspots and the binder target interface.\n"
               "  The model captures some of the biologically relevant features\n"
               "  recognized by the structural design pipeline.")
    elif overlap_pct > 0:
        msg = ("  [WEAK VALIDATION] Limited overlap detected. The model's\n"
               "  attention may focus on different structural features than\n"
               "  the binder design, suggesting complementary perspectives\n"
               "  on the Gc protein structure.")
    else:
        msg = ("  [NO OVERLAP] GeoMimic-Net's attention hotspots do not overlap\n"
               "  with the binder target interface. The model may be detecting\n"
               "  mimicry in different structural regions of Gc.")

    print(msg)
    for line in msg.strip().split('\n'):
        output_lines.append(line)

    print(f"\n  {'-' * 50}")
    print(f"  CROSS-PROJECT NARRATIVE:")
    narrative = (
        "  GeoMimic-Net was used to scan the Oropouche Gc glycoprotein\n"
        "  against the human proteome, identifying which host proteins\n"
        "  Gc structurally mimics. The model's per-residue attention map\n"
        "  reveals which regions of Gc drive the mimicry prediction.\n"
        "  These attention hotspots were then compared against the\n"
        "  binding interface targeted by the miniprotein binder designed\n"
        "  using RFdiffusion in the companion Oropouche project.\n"
        f"  Result: {overlap_pct:.0f}% overlap between predicted mimicry\n"
        "  sites and the therapeutic target interface."
    )
    print(narrative)
    for line in narrative.strip().split('\n'):
        output_lines.append(line)

def main():
    print("=" * 70)
    print("OROPOUCHE VIRUS VALIDATION")
    print("GeoMimic-Net Cross-Project Integration")
    print("=" * 70)

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("OROPOUCHE VIRUS VALIDATION - GeoMimic-Net Cross-Project Integration")
    output_lines.append("=" * 70)

    os.makedirs(SCAN_DIR, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("\nLoading GeoMimic-Net model...")
    model = load_model()
    if model is None:
        return

    print("\nLocating PDB structures...")
    all_proteins = OROPOUCHE_PROTEINS + HUMAN_PANEL
    pdb_paths = {}
    for pdb_id, name in all_proteins:
        path = find_pdb(pdb_id)
        if not path:
            path = download_pdb(pdb_id, SCAN_DIR)
        if path:
            pdb_paths[pdb_id] = path

    print(f"  Available: {len(pdb_paths)}/{len(all_proteins)} structures")

    print("\nParsing protein structures (with ESM-2 embeddings)...")
    graphs = {}
    for pdb_id, name in all_proteins:
        if pdb_id not in pdb_paths:
            continue
        try:
            graphs[pdb_id] = parse_pdb_to_pyg(pdb_paths[pdb_id], use_esm=True)
        except Exception as e:
            print(f"  [WARN] {pdb_id} ({name}): {e}")

    print(f"  Loaded: {len(graphs)} graphs")

    discoveries = tier1_discovery_scan(model, graphs, output_lines)

    hotspot_residues = tier2_attention_mapping(model, graphs, pdb_paths,
                                               discoveries, output_lines)

    tier3_cross_project(hotspot_residues, output_lines)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    output_lines.append("\n" + "=" * 70)
    output_lines.append("VALIDATION COMPLETE")
    output_lines.append("=" * 70)

    with open(RESULTS_FILE, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\n  Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
