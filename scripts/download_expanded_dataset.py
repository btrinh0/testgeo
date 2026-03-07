"""
Phase 20: Download Expanded Dataset
Downloads PDB files for 12 validated viral-human mimicry pairs + 6 negative controls.

Corrected pairs based on scientifically validated structural mimicry.
"""

import os
import urllib.request
import sys

from config.constants import (
    TRUE_PAIRS_16 as _TRUE_PAIRS_2, NEGATIVE_PDBS, PDB_BASE_URL,
)

OUTPUT_DIR = 'data/raw'

_PAIR_METADATA = [
    ('EBV BHRF1',        'Bcl-2',         'Anti-apoptosis'),
    ('Vaccinia A52',     'TRAF6',          'Signaling hijack'),
    ('KSHV vFLIP',       'FLIP',           'Death domain'),
    ('Flu NS1',          'Histone H3',     'Epigenetic'),
    ('Myxoma M11L',      'Bcl-2',          'Anti-apoptosis (Structural Homolog)'),
    ('Vaccinia B15',     'IL-1R',          'Decoy Receptor'),
    ('EBV LMP1',         'CD40',           'Receptor Mimicry'),
    ('Adenovirus E1A',   'E2F',            'Cell Cycle Hijack'),
    ('HPV E7',           'E2F',            'Cell Cycle Hijack'),
    ('HIV Nef',          'Fyn SH3',        'Kinase Mimicry'),
    ('CMV UL18',         'MHC-I HLA-A',    'Immune Evasion'),
    ('Cowpox CrmE',      'TNFR1',          'Decoy Receptor'),
    ('KSHV vBcl-2',      'Bcl-xL',         'Anti-apoptosis Homolog'),
    ('Variola CrmB',     'TNFR2',          'Decoy Receptor'),
    ('KSHV vCyclin',     'Cyclin D2',      'Cell Cycle Mimicry'),
    ('Molluscum MC159',  'FADD DED',       'Death Domain Mimicry'),
]
TRUE_PAIRS = [
    (v, vn, h, hn, mt)
    for (v, h), (vn, hn, mt) in zip(_TRUE_PAIRS_2, _PAIR_METADATA)
]

NEGATIVES = [(n, '') for n in NEGATIVE_PDBS]

def download_pdb(pdb_id, output_dir):
    """Download a PDB file from RCSB."""
    filename = f"{pdb_id}.pdb"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        if size > 100:
            print(f"  [SKIP] {pdb_id} already exists ({size:,} bytes)")
            return True

    url = f"{PDB_BASE_URL}{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"  [OK] {pdb_id}: {size:,} bytes")
        return True
    except Exception as e:
        print(f"  [ERROR] {pdb_id}: {e}")
        return False

def main():
    print("=" * 60)
    print("Phase 20: Download Expanded Dataset")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pdbs = set()

    print(f"\n--- True Mimicry Pairs ({len(TRUE_PAIRS)}) ---")
    for viral_id, viral_name, human_id, human_name, mimicry_type in TRUE_PAIRS:
        print(f"  {viral_id} ({viral_name}) -> {human_id} ({human_name}): {mimicry_type}")
        all_pdbs.add(viral_id)
        all_pdbs.add(human_id)

    print(f"\n--- Negative Controls ({len(NEGATIVES)}) ---")
    for neg_id, neg_name in NEGATIVES:
        print(f"  {neg_id} ({neg_name})")
        all_pdbs.add(neg_id)

    all_pdbs = sorted(all_pdbs)
    print(f"\n--- Downloading {len(all_pdbs)} unique PDB files ---")

    success = 0
    failed = 0

    for pdb_id in all_pdbs:
        result = download_pdb(pdb_id, OUTPUT_DIR)
        if result:
            success += 1
        else:
            failed += 1

    print(f"\n--- Download Summary ---")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Total:   {len(all_pdbs)}")

    if failed > 0:
        print(f"\n  WARNING: {failed} downloads failed. Check PDB IDs.")

    print("\n" + "=" * 60)
    print("Phase 20 Download Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
