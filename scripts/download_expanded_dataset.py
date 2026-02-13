"""
Phase 20: Download Expanded Dataset
Downloads PDB files for 12 validated viral-human mimicry pairs + 6 negative controls.

Corrected pairs based on scientifically validated structural mimicry.
"""

import os
import urllib.request
import sys

# ============================================================================
# Corrected Mimicry Pairs (12 total)
# ============================================================================

TRUE_PAIRS = [
    # (Viral ID, Viral Name, Human ID, Human Name, Mimicry Type)
    
    # --- Original 4 pairs (already in benchmark/positive) ---
    ('1Q59', 'EBV BHRF1',        '1G5M', 'Bcl-2',           'Anti-apoptosis'),
    ('2V5I', 'Vaccinia A52',     '1LB5', 'TRAF6',           'Signaling hijack'),
    ('3CL3', 'KSHV vFLIP',      '3H11', 'FLIP',            'Death domain'),
    ('2GX9', 'Flu NS1',          '1KX5', 'Histone H3',     'Epigenetic'),
    
    # --- New 8 pairs ---
    ('2JBY', 'Myxoma M11L',     '1G5M', 'Bcl-2',           'Anti-apoptosis (Structural Homolog)'),
    ('1B4C', 'Vaccinia B15',    '1ITB', 'IL-1R',            'Decoy Receptor'),
    ('1FV1', 'EBV LMP1',        '1CDF', 'CD40',             'Receptor Mimicry'),
    ('1H26', 'Adenovirus E1A',  '1CF7', 'E2F',              'Cell Cycle Hijack'),
    ('1GUX', 'HPV E7',          '1CF7', 'E2F',              'Cell Cycle Hijack'),
    ('1EFN', 'HIV Nef',         '1SHF', 'Fyn SH3',          'Kinase Mimicry'),
    ('3D2U', 'CMV UL18',        '1HHK', 'MHC-I HLA-A',     'Immune Evasion'),
    ('2UWI', 'Cowpox CrmE',     '1EXT', 'TNFR1',            'Decoy Receptor'),
    ('2BZR', 'KSHV vBcl-2',    '1MAZ', 'Bcl-xL',           'Anti-apoptosis Homolog'),
    ('2VGA', 'Variola CrmB',   '1CA9', 'TNFR2',             'Decoy Receptor'),
    ('1F5Q', 'KSHV vCyclin',   '1B7T', 'Cyclin D2',         'Cell Cycle Mimicry'),
    ('2BBR', 'Molluscum MC159', '1A1W', 'FADD DED',         'Death Domain Mimicry'),
]

# Negative controls (unrelated proteins)
NEGATIVES = [
    ('1A3N', 'Hemoglobin'),
    ('1TRZ', 'Trypsin Inhibitor'),
    ('1MBN', 'Myoglobin'),
    ('1UBQ', 'Ubiquitin'),
    ('1LYZ', 'Lysozyme'),
    ('1EMA', 'GFP'),
    ('4INS', 'Insulin'),
    ('1CLL', 'Calmodulin'),
    ('7RSA', 'RNase A'),
    ('1HRC', 'Cytochrome C'),
]

OUTPUT_DIR = 'data/raw'
PDB_BASE_URL = 'https://files.rcsb.org/download/'


def download_pdb(pdb_id, output_dir):
    """Download a PDB file from RCSB."""
    filename = f"{pdb_id}.pdb"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        if size > 100:  # Not empty
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
    
    # Collect all unique PDB IDs
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
    
    # Remove duplicates and download
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
