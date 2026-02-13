"""
Tier 1A: Download 14 New Validated Mimicry Pairs (expand 16 -> 30)
"""
import os
import urllib.request
import time

POSITIVE_DIR = 'data/benchmark/positive'
RAW_DIR = 'data/raw'
PDB_BASE_URL = 'https://files.rcsb.org/download/'

NEW_PAIRS = [
    ('1VLK', '2ILK', 'EBV vIL-10', 'Human IL-10', 'Hsu et al. 1990'),
    ('4I4Q', '3WCY', 'Vaccinia B18R', 'IFN-alpha R', 'Symons et al. 1995'),
    ('5GQN', '2ILK', 'CMV UL111A', 'Human IL-10', 'Kotenko et al. 2000'),
    ('2FAL', '3AUL', 'KSHV K3 ubiq ligase', 'MARCH E3 ligase', 'Stevenson et al. 2000'),
    ('1GKP', '1D0G', 'Adenovirus E3-RIDa', 'TRAIL-R1', 'Benedict et al. 2001'),
    ('1JFW', '1VPF', 'HIV Tat', 'VEGF', 'Albini et al. 1996'),
    ('1R7G', '1BB9', 'HCV NS5A', 'Amphiphysin SH3', 'Chimnaronk et al.'),
    ('1NEP', '2IXH', 'HSV ICP47', 'TAP transporter', 'Ahn et al. 1996'),
    ('3FKE', '3LLH', 'Ebola VP35', 'dsRNA binding', 'Leung et al. 2009'),
    ('3L32', '1BF5', 'Rabies P protein', 'STAT1', 'Vidy et al. 2005'),
    ('4GJT', '6WG5', 'Measles V protein', 'STAT2', 'Caignard et al. 2007'),
    ('7JX6', '1HHK', 'SARS-CoV-2 ORF8', 'MHC-I HLA-A', 'Zhang et al. 2021'),
    ('4GIZ', '1TSR', 'HPV E6', 'p53', 'Zanier et al. 2013'),
    ('4ORP', '6WG5', 'Nipah V protein', 'STAT2', 'Shaw et al. 2004'),
]

def download_pdb(pdb_id, output_dir):
    filepath = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        print(f"  [EXISTS] {pdb_id}")
        return filepath
    try:
        urllib.request.urlretrieve(f"{PDB_BASE_URL}{pdb_id}.pdb", filepath)
        print(f"  [OK] {pdb_id}: {os.path.getsize(filepath):,} bytes")
        time.sleep(0.3)
        return filepath
    except Exception as e:
        print(f"  [ERROR] {pdb_id}: {e}")
        return None

def main():
    print("=" * 70)
    print("Tier 1A: Downloading 14 New Validated Mimicry Pairs")
    print("=" * 70)
    os.makedirs(POSITIVE_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    
    viral_ids = set(v for v, _, _, _, _ in NEW_PAIRS)
    human_ids = set(h for _, h, _, _, _ in NEW_PAIRS)
    
    print("\nDownloading viral proteins...")
    for pdb_id in sorted(viral_ids):
        download_pdb(pdb_id, RAW_DIR)
    
    print("\nDownloading human target proteins...")
    for pdb_id in sorted(human_ids):
        download_pdb(pdb_id, POSITIVE_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
