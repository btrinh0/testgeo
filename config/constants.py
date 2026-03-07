"""
GeoMimic-Net: Centralized Constants

Single source of truth for ground-truth mimicry pairs, negative controls,
and common directory paths used across all training and evaluation scripts.
"""

POSITIVE_DIR = 'data/benchmark/positive'
NEGATIVE_DIR = 'data/benchmark/negative'
RAW_DIR = 'data/raw'
BLIND_DIR = 'data/blind_validation'

PDB_DIRS = [RAW_DIR, POSITIVE_DIR, NEGATIVE_DIR]

PRETRAINED_WEIGHTS = 'models/geomimic_net_weights_final.pth'
SUPERVISED_WEIGHTS = 'models/geomimic_net_weights_supervised.pth'

PDB_BASE_URL = 'https://files.rcsb.org/download/'

TRUE_PAIRS = [

    ('1Q59', '1G5M'),
    ('2V5I', '1LB5'),
    ('3CL3', '3H11'),
    ('2GX9', '1KX5'),
    ('2JBY', '1G5M'),
    ('1B4C', '1ITB'),
    ('1FV1', '1CDF'),
    ('1H26', '1CF7'),
    ('1GUX', '1CF7'),
    ('1EFN', '1SHF'),
    ('3D2U', '1HHK'),
    ('2UWI', '1EXT'),
    ('2BZR', '1MAZ'),
    ('2VGA', '1CA9'),
    ('1F5Q', '1B7T'),
    ('2BBR', '1A1W'),

    ('1VLK', '2ILK'),
    ('4I4Q', '3WCY'),
    ('5GQN', '2ILK'),
    ('2FAL', '3AUL'),
    ('1GKP', '1D0G'),
    ('1JFW', '1VPF'),
    ('1R7G', '1BB9'),
    ('1NEP', '2IXH'),
    ('3FKE', '3LLH'),
    ('3L32', '1BF5'),
    ('4GJT', '6WG5'),
    ('7JX6', '1HHK'),
    ('4GIZ', '1TSR'),
]

TRUE_PAIRS_16 = TRUE_PAIRS[:16]

NEGATIVE_PDBS = [
    '1A3N',
    '1TRZ',
    '1MBN',
    '1UBQ',
    '1LYZ',
    '1EMA',
    '4INS',
    '1CLL',
    '7RSA',
    '1HRC',
]

BLIND_NEGATIVES = ['1UBQ', '1LYZ', '1MBN']

BLIND_TEST_PAIRS = [
    ('5JHM', '5U6B', 'Zika Envelope vs Human AXL'),
    ('2XFB', '6NK3', 'Chikungunya E1 vs Human MxRA8'),
    ('6M0J', '1R42', 'SARS-CoV-2 RBD vs ACE2'),
    ('3JWD', '1WIO', 'HIV gp120 vs CD4'),
    ('1A1V', '4IWO', 'HCV NS3 vs TBK1'),
    ('1B3T', '1CKT', 'EBV EBNA1 vs HMGB1'),
]

ALL_HUMAN_PDBS = sorted(set(h for _, h in TRUE_PAIRS))
ALL_VIRAL_PDBS = sorted(set(v for v, _ in TRUE_PAIRS))
