# GeoMimic-Net

**Geometric Deep Learning for Viral Molecular Mimicry Detection Using Equivariant Graph Neural Networks**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)


> Molecular mimicry — the structural imitation of host proteins by viral proteins — is a key immune evasion strategy. GeoMimic-Net detects mimicry using an equivariant graph neural network (EGNN) that learns geometric and sequence-level representations of protein surface patches, operating where sequence-based methods fail (mean sequence identity of true pairs: **7.2%**).

---

## Key Results

| Metric | Value |
|--------|-------|
| Mean AUC (LOO-CV) | **0.902** |
| Top-1 Accuracy | 43.8% |
| Top-5 Accuracy | 81.2% |
| Mean Rank | 3.19 / 14 |
| Blind Validation Sensitivity | **100%** |
| Blind Validation Specificity | **100%** |
| p-value (permutation test) | < 0.001 |

GeoMimic-Net outperforms both sequence identity (Smith-Waterman) and structural similarity baselines on **all 16 benchmark pairs** (+26.6 pp over structural, +92.8 pp over sequence-based methods).

---





## Repository Structure

```
GeoMimic-Net/
├── models/
│   └── egnn.py                  # Core EGNN architecture
├── scripts/
│   ├── cross_validation.py      # Leave-one-out cross-validation
│   ├── blind_validation.py      # Held-out pair evaluation
│   ├── run_ablation.py          # Architecture ablation study
│   ├── structural_baseline.py   # TM-score baseline comparison
│   ├── sequence_baseline.py     # Smith-Waterman baseline
│   ├── discovery_scan.py        # Novel mimicry pair discovery
│   ├── attention_analysis.py    # Attention weight visualization
│   ├── functional_mapping.py    # Binding site enrichment
│   └── generate_publication_figures.py
├── data/
│   ├── raw/                     # PDB files
│   ├── benchmark/               # 16 benchmark mimicry pairs
│   ├── blind_validation/        # 6 held-out validation pairs
│   └── discovery_scan/          # Novel viral proteins
├── results/                     # Output figures and metrics
├── tests/                       # Unit tests
├── utils/                       # Helper functions
├── train.py                     # Contrastive training loop
├── geomimic_net_weights.pth     # Trained model weights
└── requirements.txt             # Dependencies
```

---

## Installation

```bash
git clone https://github.com/btrinh0/GeoMimic-Net.git
cd GeoMimic-Net
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric
- ESM (Meta AI)
- BioPython

---

## Scientific poster board
https://docs.google.com/presentation/d/1-oaQ_Od8PW2b8UnB427wmLA9-QSiO3kK04HyDX5MK_M/edit?usp=sharing
<img width="2500" height="1875" alt="image" src="https://github.com/user-attachments/assets/756f7fe3-2430-477b-bc01-74e23b63a2ba" />

