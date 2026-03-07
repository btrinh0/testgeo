"""
Tests for config/constants.py — validates the centralized constants
are self-consistent (no typos, no duplicate IDs, correct subsets).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.constants import (
    TRUE_PAIRS, TRUE_PAIRS_16, NEGATIVE_PDBS,
    BLIND_TEST_PAIRS, BLIND_NEGATIVES,
    ALL_HUMAN_PDBS, ALL_VIRAL_PDBS,
    PDB_DIRS, SUPERVISED_WEIGHTS, PRETRAINED_WEIGHTS,
)

def test_true_pairs_count():
    """29 total validated mimicry pairs."""
    assert len(TRUE_PAIRS) == 29, f"Expected 29 pairs, got {len(TRUE_PAIRS)}"

def test_true_pairs_16_is_subset():
    """The 16-pair subset must be the first 16 of the full list."""
    assert TRUE_PAIRS_16 == TRUE_PAIRS[:16]

def test_true_pairs_format():
    """Each pair is a 2-tuple of uppercase PDB IDs (4 chars)."""
    for pair in TRUE_PAIRS:
        assert len(pair) == 2, f"Expected 2-tuple, got {pair}"
        viral, human = pair
        assert len(viral) == 4, f"Viral ID wrong length: {viral}"
        assert len(human) == 4, f"Human ID wrong length: {human}"
        assert viral == viral.upper(), f"Viral ID not uppercase: {viral}"
        assert human == human.upper(), f"Human ID not uppercase: {human}"

def test_no_duplicate_pairs():
    """No pair should appear twice in TRUE_PAIRS."""
    assert len(set(TRUE_PAIRS)) == len(TRUE_PAIRS), "Duplicate pairs found"

def test_negative_pdbs_count():
    """10 negative control proteins."""
    assert len(NEGATIVE_PDBS) == 10

def test_no_overlap_positive_negative():
    """No PDB ID should be in both TRUE_PAIRS and NEGATIVE_PDBS."""
    all_positive_ids = set()
    for v, h in TRUE_PAIRS:
        all_positive_ids.add(v)
        all_positive_ids.add(h)
    overlap = all_positive_ids & set(NEGATIVE_PDBS)
    assert len(overlap) == 0, f"Overlap: {overlap}"

def test_blind_pairs_not_in_training():
    """Blind test pairs should NOT overlap with training TRUE_PAIRS."""
    training_set = set(TRUE_PAIRS)
    for viral, human, _name in BLIND_TEST_PAIRS:
        assert (viral, human) not in training_set,            f"Blind pair ({viral}, {human}) is in training set!"

def test_blind_negatives_are_subset():
    """Blind negatives should be a subset of the full negative list."""
    for neg in BLIND_NEGATIVES:
        assert neg in NEGATIVE_PDBS, f"{neg} not in NEGATIVE_PDBS"

def test_derived_sets_correct():
    """ALL_HUMAN_PDBS and ALL_VIRAL_PDBS should match what's in TRUE_PAIRS."""
    expected_human = sorted(set(h for _, h in TRUE_PAIRS))
    expected_viral = sorted(set(v for v, _ in TRUE_PAIRS))
    assert ALL_HUMAN_PDBS == expected_human
    assert ALL_VIRAL_PDBS == expected_viral

def test_paths_are_strings():
    """All path constants should be non-empty strings."""
    for path in PDB_DIRS:
        assert isinstance(path, str) and len(path) > 0
    assert isinstance(SUPERVISED_WEIGHTS, str) and len(SUPERVISED_WEIGHTS) > 0
    assert isinstance(PRETRAINED_WEIGHTS, str) and len(PRETRAINED_WEIGHTS) > 0

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
