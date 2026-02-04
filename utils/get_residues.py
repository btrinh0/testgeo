"""
Script to map atom indices from mimicry site detection to residue numbers
and generate RFdiffusion-compatible contig strings.
"""

from Bio.PDB import PDBParser
from pathlib import Path


def load_atom_indices(filepath: str) -> list[int]:
    """Load atom indices from mimicry_site.txt, skipping comment lines."""
    indices = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            try:
                indices.append(int(line))
            except ValueError:
                continue
    return indices


def build_atom_to_residue_map(pdb_path: str) -> dict[int, tuple[str, int]]:
    """
    Build a mapping from atom serial number to (chain_id, residue_number).
    Uses 1-based atom indexing to match PDB ATOM records.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    atom_to_residue = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    serial = atom.get_serial_number()
                    chain_id = chain.id
                    res_num = residue.id[1]  # Residue sequence number
                    atom_to_residue[serial] = (chain_id, res_num)
    
    return atom_to_residue


def find_contigs(residue_numbers: list[int], chain_id: str) -> list[str]:
    """
    Convert a list of residue numbers into contig strings.
    Groups consecutive residues into ranges like 'A100-A110'.
    """
    if not residue_numbers:
        return []
    
    # Sort and remove duplicates
    sorted_residues = sorted(set(residue_numbers))
    
    contigs = []
    start = sorted_residues[0]
    end = sorted_residues[0]
    
    for res in sorted_residues[1:]:
        if res == end + 1:
            # Consecutive, extend the range
            end = res
        else:
            # Gap found, save current contig and start new one
            contigs.append(f"{chain_id}{start}-{chain_id}{end}")
            start = res
            end = res
    
    # Add the last contig
    contigs.append(f"{chain_id}{start}-{chain_id}{end}")
    
    return contigs


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdb_path = project_root / "data" / "raw" / "6h3x.pdb"
    mimicry_path = project_root / "results" / "mimicry_site.txt"
    
    print(f"Loading PDB: {pdb_path}")
    print(f"Loading atom indices: {mimicry_path}")
    print()
    
    # Load data
    atom_indices = load_atom_indices(str(mimicry_path))
    print(f"Found {len(atom_indices)} atom indices in mimicry site file")
    
    # Build atom-to-residue mapping
    atom_to_residue = build_atom_to_residue_map(str(pdb_path))
    print(f"Mapped {len(atom_to_residue)} atoms from PDB file")
    print()
    
    # Map atom indices to residues
    residues_by_chain: dict[str, list[int]] = {}
    unmapped = []
    
    for atom_idx in atom_indices:
        if atom_idx in atom_to_residue:
            chain_id, res_num = atom_to_residue[atom_idx]
            if chain_id not in residues_by_chain:
                residues_by_chain[chain_id] = []
            residues_by_chain[chain_id].append(res_num)
        else:
            unmapped.append(atom_idx)
    
    if unmapped:
        print(f"Warning: {len(unmapped)} atom indices could not be mapped")
        print(f"  Unmapped indices: {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")
        print()
    
    # Generate contigs for each chain
    print("=" * 60)
    print("RESIDUE MAPPING RESULTS")
    print("=" * 60)
    
    all_contigs = []
    for chain_id, residues in sorted(residues_by_chain.items()):
        unique_residues = sorted(set(residues))
        print(f"\nChain {chain_id}:")
        print(f"  Unique residues ({len(unique_residues)}): {unique_residues}")
        
        contigs = find_contigs(residues, chain_id)
        all_contigs.extend(contigs)
        print(f"  Contigs: {contigs}")
    
    # Print final target contig for RFdiffusion
    print()
    print("=" * 60)
    print("TARGET CONTIG FOR RFdiffusion")
    print("=" * 60)
    
    # Join all contigs with commas for RFdiffusion format
    target_contig = ",".join(all_contigs)
    print(f"\n{target_contig}\n")
    
    # Also provide overall range if it's a single contiguous stretch
    if residues_by_chain:
        for chain_id, residues in sorted(residues_by_chain.items()):
            unique_residues = sorted(set(residues))
            if unique_residues:
                overall = f"{chain_id}{min(unique_residues)}-{max(unique_residues)}"
                print(f"Overall range for chain {chain_id}: {overall}")
    
    print()


if __name__ == "__main__":
    main()
