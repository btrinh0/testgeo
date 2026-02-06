import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
import numpy as np
import esm

# ========== ESM-2 Model Singleton ==========
# Load once and reuse for efficiency
_esm_model = None
_esm_batch_converter = None
_esm_alphabet = None

def get_esm_model():
    """Load ESM-2 model (singleton pattern for efficiency)."""
    global _esm_model, _esm_batch_converter, _esm_alphabet
    
    if _esm_model is None:
        print("Loading ESM-2 model (esm2_t6_8M_UR50D)...")
        _esm_model, _esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        _esm_batch_converter = _esm_alphabet.get_batch_converter()
        _esm_model.eval()  # Set to evaluation mode
        print("ESM-2 model loaded successfully!")
    
    return _esm_model, _esm_batch_converter, _esm_alphabet


def extract_sequence_from_chain(chain):
    """
    Extract amino acid sequence from a chain.
    
    Returns:
        sequence (str): One-letter amino acid sequence
        residue_list (list): List of residue objects corresponding to the sequence
    """
    sequence = ""
    residue_list = []
    
    for residue in chain:
        # Only process standard amino acids
        if is_aa(residue, standard=True):
            try:
                # Use seq1 to convert 3-letter code to 1-letter code
                one_letter = seq1(residue.get_resname())
                sequence += one_letter
                residue_list.append(residue)
            except Exception:
                # Unknown residue, skip
                continue
    
    return sequence, residue_list


def get_esm_embeddings(sequence):
    """
    Get per-residue ESM-2 embeddings for a sequence.
    
    Args:
        sequence (str): Amino acid sequence (one-letter codes)
        
    Returns:
        embeddings (torch.Tensor): Per-residue embeddings [L, 320]
    """
    model, batch_converter, alphabet = get_esm_model()
    
    # Prepare data for ESM-2
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # Get embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
    
    # Extract the last layer representations
    # Shape: [1, L+2, 320] (includes BOS and EOS tokens)
    token_representations = results["representations"][6]
    
    # Remove BOS and EOS tokens (first and last)
    # Shape: [L, 320]
    embeddings = token_representations[0, 1:-1, :]
    
    return embeddings


def parse_pdb_to_pyg(pdb_path, k=10, use_esm=True):
    """
    Parses a PDB file, extracts Chain A, and converts it to a PyTorch Geometric Data object.
    Uses ESM-2 embeddings for node features.
    
    Args:
        pdb_path (str): Path to the PDB file.
        k (int): Number of nearest neighbors for the k-NN graph.
        use_esm (bool): If True, use ESM-2 embeddings. If False, use atomic numbers.
        
    Returns:
        torch_geometric.data.Data: Graph data object with:
            - x: Node features (ESM embeddings [N, 320] or atomic numbers [N, 1])
            - pos: Node coordinates ([N, 3])
            - edge_index: Graph connectivity ([2, E])
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Select Chain A
    if 'A' in structure[0]:
        chain = structure[0]['A']
    else:
        # Fallback to the first chain if A is not found
        chain = next(iter(structure[0]))
        print(f"Warning: Chain A not found, using chain {chain.id}")

    coords = []
    atom_residue_indices = []  # Maps each atom to its residue index in residue_list
    
    if use_esm:
        # ========== ESM-2 Bilingual Brain Mode ==========
        # Step 1: Extract sequence and get ESM embeddings
        sequence, residue_list = extract_sequence_from_chain(chain)
        
        if len(sequence) == 0:
            raise ValueError("No standard amino acids found in the chain.")
        
        print(f"  Sequence length: {len(sequence)} residues")
        
        # Step 2: Get ESM-2 embeddings [L, 320]
        esm_embeddings = get_esm_embeddings(sequence)
        
        # Step 3: Build residue-to-index mapping
        residue_to_idx = {res.get_id(): idx for idx, res in enumerate(residue_list)}
        
        # Step 4: For each atom, find its residue and assign the ESM embedding
        atom_features = []
        
        for residue in chain:
            res_id = residue.get_id()
            
            if res_id in residue_to_idx:
                # Standard amino acid - use ESM embedding
                res_idx = residue_to_idx[res_id]
                esm_vec = esm_embeddings[res_idx]  # [320]
                
                for atom in residue:
                    atom_features.append(esm_vec)
                    coords.append(atom.get_coord())
            else:
                # Non-standard residue (e.g., HETATM) - use zero vector
                for atom in residue:
                    atom_features.append(torch.zeros(320))
                    coords.append(atom.get_coord())
        
        if len(atom_features) == 0:
            raise ValueError("No atoms found in the specified chain.")
        
        # Stack features: [N, 320]
        x = torch.stack(atom_features, dim=0)
        
    else:
        # ========== Legacy Mode: Atomic Numbers ==========
        atoms = []
        element_to_atomic_num = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'SE': 34, 
            'P': 15, 'FE': 26, 'MG': 12, 'CA': 20, 'ZN': 30
        }
        
        for residue in chain:
            for atom in residue:
                atoms.append(element_to_atomic_num.get(atom.element.upper(), 0))
                coords.append(atom.get_coord())
        
        if len(atoms) == 0:
            raise ValueError("No atoms found in the specified chain.")
        
        x = torch.tensor(atoms, dtype=torch.long).unsqueeze(1)  # [N, 1]
    
    # Convert coordinates
    pos = torch.tensor(np.array(coords), dtype=torch.float)  # [N, 3]
    
    # ========== Build k-NN Graph ==========
    dist = torch.cdist(pos, pos)  # [N, N]
    
    k_adj = min(k + 1, pos.size(0))
    _, indices = dist.topk(k_adj, largest=False)  # [N, k+1]
    
    # Exclude self-loops
    neighbor_indices = indices[:, 1:]  # [N, k]
    
    # Create edge_index
    num_nodes = pos.size(0)
    target = torch.arange(num_nodes, device=pos.device).repeat_interleave(neighbor_indices.size(1))
    source = neighbor_indices.flatten()
    
    edge_index = torch.stack([source, target], dim=0)
    
    data = Data(x=x, pos=pos, edge_index=edge_index)
    
    return data


# Legacy function for backward compatibility
def parse_pdb_to_pyg_legacy(pdb_path, k=10):
    """Legacy parser using atomic numbers instead of ESM embeddings."""
    return parse_pdb_to_pyg(pdb_path, k=k, use_esm=False)


if __name__ == "__main__":
    print("Protein Parser module ready (ESM-2 Bilingual Brain).")
    print("Usage: data = parse_pdb_to_pyg('path/to/file.pdb')")
    print("  - x shape: [N, 320] (ESM-2 per-residue embeddings)")
    print("  - pos shape: [N, 3]")
    print("  - edge_index shape: [2, E]")
