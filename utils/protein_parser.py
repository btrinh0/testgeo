import torch
from torch_geometric.data import Data
# from torch_geometric.nn import knn_graph # Removed dependency
from Bio.PDB import PDBParser
import numpy as np

def parse_pdb_to_pyg(pdb_path, k=10):
    """
    Parses a PDB file, extracts Chain A, and converts it to a PyTorch Geometric Data object.
    
    Args:
        pdb_path (str): Path to the PDB file.
        k (int): Number of nearest neighbors for the k-NN graph.
        
    Returns:
        torch_geometric.data.Data: Graph data object with:
            - x: Node features (Atomic numbers, [N, 1])
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

    atoms = []
    coords = []
    
    # Mapping element symbol to atomic number (simplified subset for proteins)
    element_to_atomic_num = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'SE': 34, 
        'P': 15, 'FE': 26, 'MG': 12, 'CA': 20, 'ZN': 30
        # Add more if needed
    }

    for residue in chain:
        # Optional: Filter for standard amino acids only if strictness is needed
        # if residue.id[0] != ' ': continue 
        for atom in residue:
            atoms.append(element_to_atomic_num.get(atom.element.upper(), 0)) # 0 for unknown
            coords.append(atom.get_coord())
            
    if len(atoms) == 0:
        raise ValueError("No atoms found in the specified chain.")

    x = torch.tensor(atoms, dtype=torch.long).unsqueeze(1) # [N, 1]
    pos = torch.tensor(np.array(coords), dtype=torch.float) # [N, 3]
    
    # Custom k-NN implementation to avoid torch-cluster dependency
    # Pos: [N, 3]
    # Compute pairwise distances
    # Note: For very large proteins, this might be memory intensive [N, N]. 
    # But for standard proteins (few thousand atoms), it's fine on CPU/GPU.
    dist = torch.cdist(pos, pos) # [N, N]
    
    # Get k+1 nearest neighbors (distance 0 is self)
    # largest=False -> smallest distances
    # We select k+1 to include self, then remove self.
    k_adj = min(k + 1, len(atoms))
    _, indices = dist.topk(k_adj, largest=False) # [N, k+1]
    
    # Exclude self-loops (assuming self is always closest, index 0)
    neighbor_indices = indices[:, 1:] # [N, k]
    
    # Create edge_index
    # We want edges (source, target) where source is neighbor, target is central node i
    # This matches PyG convention for aggregating FROM neighbors TO i.
    num_nodes = pos.size(0)
    target = torch.arange(num_nodes, device=pos.device).repeat_interleave(neighbor_indices.size(1))
    source = neighbor_indices.flatten()
    
    edge_index = torch.stack([source, target], dim=0)
    
    data = Data(x=x, pos=pos, edge_index=edge_index)
    
    return data

if __name__ == "__main__":
    # Test with a dummy file if one existed, or just print info
    print("Protein Parser module ready.")
    print("Usage: data = parse_pdb_to_pyg('path/to/file.pdb')")
