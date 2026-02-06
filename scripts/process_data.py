import os
import torch
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.protein_parser import parse_pdb_to_pyg
from models.egnn import EGNN

def process_data(raw_dir, processed_dir):
    """
    Processes PDB files from raw_dir and saves them to processed_dir.
    Now uses ESM-2 embeddings (320-dim) for node features.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    pdb_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.pdb')]
    
    if not pdb_files:
        print(f"No PDB files found in {raw_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files: {pdb_files}")
    
    processed_files = []
    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(raw_dir, pdb_file)
        try:
            print(f"Processing {pdb_file}...")
            data = parse_pdb_to_pyg(pdb_path)
            
            # Save processed data
            output_name = pdb_file.replace('.pdb', '.pt')
            output_path = os.path.join(processed_dir, output_name)
            torch.save(data, output_path)
            processed_files.append(output_path)
            print(f"Saved to {output_path} (x shape: {data.x.shape})")
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    return processed_files

def verify_with_model(processed_files):
    """
    Loads processed files and runs them through the EGNN model to verify compatibility.
    Updated for ESM-2 320-dimensional features.
    """
    if not processed_files:
        print("No files to verify.")
        return

    print("\nVerifying with EGNN model...")
    
    # ========== ESM-2 Bilingual Brain Configuration ==========
    # Input features are now 320-dimensional ESM-2 embeddings
    # We project them down to 32 dimensions immediately to prevent
    # the "Curse of Dimensionality" and keep training fast.
    
    esm_dim = 320      # ESM-2 embedding dimension
    node_dim = 32      # Projected dimension for EGNN
    edge_dim = 0       # Parser doesn't return edge attributes by default
    hidden_dim = 64    # Hidden dimension for EGNN layers
    
    # Input projection: 320 -> 32 (shrink ESM embeddings immediately)
    input_projector = torch.nn.Linear(esm_dim, node_dim)
    model = EGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    
    for file_path in processed_files:
        print(f"Testing {os.path.basename(file_path)}...")
        try:
            # Load the processed data
            data = torch.load(file_path, weights_only=False)
            
            # Check input dimension
            print(f"  Input x shape: {data.x.shape}")
            
            # Prepare inputs
            if data.x.dim() == 2 and data.x.size(1) == esm_dim:
                # ESM-2 embeddings: [N, 320] -> project to [N, 32]
                h = input_projector(data.x.float())
            elif data.x.dim() == 2 and data.x.size(1) == 1:
                # Legacy atomic numbers: use embedding
                print("  Warning: Using legacy atomic number features")
                embedding = torch.nn.Embedding(100, node_dim)
                x_indices = data.x.squeeze(-1).long()
                h = embedding(x_indices)
            else:
                raise ValueError(f"Unexpected input shape: {data.x.shape}")
            
            x_coord = data.pos  # [N, 3]
            edge_index = data.edge_index
            
            # Forward pass
            h_out, x_out = model(h, x_coord, edge_index)
            print(f"  Success! Output shapes: h={h_out.shape}, x={x_out.shape}")
        except Exception as e:
            print(f"  Failed processing {file_path}: {e}")

if __name__ == "__main__":
    raw_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    
    processed_files = process_data(raw_dir, processed_dir)
    verify_with_model(processed_files)
