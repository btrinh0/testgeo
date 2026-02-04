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
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    return processed_files

def verify_with_model(processed_files):
    """
    Loads processed files and runs them through the EGNN model to verify compatibility.
    """
    if not processed_files:
        print("No files to verify.")
        return

    print("\nVerifying with EGNN model...")
    
    # Initialize model
    # Assuming atomic numbers are used as features (embedding or one-hot needed usually)
    # The parser returns x as atomic numbers [N, 1].
    # The EGNN expects node_dim features. 
    # If we pass raw atomic numbers, we might need an embedding layer first.
    # For now, let's just project the scalar atomic number to hidden_dim if we want to be simple, 
    # or better: One-hot encode max atomic number.
    # But to keep it simple and consistent with previous EGNN usage:
    # We will just pass float(x) if node_dim=1, OR we will assume the model handles it.
    # Wait, the EGNN I wrote expects `h` of dim `node_dim`.
    # Let's adjust the verification to include a simple embedding or projection.
    
    node_dim = 16
    edge_dim = 0 # Parser doesn't return edge attributes by default
    hidden_dim = 32
    
    # Simple embedding layer to convert atomic number to node_dim
    embedding = torch.nn.Embedding(100, node_dim) 
    model = EGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    
    for file_path in processed_files:
        print(f"Testing {os.path.basename(file_path)}...")
        try:
            # We trust the data we just generated
            data = torch.load(file_path, weights_only=False)
            
            # Prepare inputs
            # Ensure x is LongTensor for embedding
            x_indices = data.x.squeeze(-1).long() # [N]
            h = embedding(x_indices) # [N, node_dim]
            x_coord = data.pos # [N, 3]
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
