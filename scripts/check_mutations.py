"""
Phase 19: Mutation Analysis
Visualize target binding region relative to known immune-escape mutations.

Target Region: Residues 515-534 (from Phase 16 analysis)
2024 Mutations: [507, 552, 738, 795] - Known immune-escape sites

Goal: Show that the green target box does NOT overlap with red mutation X's.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# Configuration
# ============================================================================

# Target binding region (from Phase 16 output)
TARGET_START = 515
TARGET_END = 534

# 2024 Known Immune-Escape Mutations
MUTATIONS_2024 = [507, 552, 738, 795]

# Gene length for visualization
GENE_LENGTH = 1000

OUTPUT_PATH = 'results/mutation_map.png'


# ============================================================================
# Analysis Functions
# ============================================================================

def calculate_distance_to_nearest_mutation(target_start, target_end, mutations):
    """
    Calculate the minimum sequence distance from target region to any mutation.
    
    Returns:
        (distance, nearest_mutation, relation): 
        - distance: number of residues to nearest mutation
        - nearest_mutation: the closest mutation position
        - relation: 'upstream', 'downstream', or 'overlapping'
    """
    min_distance = float('inf')
    nearest = None
    relation = None
    
    for mut in mutations:
        if target_start <= mut <= target_end:
            # Mutation is inside target region
            return 0, mut, 'overlapping'
        elif mut < target_start:
            # Mutation is upstream of target
            dist = target_start - mut
            if dist < min_distance:
                min_distance = dist
                nearest = mut
                relation = 'upstream'
        else:
            # Mutation is downstream of target
            dist = mut - target_end
            if dist < min_distance:
                min_distance = dist
                nearest = mut
                relation = 'downstream'
    
    return min_distance, nearest, relation


def create_gene_map(target_start, target_end, mutations, gene_length, output_path):
    """
    Create a visual gene map showing target region and mutations.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Draw gene backbone (gray line)
    ax.hlines(y=0.5, xmin=1, xmax=gene_length, colors='gray', linewidth=8, alpha=0.3)
    ax.text(gene_length / 2, 0.15, f'Gene (1-{gene_length} residues)', 
            ha='center', fontsize=10, color='gray')
    
    # Draw target region (green box)
    target_width = target_end - target_start
    target_rect = mpatches.FancyBboxPatch(
        (target_start, 0.35), target_width, 0.3,
        boxstyle="round,pad=0.02", 
        facecolor='green', edgecolor='darkgreen', linewidth=2, alpha=0.7
    )
    ax.add_patch(target_rect)
    ax.text(target_start + target_width/2, 0.75, 
            f'Target\n({target_start}-{target_end})', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkgreen')
    
    # Draw mutations (red X's)
    for mut in mutations:
        ax.scatter(mut, 0.5, marker='X', s=200, c='red', zorder=5, edgecolors='darkred', linewidths=1.5)
        ax.text(mut, 0.85, str(mut), ha='center', fontsize=8, color='red')
    
    # Calculate and display distance
    distance, nearest, relation = calculate_distance_to_nearest_mutation(
        target_start, target_end, mutations
    )
    
    if distance > 0:
        # Draw distance arrow
        if relation == 'upstream':
            ax.annotate('', xy=(target_start, 0.5), xytext=(nearest, 0.5),
                       arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
            mid_point = (target_start + nearest) / 2
        else:
            ax.annotate('', xy=(target_end, 0.5), xytext=(nearest, 0.5),
                       arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
            mid_point = (target_end + nearest) / 2
        
        ax.text(mid_point, 0.3, f'{distance} residues', 
                ha='center', fontsize=10, color='blue', fontweight='bold')
    
    # Title and labels
    if distance == 0:
        status = "WARNING: Target OVERLAPS with mutation!"
        title_color = 'red'
    else:
        status = f"SAFE: Target is {distance} residues from nearest mutation ({nearest})"
        title_color = 'green'
    
    ax.set_title(f'Mutation Analysis: {status}', fontsize=12, fontweight='bold', color=title_color)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='darkgreen', alpha=0.7, label='Target Binding Region'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                   markersize=12, markeredgecolor='darkred', label='2024 Immune-Escape Mutations'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Styling
    ax.set_xlim(0, gene_length + 50)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Residue Position', fontsize=11)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved gene map to {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Phase 19: Mutation Analysis")
    print("=" * 60)
    
    print(f"\nTarget Region: {TARGET_START}-{TARGET_END}")
    print(f"2024 Mutations: {MUTATIONS_2024}")
    
    # Calculate distance
    distance, nearest, relation = calculate_distance_to_nearest_mutation(
        TARGET_START, TARGET_END, MUTATIONS_2024
    )
    
    print(f"\n--- Distance Analysis ---")
    if distance == 0:
        print(f"  WARNING: Target OVERLAPS with mutation at position {nearest}!")
    else:
        print(f"  Nearest Mutation: {nearest} ({relation} of target)")
        print(f"  Distance: {distance} residues")
        print(f"  Status: SAFE - No overlap between target and mutations")
    
    # Create visualization
    print(f"\n--- Creating Gene Map ---")
    create_gene_map(TARGET_START, TARGET_END, MUTATIONS_2024, GENE_LENGTH, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("Phase 19 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
