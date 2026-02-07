"""
Phase 2: Network Reconstruction using Maximum Entropy Method (RAS Algorithm)

This script reconstructs the bilateral interbank exposure matrix from aggregate
Interbank_Assets and Interbank_Liabilities data using the RAS algorithm
(iterative proportional fitting).

Input: phase1_engineered_data.csv
Output: interbank_network_matrix.csv (50x50 matrix)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path("dataset")
INPUT_FILE = DATA_DIR / "phase1_engineered_data.csv"
OUTPUT_FILE = DATA_DIR / "interbank_network_matrix.csv"
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 1000


def ras_algorithm(row_totals, col_totals, max_iter=MAX_ITERATIONS, tol=CONVERGENCE_THRESHOLD):
    """
    RAS Algorithm (Iterative Proportional Fitting) for Maximum Entropy Network Reconstruction
    
    Given:
    - row_totals: Total interbank assets (lending) for each bank
    - col_totals: Total interbank liabilities (borrowing) for each bank
    
    Returns:
    - X: n x n matrix where X[i,j] = amount Bank i lent to Bank j
    
    The algorithm finds the matrix that:
    1. Matches the row and column constraints
    2. Has maximum entropy (most uniform distribution given constraints)
    """
    n = len(row_totals)
    
    # Initialize with uniform distribution (gravity model baseline)
    # X[i,j] proportional to assets[i] * liabilities[j]
    X = np.outer(row_totals, col_totals)
    
    # Avoid division by zero
    total_assets = row_totals.sum()
    if total_assets > 0:
        X = X / total_assets
    else:
        raise ValueError("Total interbank assets is zero - cannot reconstruct network")
    
    # Iteratively adjust to match row and column totals
    for iteration in range(max_iter):
        X_old = X.copy()
        
        # Step 1: Adjust rows to match row_totals (interbank assets)
        row_sums = X.sum(axis=1)
        # Avoid division by zero
        row_factors = np.where(row_sums > 0, row_totals / row_sums, 0)
        X = X * row_factors[:, np.newaxis]
        
        # Step 2: Adjust columns to match col_totals (interbank liabilities)
        col_sums = X.sum(axis=0)
        # Avoid division by zero
        col_factors = np.where(col_sums > 0, col_totals / col_sums, 0)
        X = X * col_factors[np.newaxis, :]
        
        # Check convergence
        max_change = np.abs(X - X_old).max()
        if max_change < tol:
            print(f"✓ RAS algorithm converged in {iteration + 1} iterations")
            print(f"  Max change: {max_change:.2e}")
            break
    else:
        print(f"⚠ Warning: RAS algorithm did not converge after {max_iter} iterations")
        print(f"  Max change: {max_change:.2e}")
    
    return X


def validate_network(matrix, row_totals, col_totals, bank_names):
    """
    Validate that the reconstructed network satisfies constraints
    """
    print("\n" + "="*60)
    print("NETWORK VALIDATION")
    print("="*60)
    
    # Check row sums (interbank assets)
    row_sums = matrix.sum(axis=1)
    row_error = np.abs(row_sums - row_totals).max()
    print(f"\n✓ Row sum validation (Interbank Assets):")
    print(f"  Max absolute error: ${row_error:.2f}B")
    print(f"  Max relative error: {(row_error / row_totals.max() * 100):.4f}%")
    
    # Check column sums (interbank liabilities)
    col_sums = matrix.sum(axis=0)
    col_error = np.abs(col_sums - col_totals).max()
    print(f"\n✓ Column sum validation (Interbank Liabilities):")
    print(f"  Max absolute error: ${col_error:.2f}B")
    print(f"  Max relative error: {(col_error / col_totals.max() * 100):.4f}%")
    
    # Network statistics
    print(f"\n✓ Network Statistics:")
    print(f"  Total network volume: ${matrix.sum():.2f}B")
    print(f"  Number of banks: {len(bank_names)}")
    print(f"  Total possible links: {len(bank_names) * (len(bank_names) - 1)}")
    print(f"  Non-zero exposures: {(matrix > 0.01).sum()}")  # Count exposures > $10M
    print(f"  Network density: {(matrix > 0.01).sum() / (len(bank_names) * (len(bank_names) - 1)) * 100:.2f}%")
    
    # Top exposures
    print(f"\n✓ Top 10 Bilateral Exposures:")
    flat_indices = np.argsort(matrix.flatten())[::-1][:10]
    for rank, idx in enumerate(flat_indices, 1):
        i, j = np.unravel_index(idx, matrix.shape)
        if i != j:  # Skip self-loops
            print(f"  {rank}. {bank_names[i]} → {bank_names[j]}: ${matrix[i, j]:.2f}B")
    
    return row_error < 1e-3 and col_error < 1e-3


def visualize_network(matrix, bank_names, output_dir=DATA_DIR):
    """
    Create visualizations of the reconstructed network
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Heatmap of full network
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, cmap='YlOrRd', cbar_kws={'label': 'Exposure ($B)'})
    plt.title('Reconstructed Interbank Network Matrix\n(Rows = Lenders, Columns = Borrowers)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Borrowing Bank', fontsize=12)
    plt.ylabel('Lending Bank', fontsize=12)
    plt.tight_layout()
    heatmap_path = output_dir / "network_heatmap_full.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved full network heatmap: {heatmap_path}")
    plt.close()
    
    # 2. Heatmap of top 20 banks (for readability)
    top_20_indices = np.argsort(matrix.sum(axis=0) + matrix.sum(axis=1))[-20:]
    matrix_top20 = matrix[np.ix_(top_20_indices, top_20_indices)]
    bank_names_top20 = [bank_names[i] for i in top_20_indices]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix_top20, xticklabels=bank_names_top20, yticklabels=bank_names_top20,
                cmap='YlOrRd', cbar_kws={'label': 'Exposure ($B)'}, annot=False)
    plt.title('Top 20 Most Connected Banks\n(Rows = Lenders, Columns = Borrowers)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Borrowing Bank', fontsize=11)
    plt.ylabel('Lending Bank', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    heatmap_top20_path = output_dir / "network_heatmap_top20.png"
    plt.savefig(heatmap_top20_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved top 20 banks heatmap: {heatmap_top20_path}")
    plt.close()
    
    # 3. Degree distribution
    in_degree = (matrix > 0.01).sum(axis=0)  # Number of creditors
    out_degree = (matrix > 0.01).sum(axis=1)  # Number of debtors
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(in_degree, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('In-Degree (Number of Creditors)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('In-Degree Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].hist(out_degree, bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Out-Degree (Number of Debtors)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Out-Degree Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    degree_path = output_dir / "network_degree_distribution.png"
    plt.savefig(degree_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved degree distribution: {degree_path}")
    plt.close()


def main():
    """
    Main execution function
    """
    print("="*60)
    print("PHASE 2: NETWORK RECONSTRUCTION")
    print("Maximum Entropy Method (RAS Algorithm)")
    print("="*60)
    
    # Load data
    print(f"\n✓ Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} banks")
    
    # Extract relevant columns
    bank_names = df['Bank'].values
    interbank_assets = df['Interbank_Assets'].values  # Total lending
    interbank_liabilities = df['Interbank_Liabilities'].values  # Total borrowing
    
    print(f"\n✓ Data Summary:")
    print(f"  Total Interbank Assets: ${interbank_assets.sum():.2f}B")
    print(f"  Total Interbank Liabilities: ${interbank_liabilities.sum():.2f}B")
    print(f"  Balance check: ${abs(interbank_assets.sum() - interbank_liabilities.sum()):.2f}B difference")
    
    # Run RAS algorithm
    print(f"\n✓ Running RAS algorithm...")
    print(f"  Convergence threshold: {CONVERGENCE_THRESHOLD}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    
    network_matrix = ras_algorithm(interbank_assets, interbank_liabilities)
    
    # Validate results
    is_valid = validate_network(network_matrix, interbank_assets, interbank_liabilities, bank_names)
    
    if is_valid:
        print("\n✓ Network reconstruction SUCCESSFUL")
    else:
        print("\n⚠ Warning: Network validation failed - check constraints")
    
    # Save matrix
    print(f"\n✓ Saving network matrix to: {OUTPUT_FILE}")
    network_df = pd.DataFrame(network_matrix, index=bank_names, columns=bank_names)
    network_df.to_csv(OUTPUT_FILE)
    print(f"  Matrix shape: {network_matrix.shape}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    
    # Create visualizations
    visualize_network(network_matrix, bank_names)
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE ✓")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  1. Network matrix: {OUTPUT_FILE}")
    print(f"  2. Full heatmap: {DATA_DIR / 'network_heatmap_full.png'}")
    print(f"  3. Top 20 heatmap: {DATA_DIR / 'network_heatmap_top20.png'}")
    print(f"  4. Degree distribution: {DATA_DIR / 'network_degree_distribution.png'}")
    print(f"\nNext step: Run phase3_temporal_contagion_simulation.py")


if __name__ == "__main__":
    main()
