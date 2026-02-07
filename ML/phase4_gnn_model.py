"""
Phase 4: Graph Neural Network for CCP Capital Prediction

This script implements a GNN model to predict the minimum CCP capital requirements
based on the banking network structure and bank-level features.

Input:
  - temporal_simulation_data.csv (training data from Phase 3)
  - interbank_network_matrix.csv (network structure from Phase 2)
  - phase1_engineered_data.csv (bank features)
  
Output:
  - gnn_ccp_capital_model.pth (trained model)
  - model_evaluation_results.json (performance metrics)
  - feature_importance.png (visualization)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("dataset")
SIMULATION_DATA = DATA_DIR / "temporal_simulation_data.csv"
NETWORK_FILE = DATA_DIR / "interbank_network_matrix.csv"
BANK_DATA_FILE = DATA_DIR / "phase1_engineered_data.csv"
MODEL_FILE = DATA_DIR / "gnn_ccp_capital_model.pth"
RESULTS_FILE = DATA_DIR / "model_evaluation_results.json"

# Model hyperparameters
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class GNNModel(nn.Module):
    """
    Graph Attention Network for CCP Capital Prediction
    
    Architecture:
    - Multiple GAT layers to capture network structure
    - Global pooling to get graph-level representation
    - MLP head for regression
    """
    
    def __init__(self, num_node_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(GNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Regression head
        self.fc1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Regression head
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x.squeeze()


def create_graph_data(scenario_row, network_matrix, bank_features, scaler):
    """
    Create PyTorch Geometric Data object for a single scenario
    
    Args:
        scenario_row: Row from simulation data (single scenario)
        network_matrix: Interbank network adjacency matrix
        bank_features: DataFrame with bank-level features
        scaler: Fitted StandardScaler for features
        
    Returns:
        PyTorch Geometric Data object
    """
    shocked_bank_idx = int(scenario_row['shocked_bank_idx'])
    shock_magnitude = scenario_row['shock_magnitude']
    target = scenario_row['min_ccp_capital']
    
    # Node features: bank characteristics + shock indicator
    node_features = []
    for idx in range(len(bank_features)):
        features = [
            bank_features.loc[idx, 'Total_Assets'],
            bank_features.loc[idx, 'Equity'],
            bank_features.loc[idx, 'HQLA'],
            bank_features.loc[idx, 'Net_Outflows_30d'],
            bank_features.loc[idx, 'Stock_Volatility'],
            bank_features.loc[idx, 'Est_CDS_Spread'],
            bank_features.loc[idx, 'Distance_to_Default'],
            bank_features.loc[idx, 'Prob_Default_1Y'],
            1.0 if idx == shocked_bank_idx else 0.0,  # Shock indicator
            shock_magnitude if idx == shocked_bank_idx else 0.0  # Shock magnitude
        ]
        node_features.append(features)
    
    node_features = np.array(node_features, dtype=np.float32)
    
    # Normalize features
    node_features = scaler.transform(node_features)
    
    # Edge index and edge attributes from network matrix
    edge_index = []
    edge_attr = []
    
    for i in range(len(network_matrix)):
        for j in range(len(network_matrix)):
            if network_matrix[i, j] > 0.01:  # Threshold for edge existence
                edge_index.append([i, j])
                edge_attr.append([network_matrix[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.float)
    )
    
    return data


def prepare_data():
    """
    Load and prepare data for training
    """
    print(f"\n{'='*60}")
    print(f"LOADING AND PREPARING DATA")
    print(f"{'='*60}")
    
    # Load simulation data
    print(f"\n✓ Loading simulation data...")
    sim_data = pd.read_csv(SIMULATION_DATA)
    print(f"  Scenarios: {len(sim_data)}")
    
    # Load network matrix
    print(f"\n✓ Loading network matrix...")
    network_df = pd.read_csv(NETWORK_FILE, index_col=0)
    network_matrix = network_df.values
    print(f"  Network shape: {network_matrix.shape}")
    
    # Load bank features
    print(f"\n✓ Loading bank features...")
    bank_features = pd.read_csv(BANK_DATA_FILE)
    print(f"  Banks: {len(bank_features)}")
    
    # Fit scaler on all node features
    print(f"\n✓ Fitting feature scaler...")
    all_features = []
    for idx in range(len(bank_features)):
        features = [
            bank_features.loc[idx, 'Total_Assets'],
            bank_features.loc[idx, 'Equity'],
            bank_features.loc[idx, 'HQLA'],
            bank_features.loc[idx, 'Net_Outflows_30d'],
            bank_features.loc[idx, 'Stock_Volatility'],
            bank_features.loc[idx, 'Est_CDS_Spread'],
            bank_features.loc[idx, 'Distance_to_Default'],
            bank_features.loc[idx, 'Prob_Default_1Y'],
            0.0,  # Shock indicator (placeholder)
            0.0   # Shock magnitude (placeholder)
        ]
        all_features.append(features)
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Create graph data objects
    print(f"\n✓ Creating graph data objects...")
    graph_data_list = []
    for idx, row in sim_data.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing scenario {idx}/{len(sim_data)}...")
        data = create_graph_data(row, network_matrix, bank_features, scaler)
        graph_data_list.append(data)
    
    print(f"  Created {len(graph_data_list)} graph objects")
    
    return graph_data_list, scaler


def train_model(train_loader, val_loader, device):
    """
    Train the GNN model
    """
    print(f"\n{'='*60}")
    print(f"TRAINING GNN MODEL")
    print(f"{'='*60}")
    
    # Initialize model
    num_node_features = 10  # 8 bank features + shock indicator + shock magnitude
    model = GNNModel(num_node_features).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n✓ Model architecture:")
    print(f"  Node features: {num_node_features}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n✓ Training configuration:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Patience: {PATIENCE}")
    
    print(f"\n{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}{'Best Val':<15}")
    print(f"{'-'*60}")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:<10}{train_loss:<15.4f}{val_loss:<15.4f}{best_val_loss:<15.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_FILE))
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set
    """
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            predictions.extend(out.cpu().numpy())
            actuals.extend(batch.y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    results = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'num_test_samples': len(actuals)
    }
    
    print(f"\n✓ Test Set Performance:")
    print(f"  RMSE: ${rmse:.2f}B")
    print(f"  MAE: ${mae:.2f}B")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Test samples: {len(actuals)}")
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {RESULTS_FILE}")
    
    # Visualization
    visualize_results(actuals, predictions)
    
    return results


def visualize_results(actuals, predictions):
    """
    Create visualizations of model performance
    """
    print(f"\n{'='*60}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Predicted vs Actual
    axes[0].scatter(actuals, predictions, alpha=0.5, s=20)
    axes[0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual CCP Capital ($B)', fontsize=11)
    axes[0].set_ylabel('Predicted CCP Capital ($B)', fontsize=11)
    axes[0].set_title('Predicted vs Actual CCP Capital', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. Residuals
    residuals = predictions - actuals
    axes[1].hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual ($B)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Prediction Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    viz_path = DATA_DIR / "model_evaluation.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved evaluation plots: {viz_path}")
    plt.close()


def main():
    """
    Main execution function
    """
    print("="*60)
    print("PHASE 4: GNN MODEL FOR CCP CAPITAL PREDICTION")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Prepare data
    graph_data_list, scaler = prepare_data()
    
    # Split data
    print(f"\n✓ Splitting data...")
    train_data, temp_data = train_test_split(graph_data_list, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train model
    model, train_losses, val_losses = train_model(train_loader, val_loader, device)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("PHASE 4 COMPLETE ✓")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  1. Trained model: {MODEL_FILE}")
    print(f"  2. Evaluation results: {RESULTS_FILE}")
    print(f"  3. Visualizations: {DATA_DIR / 'model_evaluation.png'}")
    print(f"\nModel Performance Summary:")
    print(f"  R² Score: {results['r2']:.4f}")
    print(f"  RMSE: ${results['rmse']:.2f}B")
    print(f"  MAE: ${results['mae']:.2f}B")


if __name__ == "__main__":
    main()
