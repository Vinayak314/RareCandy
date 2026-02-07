"""
Prediction Script: Use Trained GNN Model for CCP Capital Prediction

This script demonstrates how to use the trained GNN model to predict
minimum CCP capital requirements for new shock scenarios.

Usage:
    python predict_ccp_capital.py --shocked_bank 0 --shock_magnitude 0.3
    python predict_ccp_capital.py --random_scenarios 10
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("dataset")
NETWORK_FILE = DATA_DIR / "interbank_network_matrix.csv"
BANK_DATA_FILE = DATA_DIR / "phase1_engineered_data.csv"
MODEL_FILE = DATA_DIR / "gnn_ccp_capital_model.pth"

# Model hyperparameters (must match training)
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.2


class GNNModel(nn.Module):
    """Graph Attention Network for CCP Capital Prediction"""
    
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


def load_model_and_data():
    """Load trained model, network, and bank data"""
    print("="*60)
    print("LOADING MODEL AND DATA")
    print("="*60)
    
    # Load network matrix
    print("\n✓ Loading network matrix...")
    network_df = pd.read_csv(NETWORK_FILE, index_col=0)
    network_matrix = network_df.values
    print(f"  Network shape: {network_matrix.shape}")
    
    # Load bank features
    print("\n✓ Loading bank features...")
    bank_features = pd.read_csv(BANK_DATA_FILE)
    print(f"  Banks: {len(bank_features)}")
    
    # Fit scaler
    print("\n✓ Fitting feature scaler...")
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
            0.0,  # Shock indicator
            0.0   # Shock magnitude
        ]
        all_features.append(features)
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Load model
    print("\n✓ Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=10).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print(f"  Model loaded on: {device}")
    
    return model, network_matrix, bank_features, scaler, device


def create_graph_data(shocked_bank_idx, shock_magnitude, network_matrix, bank_features, scaler):
    """Create graph data for prediction"""
    
    # Node features
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
            1.0 if idx == shocked_bank_idx else 0.0,
            shock_magnitude if idx == shocked_bank_idx else 0.0
        ]
        node_features.append(features)
    
    node_features = np.array(node_features, dtype=np.float32)
    node_features = scaler.transform(node_features)
    
    # Edge index and attributes
    edge_index = []
    edge_attr = []
    
    for i in range(len(network_matrix)):
        for j in range(len(network_matrix)):
            if network_matrix[i, j] > 0.01:
                edge_index.append([i, j])
                edge_attr.append([network_matrix[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(len(node_features), dtype=torch.long)  # Single graph
    )
    
    return data


def predict_single_scenario(shocked_bank_idx, shock_magnitude, model, network_matrix, 
                           bank_features, scaler, device):
    """Predict CCP capital for a single scenario"""
    
    # Create graph data
    data = create_graph_data(shocked_bank_idx, shock_magnitude, network_matrix, 
                            bank_features, scaler)
    data = data.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(data).item()
    
    return prediction


def predict_multiple_scenarios(num_scenarios, model, network_matrix, bank_features, scaler, device):
    """Predict CCP capital for multiple random scenarios"""
    
    print(f"\n{'='*60}")
    print(f"PREDICTING {num_scenarios} RANDOM SCENARIOS")
    print(f"{'='*60}")
    
    results = []
    shock_magnitudes = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for i in range(num_scenarios):
        # Random scenario
        shocked_bank_idx = np.random.randint(0, len(bank_features))
        shock_magnitude = np.random.choice(shock_magnitudes)
        
        # Predict
        prediction = predict_single_scenario(
            shocked_bank_idx, shock_magnitude, model, network_matrix, 
            bank_features, scaler, device
        )
        
        results.append({
            'scenario_id': i,
            'shocked_bank': bank_features.loc[shocked_bank_idx, 'Bank'],
            'shocked_bank_idx': shocked_bank_idx,
            'shock_magnitude': shock_magnitude,
            'predicted_ccp_capital': prediction
        })
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{num_scenarios} scenarios...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"\nPredicted CCP Capital Statistics:")
    print(f"  Mean: ${results_df['predicted_ccp_capital'].mean():.2f}B")
    print(f"  Median: ${results_df['predicted_ccp_capital'].median():.2f}B")
    print(f"  Min: ${results_df['predicted_ccp_capital'].min():.2f}B")
    print(f"  Max: ${results_df['predicted_ccp_capital'].max():.2f}B")
    print(f"  Std: ${results_df['predicted_ccp_capital'].std():.2f}B")
    
    # Group by shock magnitude
    print(f"\nBy Shock Magnitude:")
    for mag in sorted(results_df['shock_magnitude'].unique()):
        subset = results_df[results_df['shock_magnitude'] == mag]
        print(f"  {int(mag*100)}% shock: Mean CCP Capital = ${subset['predicted_ccp_capital'].mean():.2f}B")
    
    # Save results
    output_file = DATA_DIR / "prediction_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved predictions to: {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Predict CCP Capital Requirements')
    parser.add_argument('--shocked_bank', type=int, default=None,
                       help='Index of shocked bank (0-48)')
    parser.add_argument('--shock_magnitude', type=float, default=None,
                       help='Shock magnitude (0.1-0.5)')
    parser.add_argument('--random_scenarios', type=int, default=None,
                       help='Number of random scenarios to predict')
    
    args = parser.parse_args()
    
    # Load model and data
    model, network_matrix, bank_features, scaler, device = load_model_and_data()
    
    if args.random_scenarios:
        # Predict multiple random scenarios
        predict_multiple_scenarios(args.random_scenarios, model, network_matrix, 
                                  bank_features, scaler, device)
    
    elif args.shocked_bank is not None and args.shock_magnitude is not None:
        # Predict single scenario
        print(f"\n{'='*60}")
        print("SINGLE SCENARIO PREDICTION")
        print(f"{'='*60}")
        
        shocked_bank_name = bank_features.loc[args.shocked_bank, 'Bank']
        
        print(f"\nScenario:")
        print(f"  Shocked Bank: {shocked_bank_name} (Index: {args.shocked_bank})")
        print(f"  Shock Magnitude: {args.shock_magnitude*100:.0f}%")
        
        prediction = predict_single_scenario(
            args.shocked_bank, args.shock_magnitude, model, network_matrix,
            bank_features, scaler, device
        )
        
        print(f"\nPrediction:")
        print(f"  Minimum CCP Capital Required: ${prediction:.2f}B")
    
    else:
        # Interactive mode
        print(f"\n{'='*60}")
        print("INTERACTIVE PREDICTION MODE")
        print(f"{'='*60}")
        
        print(f"\nAvailable banks (0-{len(bank_features)-1}):")
        for idx, row in bank_features.iterrows():
            print(f"  {idx}: {row['Bank']}")
        
        shocked_bank_idx = int(input(f"\nEnter shocked bank index (0-{len(bank_features)-1}): "))
        shock_magnitude = float(input("Enter shock magnitude (e.g., 0.3 for 30%): "))
        
        shocked_bank_name = bank_features.loc[shocked_bank_idx, 'Bank']
        
        print(f"\nPredicting for:")
        print(f"  Bank: {shocked_bank_name}")
        print(f"  Shock: {shock_magnitude*100:.0f}%")
        
        prediction = predict_single_scenario(
            shocked_bank_idx, shock_magnitude, model, network_matrix,
            bank_features, scaler, device
        )
        
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"\nMinimum CCP Capital Required: ${prediction:.2f}B")


if __name__ == "__main__":
    main()
