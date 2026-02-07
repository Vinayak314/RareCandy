"""
Rigorous Model Validation: Compare Predictions vs Ground Truth

This script performs comprehensive validation by:
1. Running actual contagion simulations for specific scenarios
2. Comparing model predictions against simulation results
3. Analyzing prediction errors across different conditions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("dataset")
SIMULATION_DATA = DATA_DIR / "temporal_simulation_data.csv"
NETWORK_FILE = DATA_DIR / "interbank_network_matrix.csv"
BANK_DATA_FILE = DATA_DIR / "phase1_engineered_data.csv"
MODEL_FILE = DATA_DIR / "gnn_ccp_capital_model.pth"

# Model hyperparameters
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.2

# Simulation parameters
TIME_STEPS = 30
LCR_THRESHOLD = 1.0
EQUITY_THRESHOLD = 0.0


class GNNModel(nn.Module):
    """Graph Attention Network for CCP Capital Prediction"""
    
    def __init__(self, num_node_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(GNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        self.fc1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x.squeeze()


def run_actual_simulation(shocked_bank_idx, shock_magnitude, network_matrix, bank_data):
    """
    Run actual contagion simulation to get ground truth
    """
    # Initialize state
    state = bank_data.copy()
    
    # Apply shock
    state.loc[shocked_bank_idx, 'Equity'] *= (1 - shock_magnitude)
    
    # Track defaults and CCP capital
    defaulted_banks = set()
    ccp_capital_timeline = []
    
    # Check if shocked bank defaults
    equity = state.loc[shocked_bank_idx, 'Equity']
    lcr = state.loc[shocked_bank_idx, 'HQLA'] / max(state.loc[shocked_bank_idx, 'Net_Outflows_30d'], 0.01)
    
    if equity <= EQUITY_THRESHOLD or lcr < LCR_THRESHOLD:
        defaulted_banks.add(shocked_bank_idx)
    
    # Propagate contagion over time
    for t in range(TIME_STEPS):
        if len(defaulted_banks) == 0:
            ccp_capital_timeline.append(0.0)
            continue
        
        # Calculate CCP capital needed
        total_exposure = 0.0
        for defaulted_idx in defaulted_banks:
            total_exposure += network_matrix[:, defaulted_idx].sum()
        ccp_capital_timeline.append(total_exposure)
        
        # Propagate losses
        newly_defaulted = set()
        for defaulted_idx in defaulted_banks:
            creditor_losses = network_matrix[:, defaulted_idx]
            
            for creditor_idx in range(len(state)):
                if creditor_idx in defaulted_banks:
                    continue
                
                loss = creditor_losses[creditor_idx]
                if loss > 0:
                    state.loc[creditor_idx, 'Equity'] -= loss
                    
                    # Check if creditor defaults
                    equity = state.loc[creditor_idx, 'Equity']
                    lcr = state.loc[creditor_idx, 'HQLA'] / max(state.loc[creditor_idx, 'Net_Outflows_30d'], 0.01)
                    
                    if equity <= EQUITY_THRESHOLD or lcr < LCR_THRESHOLD:
                        newly_defaulted.add(creditor_idx)
        
        defaulted_banks.update(newly_defaulted)
        
        if len(newly_defaulted) == 0:
            for _ in range(t + 1, TIME_STEPS):
                ccp_capital_timeline.append(total_exposure)
            break
    
    min_ccp_capital = max(ccp_capital_timeline) if ccp_capital_timeline else 0.0
    
    return {
        'min_ccp_capital': min_ccp_capital,
        'num_defaults': len(defaulted_banks),
        'timeline': ccp_capital_timeline
    }


def create_graph_data(shocked_bank_idx, shock_magnitude, network_matrix, bank_features, scaler):
    """Create graph data for model prediction"""
    
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
    
    edge_index = []
    edge_attr = []
    
    for i in range(len(network_matrix)):
        for j in range(len(network_matrix)):
            if network_matrix[i, j] > 0.01:
                edge_index.append([i, j])
                edge_attr.append([network_matrix[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(len(node_features), dtype=torch.long)
    )
    
    return data


def validate_model():
    """
    Comprehensive model validation
    """
    print("="*70)
    print("RIGOROUS MODEL VALIDATION: PREDICTIONS VS GROUND TRUTH")
    print("="*70)
    
    # Load data
    print("\n✓ Loading data...")
    network_df = pd.read_csv(NETWORK_FILE, index_col=0)
    network_matrix = network_df.values
    bank_features = pd.read_csv(BANK_DATA_FILE)
    sim_data = pd.read_csv(SIMULATION_DATA)
    
    # Fit scaler
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
            0.0, 0.0
        ]
        all_features.append(features)
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Load model
    print("✓ Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=10).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    
    # Select diverse test scenarios
    print("\n✓ Selecting test scenarios...")
    test_scenarios = []
    
    # Strategy 1: Different shock magnitudes for same bank
    for shock in [0.1, 0.2, 0.3, 0.4, 0.5]:
        test_scenarios.append({'bank_idx': 0, 'shock': shock, 'bank_name': 'JPM'})
    
    # Strategy 2: Same shock for different banks (small, medium, large)
    test_banks = [
        (0, 'JPM', 'Large'),
        (15, 'CFG', 'Medium'),
        (48, 'IBOC', 'Small')
    ]
    for bank_idx, bank_name, size in test_banks:
        test_scenarios.append({'bank_idx': bank_idx, 'shock': 0.3, 'bank_name': f'{bank_name} ({size})'})
    
    # Strategy 3: Random scenarios from test set
    test_sample = sim_data.sample(n=10, random_state=42)
    for _, row in test_sample.iterrows():
        bank_name = bank_features.loc[int(row['shocked_bank_idx']), 'Bank']
        test_scenarios.append({
            'bank_idx': int(row['shocked_bank_idx']),
            'shock': row['shock_magnitude'],
            'bank_name': bank_name,
            'ground_truth': row['min_ccp_capital']
        })
    
    # Run validation
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS ({len(test_scenarios)} scenarios)")
    print(f"{'='*70}")
    
    results = []
    
    print(f"\n{'Scenario':<25} {'Ground Truth':<15} {'Prediction':<15} {'Error':<15} {'Error %':<10}")
    print("-"*85)
    
    for i, scenario in enumerate(test_scenarios):
        bank_idx = scenario['bank_idx']
        shock = scenario['shock']
        bank_name = scenario['bank_name']
        
        # Get ground truth
        if 'ground_truth' in scenario:
            ground_truth = scenario['ground_truth']
        else:
            sim_result = run_actual_simulation(bank_idx, shock, network_matrix, bank_features)
            ground_truth = sim_result['min_ccp_capital']
        
        # Get model prediction
        data = create_graph_data(bank_idx, shock, network_matrix, bank_features, scaler)
        data = data.to(device)
        
        with torch.no_grad():
            prediction = model(data).item()
        
        # Calculate error
        error = prediction - ground_truth
        error_pct = (error / (ground_truth + 1e-8)) * 100 if ground_truth > 0 else 0
        
        scenario_name = f"{bank_name} ({int(shock*100)}%)"
        print(f"{scenario_name:<25} ${ground_truth:<14.2f} ${prediction:<14.2f} ${error:<14.2f} {error_pct:<9.1f}%")
        
        results.append({
            'scenario': scenario_name,
            'bank_idx': bank_idx,
            'shock': shock,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'error': error,
            'error_pct': error_pct
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    
    mae = np.abs(results_df['error']).mean()
    rmse = np.sqrt((results_df['error']**2).mean())
    mape = np.abs(results_df['error_pct']).mean()
    r2 = r2_score(results_df['ground_truth'], results_df['prediction'])
    
    print(f"\nOverall Metrics:")
    print(f"  MAE:  ${mae:.2f}B")
    print(f"  RMSE: ${rmse:.2f}B")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  R²:   {r2:.4f}")
    
    # Error by shock magnitude
    print(f"\nError by Shock Magnitude:")
    for shock in sorted(results_df['shock'].unique()):
        subset = results_df[results_df['shock'] == shock]
        print(f"  {int(shock*100)}% shock: MAE = ${np.abs(subset['error']).mean():.2f}B, "
              f"MAPE = {np.abs(subset['error_pct']).mean():.1f}%")
    
    # Visualizations
    create_validation_plots(results_df)
    
    # Save results
    output_file = DATA_DIR / "validation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved validation results to: {output_file}")
    
    return results_df


def create_validation_plots(results_df):
    """Create validation visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(results_df['ground_truth'], results_df['prediction'], alpha=0.7, s=100)
    ax.plot([results_df['ground_truth'].min(), results_df['ground_truth'].max()],
            [results_df['ground_truth'].min(), results_df['ground_truth'].max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Ground Truth CCP Capital ($B)', fontsize=11)
    ax.set_ylabel('Predicted CCP Capital ($B)', fontsize=11)
    ax.set_title('Predicted vs Ground Truth', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(results_df['error'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Prediction Error ($B)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Error by shock magnitude
    ax = axes[1, 0]
    shock_groups = results_df.groupby('shock')['error'].apply(list)
    positions = range(len(shock_groups))
    ax.boxplot(shock_groups.values, positions=positions, widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(s*100)}%" for s in shock_groups.index])
    ax.axhline(y=0, color='r', linestyle='--', lw=1)
    ax.set_xlabel('Shock Magnitude', fontsize=11)
    ax.set_ylabel('Prediction Error ($B)', fontsize=11)
    ax.set_title('Error by Shock Magnitude', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Absolute percentage error
    ax = axes[1, 1]
    scenarios = results_df['scenario'].values
    errors = np.abs(results_df['error_pct']).values
    colors = ['green' if e < 20 else 'orange' if e < 50 else 'red' for e in errors]
    
    y_pos = np.arange(len(scenarios))
    ax.barh(y_pos, errors, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios, fontsize=8)
    ax.set_xlabel('Absolute Error (%)', fontsize=11)
    ax.set_title('Percentage Error by Scenario', fontsize=12, fontweight='bold')
    ax.axvline(x=20, color='orange', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(x=50, color='red', linestyle='--', lw=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    viz_path = DATA_DIR / "validation_plots.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved validation plots: {viz_path}")
    plt.close()


if __name__ == "__main__":
    validate_model()
