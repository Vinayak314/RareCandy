"""
Phase 3: Temporal Contagion Simulation with CCP Capital Calculation

This script simulates systemic contagion cascades over time to generate training data
for the GNN model. It calculates the minimum CCP capital required to prevent systemic
collapse under various shock scenarios.

Input: 
  - phase1_engineered_data.csv (bank data)
  - interbank_network_matrix.csv (network structure)
  
Output:
  - temporal_simulation_data.csv (training data for ML model)
  - simulation_summary_stats.json (aggregate statistics)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = Path("dataset")
BANK_DATA_FILE = DATA_DIR / "phase1_engineered_data.csv"
NETWORK_FILE = DATA_DIR / "interbank_network_matrix.csv"
OUTPUT_FILE = DATA_DIR / "temporal_simulation_data.csv"
STATS_FILE = DATA_DIR / "simulation_summary_stats.json"

# Simulation parameters
NUM_SCENARIOS = 10000  # Number of Monte Carlo scenarios
TIME_STEPS = 30  # Simulate 30 days of contagion
SHOCK_MAGNITUDES = [0.10, 0.20, 0.30, 0.40, 0.50]  # 10% to 50% stock price drops
LCR_THRESHOLD = 1.0  # Liquidity Coverage Ratio minimum (100%)
EQUITY_THRESHOLD = 0.0  # Equity must be positive
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


class ContagionSimulator:
    """
    Simulates temporal contagion cascades in the banking network
    """
    
    def __init__(self, bank_data, network_matrix):
        """
        Initialize simulator with bank data and network structure
        
        Args:
            bank_data: DataFrame with bank financial data
            network_matrix: 2D array of interbank exposures
        """
        self.bank_data = bank_data.copy()
        self.network_matrix = network_matrix.copy()
        self.n_banks = len(bank_data)
        self.bank_names = bank_data['Bank'].values
        
    def apply_shock(self, bank_idx, shock_magnitude):
        """
        Apply initial shock to a bank's equity
        
        Args:
            bank_idx: Index of bank to shock
            shock_magnitude: Fraction of market cap to lose (0.0 to 1.0)
            
        Returns:
            Updated bank state
        """
        state = self.bank_data.copy()
        
        # Calculate equity loss from stock price drop
        market_cap = state.loc[bank_idx, 'Market_Cap_E'] / 1e9  # Convert to billions
        equity_loss = shock_magnitude * market_cap
        
        # Update equity
        state.loc[bank_idx, 'Equity'] -= equity_loss
        
        return state
    
    def check_default(self, state, bank_idx):
        """
        Check if a bank has defaulted based on solvency criteria
        
        Args:
            state: Current bank state DataFrame
            bank_idx: Index of bank to check
            
        Returns:
            True if bank has defaulted, False otherwise
        """
        equity = state.loc[bank_idx, 'Equity']
        hqla = state.loc[bank_idx, 'HQLA']
        net_outflows = state.loc[bank_idx, 'Net_Outflows_30d']
        
        # Check equity threshold
        if equity <= EQUITY_THRESHOLD:
            return True
        
        # Check LCR threshold
        if net_outflows > 0:
            lcr = hqla / net_outflows
            if lcr < LCR_THRESHOLD:
                return True
        
        return False
    
    def propagate_contagion_step(self, state, defaulted_banks, network):
        """
        Propagate contagion for one time step
        
        Args:
            state: Current bank state DataFrame
            defaulted_banks: Set of bank indices that have defaulted
            network: Current network matrix
            
        Returns:
            (updated_state, newly_defaulted_banks)
        """
        newly_defaulted = set()
        
        for defaulted_idx in defaulted_banks:
            # Find all creditors of the defaulted bank (column in network matrix)
            creditor_exposures = network[:, defaulted_idx]
            
            for creditor_idx in range(self.n_banks):
                if creditor_idx in defaulted_banks:
                    continue  # Already defaulted
                
                exposure = creditor_exposures[creditor_idx]
                
                if exposure > 0.01:  # Threshold to avoid tiny exposures
                    # Creditor writes off the exposure
                    state.loc[creditor_idx, 'Total_Assets'] -= exposure
                    state.loc[creditor_idx, 'Equity'] -= exposure
                    
                    # Check if creditor now defaults
                    if self.check_default(state, creditor_idx):
                        newly_defaulted.add(creditor_idx)
        
        return state, newly_defaulted
    
    def calculate_ccp_capital(self, state, defaulted_banks, network):
        """
        Calculate CCP capital required to cover losses from defaults
        
        The CCP must cover all interbank exposures to defaulted banks
        to prevent contagion.
        
        Args:
            state: Current bank state DataFrame
            defaulted_banks: Set of defaulted bank indices
            network: Network matrix
            
        Returns:
            Total CCP capital required (in billions)
        """
        total_exposure = 0.0
        
        for defaulted_idx in defaulted_banks:
            # Sum all exposures TO the defaulted bank
            total_exposure += network[:, defaulted_idx].sum()
        
        return total_exposure
    
    def simulate_scenario(self, shocked_bank_idx, shock_magnitude):
        """
        Simulate a complete contagion scenario over T time steps
        
        Args:
            shocked_bank_idx: Index of initially shocked bank
            shock_magnitude: Magnitude of initial shock
            
        Returns:
            Dictionary with simulation results
        """
        # Initialize state
        state = self.apply_shock(shocked_bank_idx, shock_magnitude)
        network = self.network_matrix.copy()
        
        # Track defaults and CCP capital over time
        defaulted_banks = set()
        ccp_capital_timeline = []
        
        # Check if initial shock causes default
        if self.check_default(state, shocked_bank_idx):
            defaulted_banks.add(shocked_bank_idx)
        
        # Simulate contagion over time
        for t in range(TIME_STEPS):
            if len(defaulted_banks) == 0:
                ccp_capital_timeline.append(0.0)
                continue
            
            # Calculate CCP capital needed at this time step
            ccp_capital = self.calculate_ccp_capital(state, defaulted_banks, network)
            ccp_capital_timeline.append(ccp_capital)
            
            # Propagate contagion
            state, newly_defaulted = self.propagate_contagion_step(
                state, defaulted_banks, network
            )
            
            # Add newly defaulted banks
            defaulted_banks.update(newly_defaulted)
            
            # If no new defaults, contagion has stopped
            if len(newly_defaulted) == 0:
                # Fill remaining time steps with current capital requirement
                for _ in range(t + 1, TIME_STEPS):
                    ccp_capital_timeline.append(ccp_capital)
                break
        
        # Calculate peak CCP capital requirement
        min_ccp_capital = max(ccp_capital_timeline) if ccp_capital_timeline else 0.0
        
        # Collect results
        results = {
            'shocked_bank_idx': shocked_bank_idx,
            'shocked_bank_name': self.bank_names[shocked_bank_idx],
            'shock_magnitude': shock_magnitude,
            'num_defaults': len(defaulted_banks),
            'min_ccp_capital': min_ccp_capital,
            'ccp_capital_timeline': ccp_capital_timeline,
            'defaulted_banks': list(defaulted_banks),
            
            # Pre-shock features
            'leverage_ratio': self.bank_data.loc[shocked_bank_idx, 'Total_Assets'] / 
                             self.bank_data.loc[shocked_bank_idx, 'Equity'],
            'lcr': self.bank_data.loc[shocked_bank_idx, 'HQLA'] / 
                   max(self.bank_data.loc[shocked_bank_idx, 'Net_Outflows_30d'], 0.01),
            'stock_volatility': self.bank_data.loc[shocked_bank_idx, 'Stock_Volatility'],
            'cds_spread': self.bank_data.loc[shocked_bank_idx, 'Est_CDS_Spread'],
            'distance_to_default': self.bank_data.loc[shocked_bank_idx, 'Distance_to_Default'],
            'prob_default_1y': self.bank_data.loc[shocked_bank_idx, 'Prob_Default_1Y'],
            'equity': self.bank_data.loc[shocked_bank_idx, 'Equity'],
            'total_assets': self.bank_data.loc[shocked_bank_idx, 'Total_Assets'],
        }
        
        return results


def run_monte_carlo_simulation(simulator, num_scenarios):
    """
    Run Monte Carlo simulation with random shocks
    
    Args:
        simulator: ContagionSimulator instance
        num_scenarios: Number of scenarios to simulate
        
    Returns:
        List of scenario results
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"RUNNING MONTE CARLO SIMULATION")
    print(f"{'='*60}")
    print(f"Number of scenarios: {num_scenarios}")
    print(f"Time horizon: {TIME_STEPS} days")
    print(f"Shock magnitudes: {[f'{s*100:.0f}%' for s in SHOCK_MAGNITUDES]}")
    
    for scenario_id in tqdm(range(num_scenarios), desc="Simulating scenarios"):
        # Randomly select bank and shock magnitude
        bank_idx = np.random.randint(0, simulator.n_banks)
        shock_mag = np.random.choice(SHOCK_MAGNITUDES)
        
        # Run simulation
        result = simulator.simulate_scenario(bank_idx, shock_mag)
        result['scenario_id'] = scenario_id
        results.append(result)
    
    return results


def analyze_results(results):
    """
    Analyze simulation results and generate statistics
    
    Args:
        results: List of scenario results
        
    Returns:
        Dictionary of summary statistics
    """
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'ccp_capital_timeline' and k != 'defaulted_banks'}
        for r in results
    ])
    
    stats = {
        'total_scenarios': len(results),
        'avg_ccp_capital': float(df['min_ccp_capital'].mean()),
        'max_ccp_capital': float(df['min_ccp_capital'].max()),
        'median_ccp_capital': float(df['min_ccp_capital'].median()),
        'std_ccp_capital': float(df['min_ccp_capital'].std()),
        'avg_defaults': float(df['num_defaults'].mean()),
        'max_defaults': int(df['num_defaults'].max()),
        'scenarios_with_contagion': int((df['num_defaults'] > 1).sum()),
        'contagion_rate': float((df['num_defaults'] > 1).mean()),
    }
    
    print(f"\n✓ CCP Capital Requirements:")
    print(f"  Average: ${stats['avg_ccp_capital']:.2f}B")
    print(f"  Median: ${stats['median_ccp_capital']:.2f}B")
    print(f"  Maximum: ${stats['max_ccp_capital']:.2f}B")
    print(f"  Std Dev: ${stats['std_ccp_capital']:.2f}B")
    
    print(f"\n✓ Contagion Statistics:")
    print(f"  Average defaults per scenario: {stats['avg_defaults']:.2f}")
    print(f"  Maximum defaults: {stats['max_defaults']}")
    print(f"  Scenarios with contagion: {stats['scenarios_with_contagion']} ({stats['contagion_rate']*100:.1f}%)")
    
    # Top vulnerable banks
    bank_shock_impact = df.groupby('shocked_bank_name')['min_ccp_capital'].mean().sort_values(ascending=False)
    print(f"\n✓ Top 10 Most Systemically Important Banks (by avg CCP capital impact):")
    for i, (bank, capital) in enumerate(bank_shock_impact.head(10).items(), 1):
        print(f"  {i}. {bank}: ${capital:.2f}B")
    
    return stats


def visualize_results(results, output_dir=DATA_DIR):
    """
    Create visualizations of simulation results
    """
    print(f"\n{'='*60}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'ccp_capital_timeline' and k != 'defaulted_banks'}
        for r in results
    ])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. CCP Capital Distribution
    axes[0, 0].hist(df['min_ccp_capital'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Minimum CCP Capital Required ($B)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of CCP Capital Requirements', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Number of Defaults Distribution
    axes[0, 1].hist(df['num_defaults'], bins=range(0, df['num_defaults'].max()+2), 
                    color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Defaults', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Distribution of Cascade Size', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. CCP Capital vs Shock Magnitude
    shock_impact = df.groupby('shock_magnitude')['min_ccp_capital'].agg(['mean', 'std'])
    axes[1, 0].bar(shock_impact.index * 100, shock_impact['mean'], 
                   yerr=shock_impact['std'], color='green', alpha=0.7, capsize=5)
    axes[1, 0].set_xlabel('Shock Magnitude (%)', fontsize=11)
    axes[1, 0].set_ylabel('Average CCP Capital ($B)', fontsize=11)
    axes[1, 0].set_title('CCP Capital vs Shock Severity', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. CDS Spread vs CCP Capital (scatter)
    axes[1, 1].scatter(df['cds_spread'], df['min_ccp_capital'], 
                       alpha=0.3, c=df['shock_magnitude'], cmap='YlOrRd', s=20)
    axes[1, 1].set_xlabel('CDS Spread (bps)', fontsize=11)
    axes[1, 1].set_ylabel('CCP Capital Required ($B)', fontsize=11)
    axes[1, 1].set_title('CDS Spread vs Systemic Impact', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Shock Magnitude', fontsize=10)
    
    plt.tight_layout()
    viz_path = output_dir / "simulation_results.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved simulation visualizations: {viz_path}")
    plt.close()


def main():
    """
    Main execution function
    """
    print("="*60)
    print("PHASE 3: TEMPORAL CONTAGION SIMULATION")
    print("Monte Carlo Simulation with CCP Capital Calculation")
    print("="*60)
    
    # Load data
    print(f"\n✓ Loading data...")
    bank_data = pd.read_csv(BANK_DATA_FILE)
    network_df = pd.read_csv(NETWORK_FILE, index_col=0)
    network_matrix = network_df.values
    
    print(f"  Banks: {len(bank_data)}")
    print(f"  Network shape: {network_matrix.shape}")
    
    # Initialize simulator
    simulator = ContagionSimulator(bank_data, network_matrix)
    
    # Run Monte Carlo simulation
    results = run_monte_carlo_simulation(simulator, NUM_SCENARIOS)
    
    # Analyze results
    stats = analyze_results(results)
    
    # Save statistics
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved summary statistics: {STATS_FILE}")
    
    # Create visualizations
    visualize_results(results)
    
    # Save training data
    print(f"\n✓ Saving training data...")
    training_data = []
    for r in results:
        row = {k: v for k, v in r.items() 
               if k not in ['ccp_capital_timeline', 'defaulted_banks', 'shocked_bank_name']}
        training_data.append(row)
    
    training_df = pd.DataFrame(training_data)
    training_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved {len(training_df)} scenarios to: {OUTPUT_FILE}")
    print(f"  Features: {list(training_df.columns)}")
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE ✓")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  1. Training data: {OUTPUT_FILE}")
    print(f"  2. Summary stats: {STATS_FILE}")
    print(f"  3. Visualizations: {DATA_DIR / 'simulation_results.png'}")
    print(f"\nNext step: Run phase4_gnn_model.py to train the GNN")


if __name__ == "__main__":
    main()
