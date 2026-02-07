"""
MODEL2.PY IMPROVEMENTS - Enhanced Contagion Simulation
=======================================================
Key improvements to integrate with model.py for a complete CCP system.
"""

# ============================================================
# IMPROVEMENT 1: Use Real Interbank Matrix Instead of Random Graph
# ============================================================
# Current: generate_random_graph_with_sccs() creates random connections
# Better: Use the actual us_banks_interbank_matrix.csv for real exposures

def load_interbank_from_matrix(csv_path):
    """Load actual interbank exposures from matrix CSV."""
    import pandas as pd
    matrix = pd.read_csv(csv_path, index_col=0)
    
    # Build graph from matrix (non-zero entries = connections)
    graph = {}
    for bank in matrix.index:
        neighbors = []
        for other in matrix.columns:
            if bank != other and matrix.loc[bank, other] > 0:
                neighbors.append(other)
        graph[bank] = {'neighbors': neighbors}
    
    return graph, matrix


# ============================================================
# IMPROVEMENT 2: Stock-Induced Shocks (Connect to Stock Holdings)
# ============================================================
# Current: Holdings are distributed but not used in shock simulation
# Better: When stock crashes, calculate impact on holding banks

def simulate_stock_crash(holdings, stock_prices, crashed_ticker, crash_pct):
    """
    Simulate a stock crash and return bank losses.
    
    Args:
        holdings: {bank: {ticker: shares}}
        stock_prices: {ticker: price}
        crashed_ticker: Ticker that crashed
        crash_pct: Percentage drop (0-100)
    
    Returns:
        {bank: loss_in_billions}
    """
    losses = {}
    original_price = stock_prices[crashed_ticker]
    loss_per_share = original_price * (crash_pct / 100)
    
    for bank, bank_holdings in holdings.items():
        shares_held = bank_holdings.get(crashed_ticker, 0)
        loss = shares_held * loss_per_share / 1e9  # Convert to billions
        if loss > 0:
            losses[bank] = loss
    
    return losses


# ============================================================
# IMPROVEMENT 3: Multi-Channel Contagion
# ============================================================
# Current: Only interbank credit channel
# Better: Add market channel (fire sales), funding channel, confidence

class EnhancedContagion:
    """Enhanced contagion with multiple transmission channels."""
    
    def __init__(self, graph, stock_holdings, stock_prices):
        self.graph = graph
        self.stock_holdings = stock_holdings
        self.stock_prices = stock_prices.copy()
    
    def propagate_with_fire_sales(self, initial_bank, shock_pct, failure_threshold=20):
        """
        Contagion with fire sale effects:
        - When bank fails, it sells assets at discount
        - This pushes down stock prices
        - Other banks holding same stocks get marked down
        """
        # Step 1: Direct shock
        # ... apply initial shock ...
        
        # Step 2: Fire sales - failing bank dumps holdings
        failed_holdings = self.stock_holdings.get(initial_bank, {})
        for ticker, shares in failed_holdings.items():
            # Price impact: 1% drop per 10M shares sold (simplified)
            price_impact = min(0.20, shares / 1e7 * 0.01)  # Cap at 20%
            self.stock_prices[ticker] *= (1 - price_impact)
        
        # Step 3: Mark-to-market losses for all banks
        market_losses = {}
        for bank in self.graph:
            if bank == initial_bank:
                continue
            loss = sum(
                shares * (self.stock_prices[t] - orig_price)
                for t, shares in self.stock_holdings.get(bank, {}).items()
                for orig_price in [self.stock_prices[t] / (1 - 0.05)]  # Approximate
            )
            market_losses[bank] = abs(loss) / 1e9
        
        # Continue with regular contagion...
        return market_losses


# ============================================================
# IMPROVEMENT 4: CCP Integration
# ============================================================
# Combine model.py (RL-based) with model2.py (simulation-based)

def integrated_ccp_check(bank, ticker, trade_amount, direction, env, agent, sim):
    """
    Combined CCP check using both RL prediction and simulation.
    
    Returns: (decision, confidence, details)
    """
    # 1. Get RL prediction (fast, learned patterns)
    from model import CCPRiskGateway
    ccp = CCPRiskGateway(env, agent)
    rl_decision, rl_loss, _ = ccp.check_trade(bank, ticker, trade_amount, direction)
    
    # 2. Run Monte Carlo simulation (accurate, slow)
    sim_losses = []
    for _ in range(100):  # 100 quick scenarios
        if direction == 'BUY':
            # Simulate if this trade triggers bank stress
            shock_pct = (trade_amount / sim.bank_states[bank]['Total_Assets']) * 100
            result = sim.propagate_devaluation(bank, shock_pct * 0.3)  # Partial exposure
            sim_losses.append(result['total_asset_loss'])
    
    avg_sim_loss = sum(sim_losses) / len(sim_losses)
    
    # 3. Combined decision (ensemble)
    combined_loss = (rl_loss + avg_sim_loss) / 2
    
    return {
        'rl_prediction': rl_loss,
        'simulation_avg': avg_sim_loss,
        'combined_loss': combined_loss,
        'recommendation': rl_decision
    }


# ============================================================
# IMPROVEMENT 5: Stress Testing Framework
# ============================================================

def run_stress_tests(sim, scenarios=None):
    """
    Run standardized stress tests across the banking system.
    """
    if scenarios is None:
        scenarios = [
            # (trigger_bank, shock_pct, name)
            ('JPM', 50, 'GSIB Failure'),
            ('BAC', 50, 'Second Largest Failure'),
            ('GS', 70, 'Investment Bank Crisis'),
            (None, 15, 'System-Wide 15% Shock'),  # All banks hit
        ]
    
    results = []
    for trigger, shock, name in scenarios:
        if trigger is None:
            # System-wide shock
            total_failed = 0
            total_loss = 0
            for bank in sim.graph:
                r = sim.propagate_devaluation(bank, shock)
                total_failed += r['num_failed_banks']
                total_loss += r['total_asset_loss']
            results.append({
                'scenario': name,
                'avg_failures': total_failed / len(sim.graph),
                'total_loss': total_loss
            })
        else:
            r = sim.propagate_devaluation(trigger, shock)
            results.append({
                'scenario': name,
                'failures': r['num_failed_banks'],
                'loss': r['total_asset_loss'],
                'collapsed': r['system_collapsed']
            })
    
    return results


# ============================================================
# IMPROVEMENT 6: Visualization & Reporting
# ============================================================

def generate_network_visualization(graph, failed_banks=None):
    """
    Generate a network visualization (requires matplotlib/networkx).
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        G = nx.DiGraph()
        for bank, data in graph.items():
            G.add_node(bank)
            for neighbor in data['neighbors']:
                G.add_edge(bank, neighbor)
        
        # Color nodes by status
        colors = []
        for node in G.nodes():
            if failed_banks and node in failed_banks:
                colors.append('red')
            else:
                colors.append('green')
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(G, pos, node_color=colors, with_labels=True, 
                node_size=500, font_size=8, arrows=True)
        plt.title("Banking Network (Red = Failed)")
        plt.savefig('network_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved network visualization to network_visualization.png")
        
    except ImportError:
        print("Install matplotlib and networkx for visualization")


if __name__ == '__main__':
    print("Improvements module loaded. Import specific functions as needed.")
    print("Key improvements:")
    print("  1. load_interbank_from_matrix() - Use real interbank data")
    print("  2. simulate_stock_crash() - Stock-induced bank shocks")
    print("  3. EnhancedContagion - Multi-channel contagion")
    print("  4. integrated_ccp_check() - Combine RL + simulation")
    print("  5. run_stress_tests() - Standardized stress testing")
    print("  6. generate_network_visualization() - Visual output")
