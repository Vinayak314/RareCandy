import random
import csv
from collections import defaultdict


def load_stock_prices(csv_path, num_stocks=10):
    """
    Load stock data from CSV and pick `num_stocks` random unique tickers.
    Returns a dict {ticker: latest_close_price} for the selected stocks,
    and the full time-series data for those tickers.

    Args:
        csv_path: Path to stocks_data_long.csv
        num_stocks: Number of unique stocks to select (default 10)

    Returns:
        selected_prices: dict {ticker: latest_close_price}
        selected_timeseries: dict {ticker: [{date, open, high, low, close, volume}, ...]}
    """
    # First pass: collect all unique tickers and their time-series
    all_data = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row['Ticker']
            all_data[ticker].append({
                'Date': row['Date'],
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': float(row['Close']),
                'Volume': float(row['Volume'])
            })

    all_tickers = list(all_data.keys())
    print(f"Total unique tickers in dataset: {len(all_tickers)}")

    # Pick num_stocks random tickers (filter out penny stocks < $5 for realism)
    viable_tickers = [
        t for t in all_tickers
        if all_data[t][-1]['Close'] >= 5.0  # latest price >= $5
    ]
    selected_tickers = random.sample(viable_tickers, min(num_stocks, len(viable_tickers)))

    # Get latest close price for each selected ticker
    selected_prices = {}
    selected_timeseries = {}
    for ticker in selected_tickers:
        ts = sorted(all_data[ticker], key=lambda x: x['Date'])
        selected_prices[ticker] = ts[-1]['Close']  # most recent close
        selected_timeseries[ticker] = ts

    print(f"Selected {len(selected_prices)} stocks: {list(selected_prices.keys())}")
    for t, p in selected_prices.items():
        print(f"  {t}: ${p:.2f}")

    return selected_prices, selected_timeseries


def distribute_shares(bank_attributes, stock_prices):
    """
    Distribute shares of selected stocks among banks such that:
      sum(shares_owned[stock] * price[stock]) == bank's Total_Assets

    Each bank gets a random allocation across the stocks.
    Different banks receive different random weightings.

    Args:
        bank_attributes: dict {bank_name: {Total_Assets: ..., ...}}
        stock_prices: dict {ticker: price_per_share}

    Returns:
        holdings: dict {bank_name: {ticker: num_shares (float)}}
    """
    tickers = list(stock_prices.keys())
    num_stocks = len(tickers)
    holdings = {}

    for bank_name, attrs in bank_attributes.items():
        total_assets = attrs['Total_Assets']  # in $B

        # Generate random weights using exponential distribution for diversity
        raw_weights = [random.expovariate(1.0) for _ in range(num_stocks)]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]  # normalize to sum=1

        # Allocate total_assets according to weights, compute shares
        bank_holdings = {}
        for i, ticker in enumerate(tickers):
            allocation = total_assets * weights[i]  # $B allocated to this stock
            price = stock_prices[ticker]
            # shares = allocation_in_dollars / price_per_share
            # Total_Assets is in $B, so allocation is in $B
            num_shares = (allocation * 1e9) / price  # convert $B to $ then divide by price
            bank_holdings[ticker] = num_shares

        holdings[bank_name] = bank_holdings

    return holdings


def print_holdings_summary(holdings, stock_prices):
    """Print a summary of stock holdings across all banks."""
    tickers = list(stock_prices.keys())

    print(f"\n{'='*90}")
    print(f"STOCK HOLDINGS DISTRIBUTION ({len(tickers)} stocks across {len(holdings)} banks)")
    print(f"{'='*90}")

    # Header
    header = f"{'Bank':<8}" + "".join(f"{t:>12}" for t in tickers) + f"{'Total($B)':>14}"
    print(header)
    print("-" * len(header))

    for bank, bank_holdings in sorted(holdings.items(), key=lambda x: -sum(
        shares * stock_prices[t] for t, shares in x[1].items()
    )):
        row = f"{bank:<8}"
        total_val = 0
        for t in tickers:
            shares = bank_holdings.get(t, 0)
            val = shares * stock_prices[t]
            total_val += val
            # Show shares in millions for readability
            row += f"{shares/1e6:>12.2f}M"
        row += f"{total_val/1e9:>12.2f}B"
        print(row)

    # System totals
    print("-" * len(header))
    total_row = f"{'TOTAL':<8}"
    grand_total = 0
    for t in tickers:
        total_shares = sum(h.get(t, 0) for h in holdings.values())
        total_val = total_shares * stock_prices[t]
        grand_total += total_val
        total_row += f"{total_shares/1e6:>12.2f}M"
    total_row += f"{grand_total/1e9:>12.2f}B"
    print(total_row)


def load_bank_attributes(csv_path):
    """Load bank attributes from CSV file."""
    banks = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bank_name = row['Bank']
            attributes = {
                'Total_Assets': float(row['Total_Assets']),
                'Equity': float(row['Equity']),
                'Total_Liabilities': float(row['Total_Liabilities']),
                'HQLA': float(row['HQLA']),
                'Net_Outflows_30d': float(row['Net_Outflows_30d']),
                'Est_CDS_Spread': float(row['Est_CDS_Spread']),
                'Stock_Volatility': float(row['Stock_Volatility']),
                'Interbank_Assets': float(row['Interbank_Assets']),
                'Interbank_Liabilities': float(row['Interbank_Liabilities']),
                'LR': float(row.get('LR', row['Equity']) if row.get('LR') else float(row['Equity']) / float(row['Total_Assets'])),
                'LCR': float(row.get('LCR', 1.0) if row.get('LCR') else float(row['HQLA']) / float(row['Net_Outflows_30d']))
            }
            banks[bank_name] = attributes
    return banks


def generate_random_graph_with_sccs(bank_attributes, num_sccs=3, prob_intra=0.5, prob_inter=0.1):
    """Generate graph with bank attributes and SCCs."""
    bank_list = list(bank_attributes.keys())
    num_nodes = len(bank_list)
    
    graph = {}
    for bank in bank_list:
        graph[bank] = {
            'neighbors': [],
            'attributes': bank_attributes[bank].copy(),
            'holdings': {}  # will be populated with stock holdings
        }
    
    scc_sizes = [num_nodes // num_sccs] * num_sccs
    for i in range(num_nodes % num_sccs):
        scc_sizes[i] += 1
    
    node_to_scc = {}
    node_idx = 0
    
    for scc_idx, size in enumerate(scc_sizes):
        scc_banks = bank_list[node_idx:node_idx + size]
        
        for i in range(len(scc_banks)):
            u = scc_banks[i]
            v = scc_banks[(i + 1) % len(scc_banks)]
            graph[u]['neighbors'].append(v)
        
        for u in scc_banks:
            for v in scc_banks:
                if u != v and random.random() < prob_intra:
                    if v not in graph[u]['neighbors']:
                        graph[u]['neighbors'].append(v)
        
        for bank in scc_banks:
            node_to_scc[bank] = scc_idx
        
        node_idx += size
    
    for u in bank_list:
        for v in bank_list:
            if node_to_scc[u] < node_to_scc[v] and random.random() < prob_inter:
                if v not in graph[u]['neighbors']:
                    graph[u]['neighbors'].append(v)
    
    return graph


class BankingNetworkContagion:
    """Simulates contagion/cascade effects in a banking network."""
    
    def __init__(self, graph):
        """
        Args:
            graph: Dict of {bank_name: {neighbors: [...], attributes: {...}}}
        """
        self.graph = graph
        self.bank_states = {}  # Track current health of each bank
        self.failed_banks = set()
        self.history = []
        self.initialize_states()
    
    def initialize_states(self):
        """Initialize each bank's state as a copy of its attributes."""
        for bank in self.graph:
            self.bank_states[bank] = self.graph[bank]['attributes'].copy()
    
    def get_bank_health(self, bank):
        """
        Calculate bank health score (0-100).
        Uses capital adequacy, liquidity, and market risk indicators.
        
        Components (each contributes a portion of the 0-100 scale):
          - Capital adequacy (equity/assets ratio):  0-40 points
          - Liquidity (LCR):                         0-30 points
          - Market risk (CDS + volatility penalty):   0 to -20 points
        """
        if bank in self.failed_banks:
            return 0.0
        
        attrs = self.bank_states[bank]
        total_assets = attrs['Total_Assets']
        equity = attrs['Equity']
        
        if total_assets <= 0:
            return 0.0
        
        # 1. Capital adequacy: equity/assets ratio (typical range 5-15%)
        #    Map 0% → 0 pts, 8% → 30 pts, 15%+ → 40 pts
        equity_ratio = equity / total_assets
        capital_score = min(40.0, (equity_ratio / 0.15) * 40.0)
        
        # 2. Liquidity: LCR (regulatory minimum is 1.0)
        #    Map 0 → 0 pts, 1.0 → 20 pts, 1.5+ → 30 pts
        lcr = attrs['LCR']
        liquidity_score = min(30.0, (lcr / 1.5) * 30.0)
        
        # 3. Market risk penalty from CDS spread and stock volatility
        #    Higher CDS spread = higher default risk
        #    Normalize CDS: typical range 50-400 bps → penalty 0-15
        cds_penalty = min(15.0, (attrs['Est_CDS_Spread'] / 400.0) * 15.0)
        #    Normalize volatility: typical range 0.15-0.50 → penalty 0-10
        vol_penalty = min(10.0, (attrs['Stock_Volatility'] / 0.50) * 10.0)
        
        health = capital_score + liquidity_score - cds_penalty - vol_penalty
        return max(0.0, min(100.0, health))
    
    def mark_bank_failed(self, bank):
        """Mark a bank as failed."""
        if bank not in self.failed_banks:
            self.failed_banks.add(bank)
    
    def propagate_devaluation(self, initial_bank, devaluation_shock, max_rounds=100, failure_threshold=20.0):
        """
        Simulate contagion from initial_bank devaluation.
        
        Args:
            initial_bank: Name of bank experiencing initial shock
            devaluation_shock: Percentage loss to initial bank's assets (0-100)
            max_rounds: Maximum rounds of contagion
            failure_threshold: Health score below which bank fails
        
        Returns:
            dict: Simulation results
        """
        self.initialize_states()
        self.failed_banks = set()
        self.history = []
        
        # Apply initial shock
        initial_health = self.get_bank_health(initial_bank)
        shock_amount = self.bank_states[initial_bank]['Total_Assets'] * (devaluation_shock / 100.0)
        self.bank_states[initial_bank]['Total_Assets'] -= shock_amount
        self.bank_states[initial_bank]['Equity'] -= shock_amount
        
        if self.get_bank_health(initial_bank) < failure_threshold:
            self.mark_bank_failed(initial_bank)
        
        round_num = 0
        # Only propagate contagion if the initial bank actually failed
        newly_failed = {initial_bank} if initial_bank in self.failed_banks else set()
        
        while newly_failed and round_num < max_rounds:
            round_num += 1
            previously_failed = newly_failed.copy()
            newly_failed = set()
            
            # Contagion dampening: each round transmits less shock (absorbs/hedges)
            round_dampening = 0.7 ** round_num  # 30% attenuation per round
            
            # For each failed bank, propagate impact to neighbors
            for failed_bank in previously_failed:
                impact = self._compute_bank_impact(failed_bank)
                
                # Impact creditors (banks that this bank owes to)
                for creditor in self.graph[failed_bank]['neighbors']:
                    if creditor not in self.failed_banks:
                        exposure = self._get_exposure(failed_bank, creditor)
                        loss = impact * exposure * round_dampening
                        
                        # Sanity check: loss cannot exceed a fraction of creditor's assets
                        creditor_assets = self.bank_states[creditor]['Total_Assets']
                        loss = min(loss, creditor_assets * 0.15)  # Cap at 15% per event
                        
                        self.bank_states[creditor]['Total_Assets'] -= loss
                        self.bank_states[creditor]['Equity'] -= loss
                        
                        if self.get_bank_health(creditor) < failure_threshold:
                            self.mark_bank_failed(creditor)
                            newly_failed.add(creditor)
                
                # Impact borrowers (banks that owe to this bank) — weaker channel
                for borrower in self.graph:
                    if failed_bank in self.graph[borrower]['neighbors']:
                        if borrower not in self.failed_banks:
                            exposure = self._get_exposure(borrower, failed_bank)
                            loss = impact * exposure * 0.3 * round_dampening  # Weaker: funding disruption
                            
                            borrower_assets = self.bank_states[borrower]['Total_Assets']
                            loss = min(loss, borrower_assets * 0.10)  # Cap at 10%
                            
                            self.bank_states[borrower]['Total_Assets'] -= loss
                            self.bank_states[borrower]['Equity'] -= loss
                            
                            if self.get_bank_health(borrower) < failure_threshold:
                                self.mark_bank_failed(borrower)
                                newly_failed.add(borrower)
            
            # Record state
            round_state = {
                'round': round_num,
                'failed_banks': len(self.failed_banks),
                'newly_failed': list(newly_failed),
                'total_asset_loss': self._compute_total_asset_loss()
            }
            self.history.append(round_state)
        
        return self._generate_report(initial_bank, devaluation_shock, failure_threshold)
    
    def _compute_bank_impact(self, bank):
        """Compute the loss magnitude from a failed bank, scaled by bank size."""
        attrs = self.bank_states[bank]
        original_assets = self.graph[bank]['attributes']['Total_Assets']
        # What fraction of assets were lost
        asset_loss_ratio = max(0, 1 - (attrs['Total_Assets'] / original_assets))
        # Impact is the actual dollar losses this bank can transmit through interbank channels
        # Capped by the bank's original interbank liabilities (what it actually owes)
        interbank_liab = self.graph[bank]['attributes']['Interbank_Liabilities']
        return asset_loss_ratio * interbank_liab
    
    def _get_exposure(self, from_bank, to_bank):
        """
        Get the fraction of 'impact' that to_bank actually absorbs from from_bank's failure.
        
        Key principle: A large bank's exposure to a small bank is limited by the small bank's
        size. A small bank cannot cause losses larger than its own interbank footprint.
        """
        # Use original (pre-shock) sizes for stable exposure calculation
        from_original = self.graph[from_bank]['attributes']['Total_Assets']
        to_original = self.graph[to_bank]['attributes']['Total_Assets']
        num_neighbors = max(1, len(self.graph[from_bank]['neighbors']))
        
        # The failing bank's interbank liabilities are spread across its creditors
        # Each creditor's share is roughly 1/num_neighbors of total interbank liabilities
        share_of_liabilities = 1.0 / num_neighbors
        
        # Scale by relative size: a large bank is less affected by a small bank's failure
        # A bank cannot lose more than the smaller bank's proportional size relative to it
        size_ratio = min(1.0, from_original / (to_original + 1e-6))
        
        # Combined exposure factor, capped conservatively
        exposure = share_of_liabilities * size_ratio
        return min(0.05, exposure)  # Hard cap: no single counterparty causes >5% of impact
    
    def _compute_total_asset_loss(self):
        """Compute total system asset loss."""
        total_loss = 0
        for bank in self.bank_states:
            original = self.graph[bank]['attributes']['Total_Assets']
            current = self.bank_states[bank]['Total_Assets']
            total_loss += max(0, original - current)
        return total_loss
    
    def _generate_report(self, initial_bank, shock, threshold):
        """Generate comprehensive contagion report."""
        num_failed = len(self.failed_banks)
        num_total = len(self.graph)
        collapse_ratio = num_failed / num_total
        total_loss = self._compute_total_asset_loss()
        
        survived = set(self.graph.keys()) - self.failed_banks
        avg_survivor_health = sum(self.get_bank_health(b) for b in survived) / len(survived) if survived else 0
        
        return {
            'initial_bank': initial_bank,
            'initial_shock_pct': shock,
            'failure_threshold': threshold,
            'num_failed_banks': num_failed,
            'total_banks': num_total,
            'collapse_ratio': collapse_ratio,
            'total_asset_loss': total_loss,
            'avg_survivor_health': avg_survivor_health,
            'rounds_until_stability': len(self.history),
            'failed_banks': list(self.failed_banks),
            'contagion_history': self.history,
            'system_collapsed': collapse_ratio > 0.5
        }


def print_report(result):
    """Pretty-print contagion simulation results."""
    print(f"Initial Bank: {result['initial_bank']}")
    print(f"Initial Shock: {result['initial_shock_pct']}%")
    print(f"\nResults:")
    print(f"  Banks Failed: {result['num_failed_banks']}/{result['total_banks']}")
    print(f"  Collapse Ratio: {result['collapse_ratio']:.2%}")
    print(f"  System Collapsed: {'YES' if result['system_collapsed'] else 'NO'}")
    print(f"  Total Asset Loss: ${result['total_asset_loss']:.2f}B")
    print(f"  Avg Survivor Health: {result['avg_survivor_health']:.2f}")
    print(f"  Rounds Until Stability: {result['rounds_until_stability']}")
    
    if result['failed_banks']:
        print(f"\n  Failed Banks: {', '.join(result['failed_banks'][:10])}")
        if len(result['failed_banks']) > 10:
            print(f"    ... and {len(result['failed_banks']) - 10} more")
    
    print(f"\n  Contagion Progression:")
    for state in result['contagion_history']:
        print(f"    Round {state['round']}: {state['failed_banks']} failed " +
              f"(+{len(state['newly_failed'])} new)")


# Example usage
if __name__ == '__main__':
    # Load banks and generate network (fixed paths for running from dataset folder)
    bank_attrs = load_bank_attributes('us_banks_top50_nodes_final.csv')
    G = generate_random_graph_with_sccs(bank_attrs, num_sccs=4, prob_intra=0.4, prob_inter=0.05)

    # --- Load stocks and distribute shares among banks ---
    stock_prices, stock_timeseries = load_stock_prices(
        'stocks_data_long.csv', num_stocks=10
    )
    holdings = distribute_shares(bank_attrs, stock_prices)

    # Attach holdings to the graph nodes
    for bank in G:
        if bank in holdings:
            G[bank]['holdings'] = holdings[bank]

    # Print holdings summary
    print_holdings_summary(holdings, stock_prices)
    print()

    # Verify: check that holdings value matches Total_Assets for a few banks
    print("Verification (holdings value vs Total_Assets):")
    for bank in ['JPM', 'BAC', 'IBOC', 'MS']:
        if bank in holdings:
            val = sum(holdings[bank][t] * stock_prices[t] for t in stock_prices) / 1e9
            print(f"  {bank}: Holdings=${val:.2f}B, Total_Assets=${bank_attrs[bank]['Total_Assets']:.2f}B")
    print()

    # Run contagion simulation
    contagion = BankingNetworkContagion(G)
    
    # Debug: Print initial health scores to verify they're reasonable
    print("Initial Bank Health Scores (sample):")
    sample_banks = ['JPM', 'BAC', 'GS', 'MS', 'IBOC', 'WFC']
    for b in sample_banks:
        if b in contagion.graph:
            print(f"  {b}: {contagion.get_bank_health(b):.1f}/100")
    print()
    
    # Scenario 1: Small shock (10%)
    print("=" * 70)
    print("SCENARIO 1: Small Devaluation Shock (10%) to IBOC")
    print("=" * 70)
    result1 = contagion.propagate_devaluation('IBOC', devaluation_shock=10, failure_threshold=20)
    print_report(result1)
    
    # Scenario 2: Large shock (50%)
    print("\n" + "=" * 70)
    print("SCENARIO 2: Large Devaluation Shock (50%) to IBOC")
    print("=" * 70)
    result2 = contagion.propagate_devaluation('IBOC', devaluation_shock=50, failure_threshold=20)
    print_report(result2)
    
    # Scenario 3: Shock to large bank
    print("\n" + "=" * 70)
    print("SCENARIO 3: Shock to JPM (80% shock)")
    print("=" * 70)
    result3 = contagion.propagate_devaluation('JPM', devaluation_shock=80, failure_threshold=20)
    print_report(result3)