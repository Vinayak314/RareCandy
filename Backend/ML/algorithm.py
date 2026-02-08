import random
import csv
from collections import defaultdict


def generate_margin_requirements(bank_attributes, margin_ratio_range=(0.02, 0.08)):
    """
    Generate margin requirements for each bank.
    
    Margin requirements are funds that must be held as collateral and cannot be used
    for normal operations. However, during devaluation events, this margin can be
    released to absorb losses.
    
    Args:
        bank_attributes: dict {bank_name: {Total_Assets: ..., ...}}
        margin_ratio_range: (min, max) ratio of Total_Assets to hold as margin
    
    Returns:
        margin_requirements: dict {bank_name: margin_amount_in_billions}
    """
    margin_requirements = {}
    
    for bank_name, attrs in bank_attributes.items():
        total_assets = attrs['Total_Assets']
        # Random margin ratio within range
        margin_ratio = random.uniform(margin_ratio_range[0], margin_ratio_range[1])
        margin_amount = total_assets * margin_ratio
        margin_requirements[bank_name] = margin_amount
    
    return margin_requirements


def print_margin_summary(margin_requirements, bank_attributes):
    """Print summary of margin requirements."""
    print(f"\n{'='*60}")
    print(f"MARGIN REQUIREMENTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Bank':<10}{'Total Assets':>15}{'Margin':>15}{'Margin %':>12}")
    print("-" * 52)
    
    total_margin = 0
    for bank, margin in sorted(margin_requirements.items(), key=lambda x: -x[1])[:15]:
        assets = bank_attributes[bank]['Total_Assets']
        pct = (margin / assets) * 100
        total_margin += margin
        print(f"{bank:<10}${assets:>13.2f}B${margin:>13.2f}B{pct:>10.1f}%")
    
    if len(margin_requirements) > 15:
        print(f"... and {len(margin_requirements) - 15} more banks")
    
    print("-" * 52)
    print(f"{'TOTAL':<10}{'':<15}${total_margin:>13.2f}B")
    print()


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

    Small banks have concentrated portfolios (own 1-3 stocks, one dominates).
    Large banks have more diversified portfolios (own 5-10 stocks).
    Banks don't own all stocks - they're only affected by devaluation of stocks they hold.

    Args:
        bank_attributes: dict {bank_name: {Total_Assets: ..., ...}}
        stock_prices: dict {ticker: price_per_share}

    Returns:
        holdings: dict {bank_name: {ticker: num_shares (float)}}
    """
    import numpy as np
    
    tickers = list(stock_prices.keys())
    num_stocks = len(tickers)
    holdings = {}
    
    # Determine bank size distribution for relative sizing
    all_assets = [attrs['Total_Assets'] for attrs in bank_attributes.values()]
    max_assets = max(all_assets)
    min_assets = min(all_assets)
    asset_range = max_assets - min_assets if max_assets > min_assets else 1

    for bank_name, attrs in bank_attributes.items():
        total_assets = attrs['Total_Assets']  # in $B
        
        # Normalize bank size: 0 = smallest, 1 = largest
        size_percentile = (total_assets - min_assets) / asset_range
        
        # Determine how many stocks this bank owns based on size
        # Small banks: 1-3 stocks, Medium: 3-6 stocks, Large: 5-10 stocks
        if size_percentile < 0.3:  # Small banks
            num_owned = random.randint(1, min(3, num_stocks))
        elif size_percentile < 0.6:  # Medium banks
            num_owned = random.randint(3, min(6, num_stocks))
        else:  # Large banks
            num_owned = random.randint(max(5, num_stocks // 2), num_stocks)
        
        # Select which stocks this bank owns
        owned_tickers = random.sample(tickers, num_owned)
        
        # Concentration parameter for Dirichlet distribution among owned stocks
        # Small banks: very concentrated (one stock dominates)
        # Large banks: more diversified
        base_alpha = 0.1 + size_percentile * 1.9  # Range: 0.1 to 2.0
        
        # For small banks, pick a "primary" stock that gets extra weight
        if size_percentile < 0.3:  # Small banks
            # Create asymmetric alphas: one stock gets much higher alpha
            primary_idx = 0  # First owned stock is primary
            alphas = [base_alpha * 0.3 for _ in range(num_owned)]  # Low base for all
            alphas[primary_idx] = base_alpha * 5.0  # Primary stock gets 5x weight
        elif size_percentile < 0.6:  # Medium banks
            # Moderately concentrated: 1-2 stocks get more weight
            primary_idx = 0
            alphas = [base_alpha for _ in range(num_owned)]
            alphas[primary_idx] = base_alpha * 2.0  # Primary gets 2x
        else:  # Large banks
            # More even distribution
            alphas = [base_alpha for _ in range(num_owned)]
        
        # Generate weights using Dirichlet distribution for owned stocks only
        weights = np.random.dirichlet(alphas)

        # Allocate total_assets according to weights, compute shares
        bank_holdings = {}
        for i, ticker in enumerate(owned_tickers):
            allocation = total_assets * weights[i]  # $B allocated to this stock
            price = stock_prices[ticker]
            # shares = allocation_in_dollars / price_per_share
            # Total_Assets is in $B, so allocation is in $B
            num_shares = (allocation * 1e9) / price  # convert $B to $ then divide by price
            bank_holdings[ticker] = num_shares
        
        # Stocks not owned have 0 shares (not included in bank_holdings)
        holdings[bank_name] = bank_holdings

    return holdings


def print_holdings_summary(holdings, stock_prices):
    """Print a summary of stock holdings across all banks."""
    tickers = list(stock_prices.keys())

    print(f"\n{'='*90}")
    print(f"STOCK HOLDINGS DISTRIBUTION ({len(tickers)} stocks across {len(holdings)} banks)")
    print(f"{'='*90}")

    # Header
    header = f"{'Bank':<8}" + "".join(f"{t:>12}" for t in tickers) + f"{'Total($B)':>14}" + f"{'#Stocks':>8}"
    print(header)
    print("-" * len(header))

    for bank, bank_holdings in sorted(holdings.items(), key=lambda x: -sum(
        shares * stock_prices[t] for t, shares in x[1].items()
    )):
        row = f"{bank:<8}"
        total_val = 0
        num_owned = 0
        for t in tickers:
            shares = bank_holdings.get(t, 0)
            val = shares * stock_prices[t]
            total_val += val
            if shares > 0:
                num_owned += 1
                # Show shares in millions for readability
                row += f"{shares/1e6:>12.2f}M"
            else:
                row += f"{'--':>12}"
        row += f"{total_val/1e9:>12.2f}B"
        row += f"{num_owned:>8}"
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
                'LR': float(row['LR']),
                'LCR': float(row['LCR'])
            }
            banks[bank_name] = attributes
    return banks


def generate_random_graph_with_sccs(bank_attributes, num_sccs=5, prob_intra=0.5, prob_inter=0.1):
    """
    Generate graph with tiered core-periphery structure.
    
    Large banks (top 20% by Total_Assets) form a strongly connected component (SCC).
    Small banks connect primarily to large banks with fewer inter-small connections.
    
    Args:
        bank_attributes: Dict of bank attributes with Total_Assets
        num_sccs: Number of additional SCCs among small banks (default 5)
        prob_intra: Edge probability within small bank SCCs (default 0.5)
        prob_inter: Edge probability between small bank SCCs (default 0.1)
    
    Returns:
        graph: Dict with bank connections and attributes
    """
    bank_list = list(bank_attributes.keys())
    num_nodes = len(bank_list)
    
    # Initialize graph structure
    graph = {}
    for bank in bank_list:
        graph[bank] = {
            'neighbors': [],
            'attributes': bank_attributes[bank].copy(),
            'holdings': {}
        }
    
    # Sort banks by Total_Assets (descending) to identify large vs small
    sorted_banks = sorted(bank_list, key=lambda b: bank_attributes[b]['Total_Assets'], reverse=True)
    
    # Top 20% are large banks, rest are small banks
    large_bank_ratio = 0.2
    num_large = max(2, int(num_nodes * large_bank_ratio))
    large_banks = sorted_banks[:num_large]
    small_banks = sorted_banks[num_large:]
    large_bank_set = set(large_banks)
    
    print(f"Network Structure: {num_large} large banks, {len(small_banks)} small banks")
    print(f"Large banks: {large_banks}")
    
    # =========================================================================
    # 1. CREATE SCC AMONG LARGE BANKS (core)
    # =========================================================================
    # Create a directed cycle to guarantee SCC
    for i in range(len(large_banks)):
        u = large_banks[i]
        v = large_banks[(i + 1) % len(large_banks)]
        if v not in graph[u]['neighbors']:
            graph[u]['neighbors'].append(v)
    
    # Add additional dense connections among large banks (80% probability)
    large_to_large_density = 0.8
    for u in large_banks:
        for v in large_banks:
            if u != v and v not in graph[u]['neighbors']:
                if random.random() < large_to_large_density:
                    graph[u]['neighbors'].append(v)
    
    # =========================================================================
    # 2. CREATE LARGE-TO-SMALL EDGES (core-periphery connections)
    # =========================================================================
    large_to_small_density = 0.4  # High: large banks lend to many small banks
    small_to_large_density = 0.2  # Lower: small banks have fewer connections to large
    
    for large in large_banks:
        for small in small_banks:
            # Large -> Small (lending/exposure)
            if random.random() < large_to_small_density:
                if small not in graph[large]['neighbors']:
                    graph[large]['neighbors'].append(small)
            # Small -> Large (borrowing/deposits)
            if random.random() < small_to_large_density:
                if large not in graph[small]['neighbors']:
                    graph[small]['neighbors'].append(large)
    
    # Ensure every small bank has at least one connection to a large bank
    for small in small_banks:
        has_large_in = any(small in graph[large]['neighbors'] for large in large_banks)
        has_large_out = any(large in graph[small]['neighbors'] for large in large_banks)
        
        if not has_large_in and not has_large_out:
            # Connect to a random large bank
            large = random.choice(large_banks)
            if random.random() < 0.5:
                graph[large]['neighbors'].append(small)
            else:
                graph[small]['neighbors'].append(large)
    
    # =========================================================================
    # 3. CREATE SPARSE SMALL-TO-SMALL EDGES (periphery)
    # =========================================================================
    small_to_small_density = 0.05  # Very sparse connections among small banks
    
    # Optionally create small SCCs among small banks for more realistic structure
    if num_sccs > 0 and len(small_banks) > num_sccs:
        scc_sizes = [len(small_banks) // num_sccs] * num_sccs
        for i in range(len(small_banks) % num_sccs):
            scc_sizes[i] += 1
        
        node_idx = 0
        node_to_scc = {}
        
        for scc_idx, size in enumerate(scc_sizes):
            scc_banks = small_banks[node_idx:node_idx + size]
            
            # Create cycle within small SCC (weaker guarantee)
            if len(scc_banks) > 1 and random.random() < 0.3:  # Only 30% chance of forming mini-SCC
                for i in range(len(scc_banks)):
                    u = scc_banks[i]
                    v = scc_banks[(i + 1) % len(scc_banks)]
                    if v not in graph[u]['neighbors']:
                        graph[u]['neighbors'].append(v)
            
            # Sparse intra-SCC edges among small banks
            for u in scc_banks:
                for v in scc_banks:
                    if u != v and random.random() < small_to_small_density:
                        if v not in graph[u]['neighbors']:
                            graph[u]['neighbors'].append(v)
            
            for bank in scc_banks:
                node_to_scc[bank] = scc_idx
            
            node_idx += size
        
        # Very sparse inter-small-SCC edges
        inter_small_density = 0.02
        for u in small_banks:
            for v in small_banks:
                if node_to_scc.get(u, -1) != node_to_scc.get(v, -1) and random.random() < inter_small_density:
                    if v not in graph[u]['neighbors']:
                        graph[u]['neighbors'].append(v)
    else:
        # Simple sparse connections among small banks
        for u in small_banks:
            for v in small_banks:
                if u != v and random.random() < small_to_small_density:
                    if v not in graph[u]['neighbors']:
                        graph[u]['neighbors'].append(v)
    
    # Print edge statistics
    total_edges = sum(len(graph[b]['neighbors']) for b in graph)
    large_to_large = sum(
        1 for u in large_banks for v in graph[u]['neighbors'] if v in large_bank_set
    )
    large_to_small = sum(
        1 for u in large_banks for v in graph[u]['neighbors'] if v not in large_bank_set
    )
    small_to_large = sum(
        1 for u in small_banks for v in graph[u]['neighbors'] if v in large_bank_set
    )
    small_to_small = sum(
        1 for u in small_banks for v in graph[u]['neighbors'] if v not in large_bank_set
    )
    
    print(f"\nEdge Statistics:")
    print(f"  Total edges: {total_edges}")
    print(f"  Large->Large (SCC core): {large_to_large}")
    print(f"  Large->Small: {large_to_small}")
    print(f"  Small->Large: {small_to_large}")
    print(f"  Small->Small: {small_to_small}")
    
    return graph


class BankingNetworkContagion:
    """Simulates contagion/cascade effects in a banking network."""
    
    def __init__(self, graph, stock_prices=None, margin_requirements=None):
        """
        Args:
            graph: Dict of {bank_name: {neighbors: [...], attributes: {...}, holdings: {...}}}
            stock_prices: Dict of {ticker: price} for stock devaluation scenarios
            margin_requirements: Dict of {bank_name: margin_amount} - locked collateral
        """
        self.graph = graph
        self.stock_prices = stock_prices.copy() if stock_prices else {}
        self.current_stock_prices = stock_prices.copy() if stock_prices else {}
        self.margin_requirements = margin_requirements.copy() if margin_requirements else {}
        self.bank_states = {}  # Track current health of each bank
        self.margin_states = {}  # Track remaining margin for each bank
        self.failed_banks = set()
        self.history = []
        self.initialize_states()
    
    def initialize_states(self):
        """Initialize each bank's state as a copy of its attributes."""
        for bank in self.graph:
            self.bank_states[bank] = self.graph[bank]['attributes'].copy()
            # Initialize margin: locked amount that can be used as buffer during stress
            self.margin_states[bank] = self.margin_requirements.get(bank, 0)
            # Reduce available HQLA by margin (margin is locked)
            if self.margin_states[bank] > 0:
                self.bank_states[bank]['HQLA'] = max(0, self.bank_states[bank]['HQLA'] - self.margin_states[bank])
        # Reset stock prices to original
        self.current_stock_prices = self.stock_prices.copy()
    
    def _use_margin_buffer(self, bank, loss_amount):
        """
        Use margin as a buffer to absorb losses during devaluation.
        
        Args:
            bank: Bank name
            loss_amount: Amount of loss to absorb
        
        Returns:
            actual_loss: Loss after margin absorption (may be less than loss_amount)
            margin_used: Amount of margin consumed
        """
        available_margin = self.margin_states.get(bank, 0)
        
        if available_margin <= 0:
            return loss_amount, 0
        
        # Margin can absorb up to 50% of the loss (partial protection)
        max_absorption = loss_amount * 0.5
        margin_used = min(available_margin, max_absorption)
        
        # Reduce available margin
        self.margin_states[bank] -= margin_used
        
        # Actual loss is reduced by margin used
        actual_loss = loss_amount - margin_used
        
        return actual_loss, margin_used
    
    def get_bank_health(self, bank):
        """
        Calculate bank health score (0-100).
        Based on equity-to-assets ratio and liquidity.
        """
        if bank in self.failed_banks:
            return 0.0
        
        attrs = self.bank_states[bank]
        total_assets = attrs['Total_Assets']
        equity = attrs['Equity']
        
        if total_assets <= 0:
            return 0.0
        
        # Health inversely correlated with CDS spread and volatility
        # Positive correlation with equity and LCR
        equity_ratio = (equity / total_assets) * 100
        solvency_score = min(equity_ratio/20.0, 1.0) * 100  # Cap at 100 points for equity ratio
        lcr_score = min(attrs['LCR'] / 2.0, 1.0) * 100  # Cap at 100 points for LCR
        cds_val = attrs['Est_CDS_Spread']
        cds_score = max(0, 1.0 - (cds_val / 500.0)) * 100
        vol_val = attrs['Stock_Volatility']
        vol_score = max(0, 1.0 - (vol_val / 0.8)) * 100    

        health = (
            (solvency_score * 0.40) +
            (lcr_score      * 0.30) +
            (cds_score      * 0.15) +
            (vol_score      * 0.15)
        )
        health = round(health, 2)

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
                        
                        # Use margin buffer for contagion losses
                        actual_loss, _ = self._use_margin_buffer(creditor, loss)
                        
                        self.bank_states[creditor]['Total_Assets'] -= actual_loss
                        self.bank_states[creditor]['Equity'] -= actual_loss
                        
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
                            
                            # Use margin buffer for contagion losses
                            actual_loss, _ = self._use_margin_buffer(borrower, loss)
                            
                            self.bank_states[borrower]['Total_Assets'] -= actual_loss
                            self.bank_states[borrower]['Equity'] -= actual_loss
                            
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
    
    def propagate_stock_devaluation(self, stock_ticker, devaluation_pct, max_rounds=100, failure_threshold=20.0):
        """
        Simulate contagion from stock price devaluation.
        
        When a stock is devalued, all banks holding that stock see their assets reduced
        proportionally to their holdings, which can trigger failures and contagion.
        
        Args:
            stock_ticker: Ticker of the stock to devalue
            devaluation_pct: Percentage drop in stock price (0-100)
            max_rounds: Maximum rounds of contagion
            failure_threshold: Health score below which bank fails
        
        Returns:
            dict: Simulation results
        """
        self.initialize_states()
        self.failed_banks = set()
        self.history = []
        
        if stock_ticker not in self.stock_prices:
            raise ValueError(f"Stock {stock_ticker} not found in stock_prices")
        
        # Calculate new stock price
        old_price = self.stock_prices[stock_ticker]
        new_price = old_price * (1 - devaluation_pct / 100.0)
        self.current_stock_prices[stock_ticker] = new_price
        
        print(f"\n  Stock {stock_ticker} devalued: ${old_price:.2f} -> ${new_price:.2f} ({-devaluation_pct:.1f}%)")
        
        # Apply losses to all banks holding this stock
        initial_failures = set()
        banks_affected = []
        
        for bank in self.graph:
            holdings = self.graph[bank].get('holdings', {})
            if stock_ticker in holdings:
                shares = holdings[stock_ticker]
                loss = shares * (old_price - new_price)  # Loss in dollars
                loss_billions = loss / 1e9  # Convert to billions
                
                if loss_billions > 0:
                    # Use margin buffer to reduce loss
                    actual_loss, margin_used = self._use_margin_buffer(bank, loss_billions)
                    banks_affected.append((bank, actual_loss, margin_used))
                    self.bank_states[bank]['Total_Assets'] -= actual_loss
                    self.bank_states[bank]['Equity'] -= actual_loss
                    
                    if self.get_bank_health(bank) < failure_threshold:
                        self.mark_bank_failed(bank)
                        initial_failures.add(bank)
        
        # Print affected banks
        if banks_affected:
            banks_affected.sort(key=lambda x: -x[1])
            print(f"  Banks affected by {stock_ticker} devaluation:")
            for bank, loss, margin_used in banks_affected[:5]:
                status = " [FAILED]" if bank in initial_failures else ""
                margin_info = f" (margin buffer: ${margin_used:.2f}B)" if margin_used > 0 else ""
                print(f"    {bank}: -${loss:.2f}B{margin_info}{status}")
            if len(banks_affected) > 5:
                print(f"    ... and {len(banks_affected) - 5} more banks")
        
        # Propagate contagion from initially failed banks
        round_num = 0
        newly_failed = initial_failures.copy()
        
        while newly_failed and round_num < max_rounds:
            round_num += 1
            previously_failed = newly_failed.copy()
            newly_failed = set()
            
            round_dampening = 0.7 ** round_num
            
            for failed_bank in previously_failed:
                impact = self._compute_bank_impact(failed_bank)
                
                # Impact creditors
                for creditor in self.graph[failed_bank]['neighbors']:
                    if creditor not in self.failed_banks:
                        exposure = self._get_exposure(failed_bank, creditor)
                        loss = impact * exposure * round_dampening
                        
                        creditor_assets = self.bank_states[creditor]['Total_Assets']
                        loss = min(loss, creditor_assets * 0.15)
                        
                        # Use margin buffer for contagion losses
                        actual_loss, _ = self._use_margin_buffer(creditor, loss)
                        
                        self.bank_states[creditor]['Total_Assets'] -= actual_loss
                        self.bank_states[creditor]['Equity'] -= actual_loss
                        
                        if self.get_bank_health(creditor) < failure_threshold:
                            self.mark_bank_failed(creditor)
                            newly_failed.add(creditor)
                
                # Impact borrowers
                for borrower in self.graph:
                    if failed_bank in self.graph[borrower]['neighbors']:
                        if borrower not in self.failed_banks:
                            exposure = self._get_exposure(borrower, failed_bank)
                            loss = impact * exposure * 0.3 * round_dampening
                            
                            borrower_assets = self.bank_states[borrower]['Total_Assets']
                            loss = min(loss, borrower_assets * 0.10)
                            
                            # Use margin buffer for contagion losses
                            actual_loss, _ = self._use_margin_buffer(borrower, loss)
                            
                            self.bank_states[borrower]['Total_Assets'] -= actual_loss
                            self.bank_states[borrower]['Equity'] -= actual_loss
                            
                            if self.get_bank_health(borrower) < failure_threshold:
                                self.mark_bank_failed(borrower)
                                newly_failed.add(borrower)
            
            round_state = {
                'round': round_num,
                'failed_banks': len(self.failed_banks),
                'newly_failed': list(newly_failed),
                'total_asset_loss': self._compute_total_asset_loss()
            }
            self.history.append(round_state)
        
        return self._generate_stock_report(stock_ticker, devaluation_pct, failure_threshold, initial_failures)
    
    def propagate_multi_stock_devaluation(self, stock_devaluations, max_rounds=100, failure_threshold=20.0):
        """
        Simulate contagion from multiple stock price devaluations.
        
        When multiple stocks are devalued, all banks holding those stocks see their assets reduced
        proportionally to their holdings, which can trigger failures and contagion.
        
        Args:
            stock_devaluations: Dict of {ticker: devaluation_pct} for stocks to devalue
            max_rounds: Maximum rounds of contagion
            failure_threshold: Health score below which bank fails
        
        Returns:
            dict: Simulation results
        """
        self.initialize_states()
        self.failed_banks = set()
        self.history = []
        
        # Validate all stocks exist
        for stock_ticker in stock_devaluations:
            if stock_ticker not in self.stock_prices:
                raise ValueError(f"Stock {stock_ticker} not found in stock_prices")
        
        print(f"\n  Devaluing {len(stock_devaluations)} stocks:")
        
        # Apply all stock devaluations
        all_banks_affected = {}  # bank -> {loss, margin_used}
        initial_failures = set()
        
        for stock_ticker, devaluation_pct in stock_devaluations.items():
            # Calculate new stock price
            old_price = self.stock_prices[stock_ticker]
            new_price = old_price * (1 - devaluation_pct / 100.0)
            self.current_stock_prices[stock_ticker] = new_price
            
            print(f"    {stock_ticker}: ${old_price:.2f} -> ${new_price:.2f} ({-devaluation_pct:.1f}%)")
            
            # Apply losses to all banks holding this stock
            for bank in self.graph:
                holdings = self.graph[bank].get('holdings', {})
                if stock_ticker in holdings:
                    shares = holdings[stock_ticker]
                    loss = shares * (old_price - new_price)  # Loss in dollars
                    loss_billions = loss / 1e9  # Convert to billions
                    
                    if loss_billions > 0:
                        # Use margin buffer to reduce loss
                        actual_loss, margin_used = self._use_margin_buffer(bank, loss_billions)
                        
                        if bank not in all_banks_affected:
                            all_banks_affected[bank] = {'loss': 0, 'margin_used': 0}
                        all_banks_affected[bank]['loss'] += actual_loss
                        all_banks_affected[bank]['margin_used'] += margin_used
                        
                        self.bank_states[bank]['Total_Assets'] -= actual_loss
                        self.bank_states[bank]['Equity'] -= actual_loss
        
        # Check for initial failures after all devaluations applied
        for bank in all_banks_affected:
            if self.get_bank_health(bank) < failure_threshold:
                self.mark_bank_failed(bank)
                initial_failures.add(bank)
        
        # Print affected banks
        if all_banks_affected:
            sorted_affected = sorted(all_banks_affected.items(), key=lambda x: -x[1]['loss'])
            print(f"  Banks affected by stock devaluations:")
            for bank, data in sorted_affected[:5]:
                status = " [FAILED]" if bank in initial_failures else ""
                margin_info = f" (margin buffer: ${data['margin_used']:.2f}B)" if data['margin_used'] > 0 else ""
                print(f"    {bank}: -${data['loss']:.2f}B{margin_info}{status}")
            if len(sorted_affected) > 5:
                print(f"    ... and {len(sorted_affected) - 5} more banks")
        
        # Propagate contagion from initially failed banks
        round_num = 0
        newly_failed = initial_failures.copy()
        
        while newly_failed and round_num < max_rounds:
            round_num += 1
            previously_failed = newly_failed.copy()
            newly_failed = set()
            
            round_dampening = 0.7 ** round_num
            
            for failed_bank in previously_failed:
                impact = self._compute_bank_impact(failed_bank)
                
                # Impact creditors
                for creditor in self.graph[failed_bank]['neighbors']:
                    if creditor not in self.failed_banks:
                        exposure = self._get_exposure(failed_bank, creditor)
                        loss = impact * exposure * round_dampening
                        
                        creditor_assets = self.bank_states[creditor]['Total_Assets']
                        loss = min(loss, creditor_assets * 0.15)
                        
                        # Use margin buffer for contagion losses
                        actual_loss, _ = self._use_margin_buffer(creditor, loss)
                        
                        self.bank_states[creditor]['Total_Assets'] -= actual_loss
                        self.bank_states[creditor]['Equity'] -= actual_loss
                        
                        if self.get_bank_health(creditor) < failure_threshold:
                            self.mark_bank_failed(creditor)
                            newly_failed.add(creditor)
                
                # Impact borrowers
                for borrower in self.graph:
                    if failed_bank in self.graph[borrower]['neighbors']:
                        if borrower not in self.failed_banks:
                            exposure = self._get_exposure(borrower, failed_bank)
                            loss = impact * exposure * 0.3 * round_dampening
                            
                            borrower_assets = self.bank_states[borrower]['Total_Assets']
                            loss = min(loss, borrower_assets * 0.10)
                            
                            # Use margin buffer for contagion losses
                            actual_loss, _ = self._use_margin_buffer(borrower, loss)
                            
                            self.bank_states[borrower]['Total_Assets'] -= actual_loss
                            self.bank_states[borrower]['Equity'] -= actual_loss
                            
                            if self.get_bank_health(borrower) < failure_threshold:
                                self.mark_bank_failed(borrower)
                                newly_failed.add(borrower)
            
            round_state = {
                'round': round_num,
                'failed_banks': len(self.failed_banks),
                'newly_failed': list(newly_failed),
                'total_asset_loss': self._compute_total_asset_loss()
            }
            self.history.append(round_state)
        
        return self._generate_multi_stock_report(stock_devaluations, failure_threshold, initial_failures)
    
    def _generate_multi_stock_report(self, stock_devaluations, threshold, initial_failures):
        """Generate comprehensive multi-stock devaluation report."""
        num_failed = len(self.failed_banks)
        num_total = len(self.graph)
        collapse_ratio = num_failed / num_total
        total_loss = self._compute_total_asset_loss()
        
        survived = set(self.graph.keys()) - self.failed_banks
        avg_survivor_health = sum(self.get_bank_health(b) for b in survived) / len(survived) if survived else 0
        
        return {
            'stock_devaluations': stock_devaluations,
            'failure_threshold': threshold,
            'num_failed_banks': num_failed,
            'total_banks': num_total,
            'collapse_ratio': collapse_ratio,
            'total_asset_loss': total_loss,
            'avg_survivor_health': avg_survivor_health,
            'rounds_until_stability': len(self.history),
            'failed_banks': list(self.failed_banks),
            'initial_failures': list(initial_failures),
            'contagion_history': self.history,
            'system_collapsed': collapse_ratio > 0.5
        }

    def _generate_stock_report(self, stock_ticker, devaluation_pct, threshold, initial_failures):
        """Generate comprehensive stock devaluation report."""
        num_failed = len(self.failed_banks)
        num_total = len(self.graph)
        collapse_ratio = num_failed / num_total
        total_loss = self._compute_total_asset_loss()
        
        survived = set(self.graph.keys()) - self.failed_banks
        avg_survivor_health = sum(self.get_bank_health(b) for b in survived) / len(survived) if survived else 0
        
        return {
            'stock_ticker': stock_ticker,
            'devaluation_pct': devaluation_pct,
            'failure_threshold': threshold,
            'num_failed_banks': num_failed,
            'total_banks': num_total,
            'collapse_ratio': collapse_ratio,
            'total_asset_loss': total_loss,
            'avg_survivor_health': avg_survivor_health,
            'rounds_until_stability': len(self.history),
            'failed_banks': list(self.failed_banks),
            'initial_failures': list(initial_failures),
            'contagion_history': self.history,
            'system_collapsed': collapse_ratio > 0.5
        }

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


def print_stock_report(result):
    """Pretty-print stock devaluation simulation results."""
    print(f"Stock Devalued: {result['stock_ticker']}")
    print(f"Devaluation: {result['devaluation_pct']}%")
    print(f"\nResults:")
    print(f"  Banks Failed: {result['num_failed_banks']}/{result['total_banks']}")
    print(f"  Collapse Ratio: {result['collapse_ratio']:.2%}")
    print(f"  System Collapsed: {'YES' if result['system_collapsed'] else 'NO'}")
    print(f"  Total Asset Loss: ${result['total_asset_loss']:.2f}B")
    print(f"  Avg Survivor Health: {result['avg_survivor_health']:.2f}")
    print(f"  Rounds Until Stability: {result['rounds_until_stability']}")
    
    if result['initial_failures']:
        print(f"\n  Banks Failed from Direct Stock Loss: {', '.join(result['initial_failures'][:10])}")
        if len(result['initial_failures']) > 10:
            print(f"    ... and {len(result['initial_failures']) - 10} more")
    
    if result['failed_banks']:
        contagion_failures = set(result['failed_banks']) - set(result['initial_failures'])
        if contagion_failures:
            print(f"  Banks Failed from Contagion: {', '.join(list(contagion_failures)[:10])}")
    
    if result['contagion_history']:
        print(f"\n  Contagion Progression:")
        for state in result['contagion_history']:
            print(f"    Round {state['round']}: {state['failed_banks']} failed " +
                  f"(+{len(state['newly_failed'])} new)")


# Example usage
if __name__ == '__main__':
    # Load banks and generate network
    bank_attrs = load_bank_attributes('./dataset/us_banks_top50_nodes_final.csv')
    G = generate_random_graph_with_sccs(bank_attrs, num_sccs=4, prob_intra=0.4, prob_inter=0.05)

    # --- Load stocks and distribute shares among banks ---
    stock_prices, stock_timeseries = load_stock_prices(
        './dataset/stocks_data_long.csv', num_stocks=10
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
            val = sum(shares * stock_prices[t] for t, shares in holdings[bank].items()) / 1e9
            print(f"  {bank}: Holdings=${val:.2f}B, Total_Assets=${bank_attrs[bank]['Total_Assets']:.2f}B, #Stocks={len(holdings[bank])}")
    print()

    # --- Generate margin requirements for each bank ---
    # Margin is locked collateral that reduces liquidity but provides buffer during devaluation
    # For now using random values (2-8% of Total_Assets)
    # In production, this would be input by user or loaded from regulatory data
    margin_requirements = generate_margin_requirements(bank_attrs, margin_ratio_range=(0.02, 0.08))
    print_margin_summary(margin_requirements, bank_attrs)

    # Run contagion simulation with stock prices and margin requirements
    contagion = BankingNetworkContagion(G, stock_prices, margin_requirements)
    
    # Debug: Print initial health scores to verify they're reasonable
    print("Initial Bank Health Scores (sample):")
    sample_banks = ['JPM', 'BAC', 'GS', 'MS', 'IBOC', 'WFC']
    for b in sample_banks:
        if b in contagion.graph:
            margin = margin_requirements.get(b, 0)
            print(f"  {b}: Health={contagion.get_bank_health(b):.1f}/100, Margin=${margin:.2f}B")
    print()
    
    # Get list of stock tickers for scenarios
    tickers = list(stock_prices.keys())
    
    def generate_random_stock_scenario(tickers):
        """Generate a random stock devaluation scenario with 1-3 stocks."""
        num_stocks = random.randint(1, 3)
        selected_stocks = random.sample(tickers, min(num_stocks, len(tickers)))
        # Random continuous devaluation between 10% and 40%
        return {stock: round(random.uniform(0.1,40), 1) for stock in selected_stocks}
    
    def print_multi_stock_report(result):
        """Print report for multi-stock devaluation scenario."""
        print(f"\n  Results:")
        print(f"    Failed Banks: {result['num_failed_banks']}/{result['total_banks']} ({result['collapse_ratio']*100:.1f}%)")
        print(f"    Total Asset Loss: ${result['total_asset_loss']:.2f}B")
        print(f"    Rounds to Stability: {result['rounds_until_stability']}")
        print(f"    Avg Survivor Health: {result['avg_survivor_health']:.1f}/100")
        if result['failed_banks']:
            print(f"    Failed: {', '.join(result['failed_banks'][:10])}{'...' if len(result['failed_banks']) > 10 else ''}")
        print(f"    System Collapsed: {'YES' if result['system_collapsed'] else 'NO'}")
    
    # Run 4 random multi-stock devaluation scenarios
    for scenario_num in range(1, 5):
        scenario = generate_random_stock_scenario(tickers)
        scenario_desc = ", ".join([f"{s}:{p}%" for s, p in scenario.items()])
        
        print("\n" + "=" * 70)
        print(f"SCENARIO {scenario_num}: Multi-Stock Devaluation ({scenario_desc})")
        print("=" * 70)
        
        result = contagion.propagate_multi_stock_devaluation(scenario, failure_threshold=20)
        print_multi_stock_report(result)