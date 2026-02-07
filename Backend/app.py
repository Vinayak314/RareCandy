"""
Flask backend for the Banking Network Contagion Simulation.
Exposes REST API endpoints for the React frontend.
"""

import os
import sys
import csv
import random
import pickle
from collections import defaultdict
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add ML directory to path so we can reuse data loading utilities
ML_DIR = os.path.join(os.path.dirname(__file__), '..', 'ML')
DATASET_DIR = os.path.join(ML_DIR, 'dataset')
BACKEND_DIR = os.path.dirname(__file__)

app = Flask(__name__)
CORS(app)

# ─── Global simulation state ─────────────────────────────────────────────────
SIM_STATE = {
    'bank_attrs': None,
    'graph': None,
    'stock_prices': None,
    'stock_timeseries': None,
    'holdings': None,
    'contagion': None,
    'interbank_matrix': None,
}


# ─── Data loading functions (adapted from train2.py) ─────────────────────────

def load_bank_attributes(csv_path):
    banks = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bank_name = row['Bank']
            banks[bank_name] = {
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
                'LCR': float(row['LCR']),
            }
    return banks


def load_interbank_matrix(csv_path):
    """Load interbank exposure matrix."""
    matrix = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        bank_names = header[1:]  # skip first empty column
        for row in reader:
            from_bank = row[0]
            matrix[from_bank] = {}
            for i, to_bank in enumerate(bank_names):
                matrix[from_bank][to_bank] = float(row[i + 1])
    return matrix


def load_stock_prices(csv_path, num_stocks=10):
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
                'Volume': float(row['Volume']),
            })

    viable_tickers = [
        t for t in all_data if all_data[t][-1]['Close'] >= 5.0
    ]
    selected_tickers = random.sample(viable_tickers, min(num_stocks, len(viable_tickers)))

    selected_prices = {}
    selected_timeseries = {}
    for ticker in selected_tickers:
        ts = sorted(all_data[ticker], key=lambda x: x['Date'])
        selected_prices[ticker] = ts[-1]['Close']
        selected_timeseries[ticker] = ts

    return selected_prices, selected_timeseries


def distribute_shares(bank_attributes, stock_prices):
    import numpy as np
    tickers = list(stock_prices.keys())
    num_stocks = len(tickers)
    holdings = {}

    all_assets = [a['Total_Assets'] for a in bank_attributes.values()]
    max_assets, min_assets = max(all_assets), min(all_assets)
    asset_range = max_assets - min_assets if max_assets > min_assets else 1

    for bank_name, attrs in bank_attributes.items():
        total_assets = attrs['Total_Assets']
        size_pct = (total_assets - min_assets) / asset_range

        base_alpha = 0.1 + size_pct * 1.9
        if size_pct < 0.3:
            alphas = [base_alpha * 0.3] * num_stocks
            alphas[random.randint(0, num_stocks - 1)] = base_alpha * 5.0
        elif size_pct < 0.6:
            alphas = [base_alpha] * num_stocks
            alphas[random.randint(0, num_stocks - 1)] = base_alpha * 2.0
        else:
            alphas = [base_alpha] * num_stocks

        weights = np.random.dirichlet(alphas)
        bh = {}
        for i, ticker in enumerate(tickers):
            allocation = total_assets * weights[i]
            bh[ticker] = (allocation * 1e9) / stock_prices[ticker]
        holdings[bank_name] = bh

    return holdings


def build_graph(bank_attributes, interbank_matrix):
    """Build graph using actual interbank exposure matrix instead of random generation."""
    bank_list = list(bank_attributes.keys())

    graph = {}
    for bank in bank_list:
        graph[bank] = {
            'neighbors': [],
            'attributes': bank_attributes[bank].copy(),
            'holdings': {},
        }

    # Create edges based on interbank matrix (threshold: exposure > 0.01 $B)
    for from_bank in bank_list:
        if from_bank not in interbank_matrix:
            continue
        for to_bank in bank_list:
            if from_bank == to_bank:
                continue
            exposure = interbank_matrix.get(from_bank, {}).get(to_bank, 0)
            if exposure > 0.01:
                graph[from_bank]['neighbors'].append(to_bank)

    return graph


# ─── Contagion engine (self-contained, adapted from train2.py) ────────────────

class BankingNetworkContagion:
    def __init__(self, graph, stock_prices=None, interbank_matrix=None):
        self.graph = graph
        self.stock_prices = stock_prices.copy() if stock_prices else {}
        self.current_stock_prices = stock_prices.copy() if stock_prices else {}
        self.interbank_matrix = interbank_matrix or {}
        self.bank_states = {}
        self.failed_banks = set()
        self.history = []
        self.initialize_states()

    def initialize_states(self):
        for bank in self.graph:
            self.bank_states[bank] = self.graph[bank]['attributes'].copy()
        self.current_stock_prices = self.stock_prices.copy()

    def get_bank_health(self, bank):
        if bank in self.failed_banks:
            return 0.0
        attrs = self.bank_states[bank]
        total_assets = attrs['Total_Assets']
        equity = attrs['Equity']
        if total_assets <= 0:
            return 0.0

        equity_ratio = (equity / total_assets) * 100
        solvency_score = min(equity_ratio / 20.0, 1.0) * 100
        lcr_score = min(attrs['LCR'] / 2.0, 1.0) * 100
        cds_score = max(0, 1.0 - (attrs['Est_CDS_Spread'] / 500.0)) * 100
        vol_score = max(0, 1.0 - (attrs['Stock_Volatility'] / 0.8)) * 100

        health = (solvency_score * 0.40 + lcr_score * 0.30 +
                  cds_score * 0.15 + vol_score * 0.15)
        return round(max(0.0, min(100.0, health)), 2)

    def mark_bank_failed(self, bank):
        if bank not in self.failed_banks:
            self.failed_banks.add(bank)

    def _get_interbank_exposure(self, from_bank, to_bank):
        """Get actual interbank exposure from matrix."""
        return self.interbank_matrix.get(from_bank, {}).get(to_bank, 0)

    def propagate_bank_shock(self, initial_bank, devaluation_shock,
                             max_rounds=100, failure_threshold=20.0):
        self.initialize_states()
        self.failed_banks = set()
        self.history = []

        # Record initial state for all banks
        initial_healths = {b: self.get_bank_health(b) for b in self.graph}

        shock_amount = self.bank_states[initial_bank]['Total_Assets'] * (devaluation_shock / 100.0)
        self.bank_states[initial_bank]['Total_Assets'] -= shock_amount
        self.bank_states[initial_bank]['Equity'] -= shock_amount

        if self.get_bank_health(initial_bank) < failure_threshold:
            self.mark_bank_failed(initial_bank)

        newly_failed = {initial_bank} if initial_bank in self.failed_banks else set()

        round_num = 0
        while newly_failed and round_num < max_rounds:
            round_num += 1
            previously_failed = newly_failed.copy()
            newly_failed = set()
            round_dampening = 0.7 ** round_num

            for failed_bank in previously_failed:
                fb_attrs = self.bank_states[failed_bank]
                original_assets = self.graph[failed_bank]['attributes']['Total_Assets']
                asset_loss_ratio = max(0, 1 - (fb_attrs['Total_Assets'] / original_assets))

                for neighbor in self.graph[failed_bank]['neighbors']:
                    if neighbor not in self.failed_banks:
                        exposure = self._get_interbank_exposure(failed_bank, neighbor)
                        loss = asset_loss_ratio * exposure * round_dampening
                        creditor_assets = self.bank_states[neighbor]['Total_Assets']
                        loss = min(loss, creditor_assets * 0.15)

                        self.bank_states[neighbor]['Total_Assets'] -= loss
                        self.bank_states[neighbor]['Equity'] -= loss

                        if self.get_bank_health(neighbor) < failure_threshold:
                            self.mark_bank_failed(neighbor)
                            newly_failed.add(neighbor)

                # Reverse channel (weaker)
                for borrower in self.graph:
                    if failed_bank in self.graph[borrower]['neighbors']:
                        if borrower not in self.failed_banks:
                            exposure = self._get_interbank_exposure(borrower, failed_bank)
                            loss = asset_loss_ratio * exposure * 0.3 * round_dampening
                            b_assets = self.bank_states[borrower]['Total_Assets']
                            loss = min(loss, b_assets * 0.10)

                            self.bank_states[borrower]['Total_Assets'] -= loss
                            self.bank_states[borrower]['Equity'] -= loss

                            if self.get_bank_health(borrower) < failure_threshold:
                                self.mark_bank_failed(borrower)
                                newly_failed.add(borrower)

            self.history.append({
                'round': round_num,
                'failed_banks': len(self.failed_banks),
                'newly_failed': list(newly_failed),
                'total_asset_loss': self._total_loss(),
            })

        return self._build_result(initial_healths, failure_threshold)

    def propagate_stock_shock(self, stock_devaluations,
                              max_rounds=100, failure_threshold=20.0):
        self.initialize_states()
        self.failed_banks = set()
        self.history = []

        initial_healths = {b: self.get_bank_health(b) for b in self.graph}
        initial_failures = set()

        for ticker, pct in stock_devaluations.items():
            if ticker not in self.stock_prices:
                continue
            old_price = self.stock_prices[ticker]
            new_price = old_price * (1 - pct / 100.0)
            self.current_stock_prices[ticker] = new_price

            for bank in self.graph:
                holdings = self.graph[bank].get('holdings', {})
                if ticker in holdings:
                    shares = holdings[ticker]
                    loss_b = shares * (old_price - new_price) / 1e9
                    if loss_b > 0:
                        self.bank_states[bank]['Total_Assets'] -= loss_b
                        self.bank_states[bank]['Equity'] -= loss_b

        for bank in self.graph:
            if self.get_bank_health(bank) < failure_threshold:
                self.mark_bank_failed(bank)
                initial_failures.add(bank)

        newly_failed = initial_failures.copy()
        round_num = 0
        while newly_failed and round_num < max_rounds:
            round_num += 1
            previously_failed = newly_failed.copy()
            newly_failed = set()
            round_dampening = 0.7 ** round_num

            for failed_bank in previously_failed:
                fb_attrs = self.bank_states[failed_bank]
                original_assets = self.graph[failed_bank]['attributes']['Total_Assets']
                asset_loss_ratio = max(0, 1 - (fb_attrs['Total_Assets'] / original_assets))

                for neighbor in self.graph[failed_bank]['neighbors']:
                    if neighbor not in self.failed_banks:
                        exposure = self._get_interbank_exposure(failed_bank, neighbor)
                        loss = asset_loss_ratio * exposure * round_dampening
                        loss = min(loss, self.bank_states[neighbor]['Total_Assets'] * 0.15)
                        self.bank_states[neighbor]['Total_Assets'] -= loss
                        self.bank_states[neighbor]['Equity'] -= loss
                        if self.get_bank_health(neighbor) < failure_threshold:
                            self.mark_bank_failed(neighbor)
                            newly_failed.add(neighbor)

                for borrower in self.graph:
                    if failed_bank in self.graph[borrower]['neighbors']:
                        if borrower not in self.failed_banks:
                            exposure = self._get_interbank_exposure(borrower, failed_bank)
                            loss = asset_loss_ratio * exposure * 0.3 * round_dampening
                            loss = min(loss, self.bank_states[borrower]['Total_Assets'] * 0.10)
                            self.bank_states[borrower]['Total_Assets'] -= loss
                            self.bank_states[borrower]['Equity'] -= loss
                            if self.get_bank_health(borrower) < failure_threshold:
                                self.mark_bank_failed(borrower)
                                newly_failed.add(borrower)

            self.history.append({
                'round': round_num,
                'failed_banks': len(self.failed_banks),
                'newly_failed': list(newly_failed),
                'total_asset_loss': self._total_loss(),
            })

        return self._build_result(initial_healths, failure_threshold)

    def _total_loss(self):
        return sum(
            max(0, self.graph[b]['attributes']['Total_Assets'] - self.bank_states[b]['Total_Assets'])
            for b in self.bank_states
        )

    def _build_result(self, initial_healths, threshold):
        num_failed = len(self.failed_banks)
        num_total = len(self.graph)
        survived = set(self.graph.keys()) - self.failed_banks
        avg_health = (sum(self.get_bank_health(b) for b in survived) / len(survived)) if survived else 0

        # Per-bank detail for frontend
        banks_detail = []
        for bank in self.graph:
            attrs = self.bank_states[bank]
            orig = self.graph[bank]['attributes']
            banks_detail.append({
                'id': bank,
                'health': self.get_bank_health(bank),
                'initial_health': initial_healths.get(bank, 0),
                'failed': bank in self.failed_banks,
                'total_assets': round(attrs['Total_Assets'], 2),
                'original_assets': round(orig['Total_Assets'], 2),
                'equity': round(attrs['Equity'], 2),
                'asset_loss': round(max(0, orig['Total_Assets'] - attrs['Total_Assets']), 2),
            })

        return {
            'num_failed': num_failed,
            'total_banks': num_total,
            'collapse_ratio': round(num_failed / num_total, 4),
            'total_asset_loss': round(self._total_loss(), 2),
            'avg_survivor_health': round(avg_health, 2),
            'rounds': len(self.history),
            'system_collapsed': num_failed / num_total > 0.5,
            'failed_banks': list(self.failed_banks),
            'contagion_history': self.history,
            'banks': banks_detail,
        }


# ─── Initialization ──────────────────────────────────────────────────────────

def init_simulation():
    """Load all data and build the simulation state once at startup."""
    print("[INIT] Loading bank attributes...")
    SIM_STATE['bank_attrs'] = load_bank_attributes(
        os.path.join(DATASET_DIR, 'us_banks_top50_nodes_final.csv'))

    print("[INIT] Loading interbank matrix...")
    SIM_STATE['interbank_matrix'] = load_interbank_matrix(
        os.path.join(DATASET_DIR, 'us_banks_interbank_matrix.csv'))

    print("[INIT] Loading stock prices (selecting 10 random)...")
    prices, ts = load_stock_prices(
        os.path.join(DATASET_DIR, 'stocks_data_long.csv'), num_stocks=10)
    SIM_STATE['stock_prices'] = prices
    SIM_STATE['stock_timeseries'] = ts

    print("[INIT] Distributing shares among banks...")
    SIM_STATE['holdings'] = distribute_shares(SIM_STATE['bank_attrs'], prices)

    print("[INIT] Building interbank graph...")
    SIM_STATE['graph'] = build_graph(SIM_STATE['bank_attrs'], SIM_STATE['interbank_matrix'])

    # Attach holdings to graph
    for bank in SIM_STATE['graph']:
        if bank in SIM_STATE['holdings']:
            SIM_STATE['graph'][bank]['holdings'] = SIM_STATE['holdings'][bank]

    SIM_STATE['contagion'] = BankingNetworkContagion(
        SIM_STATE['graph'], prices, SIM_STATE['interbank_matrix'])

    print(f"[INIT] Ready — {len(SIM_STATE['bank_attrs'])} banks, "
          f"{len(prices)} stocks")


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/banks', methods=['GET'])
def get_banks():
    """Return list of banks with attributes and health scores."""
    contagion = SIM_STATE['contagion']
    contagion.initialize_states()  # fresh state
    contagion.failed_banks = set()

    banks = []
    for name, attrs in SIM_STATE['bank_attrs'].items():
        health = contagion.get_bank_health(name)
        banks.append({
            'id': name,
            'total_assets': attrs['Total_Assets'],
            'equity': attrs['Equity'],
            'health': health,
            'lcr': attrs['LCR'],
            'cds_spread': attrs['Est_CDS_Spread'],
            'volatility': attrs['Stock_Volatility'],
            'interbank_assets': attrs['Interbank_Assets'],
            'interbank_liabilities': attrs['Interbank_Liabilities'],
        })

    banks.sort(key=lambda b: -b['total_assets'])
    return jsonify(banks)


@app.route('/api/network', methods=['GET'])
def get_network():
    """Return the network graph as nodes + links for force-graph."""
    graph = SIM_STATE['graph']
    contagion = SIM_STATE['contagion']
    contagion.initialize_states()
    contagion.failed_banks = set()

    nodes = []
    for bank in graph:
        attrs = SIM_STATE['bank_attrs'][bank]
        nodes.append({
            'id': bank,
            'name': bank,
            'val': max(3, attrs['Total_Assets'] / 200),  # node size
            'total_assets': attrs['Total_Assets'],
            'health': contagion.get_bank_health(bank),
        })

    links = []
    seen = set()
    for from_bank in graph:
        for to_bank in graph[from_bank]['neighbors']:
            key = (from_bank, to_bank)
            if key not in seen:
                seen.add(key)
                exposure = SIM_STATE['interbank_matrix'].get(from_bank, {}).get(to_bank, 0)
                links.append({
                    'source': from_bank,
                    'target': to_bank,
                    'value': round(exposure, 2),
                })

    return jsonify({'nodes': nodes, 'links': links})


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """Return the 10 selected stocks with current prices and time-series."""
    prices = SIM_STATE['stock_prices']
    ts = SIM_STATE['stock_timeseries']

    stocks = []
    for ticker, price in prices.items():
        series = ts.get(ticker, [])
        stocks.append({
            'ticker': ticker,
            'price': round(price, 2),
            'timeseries': series[-30:] if len(series) > 30 else series,  # last 30 days
        })
    return jsonify(stocks)


@app.route('/api/holdings', methods=['GET'])
def get_holdings():
    """Return share holdings for all banks."""
    holdings = SIM_STATE['holdings']
    prices = SIM_STATE['stock_prices']
    result = {}
    for bank, bh in holdings.items():
        result[bank] = {}
        for ticker, shares in bh.items():
            result[bank][ticker] = {
                'shares': round(shares, 0),
                'value_B': round(shares * prices[ticker] / 1e9, 4),
            }
    return jsonify(result)


@app.route('/api/simulate/bank-shock', methods=['POST'])
def simulate_bank_shock():
    """
    Run a bank-level shock simulation.
    Body: { "bank": "JPM", "shock_pct": 50, "failure_threshold": 20 }
    """
    data = request.get_json()
    bank = data.get('bank')
    shock_pct = data.get('shock_pct', 50)
    threshold = data.get('failure_threshold', 20)

    if bank not in SIM_STATE['bank_attrs']:
        return jsonify({'error': f'Bank {bank} not found'}), 400

    contagion = SIM_STATE['contagion']
    result = contagion.propagate_bank_shock(bank, shock_pct, failure_threshold=threshold)

    # Also include the current network state for graph re-rendering
    network = _get_post_sim_network(contagion)
    result['network'] = network

    return jsonify(result)


@app.route('/api/simulate/stock-shock', methods=['POST'])
def simulate_stock_shock():
    """
    Run a stock-level shock simulation.
    Body: { "shocks": {"AAPL": 30, "MSFT": 20}, "failure_threshold": 20 }
    """
    data = request.get_json()
    shocks = data.get('shocks', {})
    threshold = data.get('failure_threshold', 20)

    if not shocks:
        return jsonify({'error': 'No stock shocks provided'}), 400

    contagion = SIM_STATE['contagion']
    result = contagion.propagate_stock_shock(shocks, failure_threshold=threshold)

    network = _get_post_sim_network(contagion)
    result['network'] = network

    return jsonify(result)


def _get_post_sim_network(contagion):
    """Build network data reflecting post-simulation state."""
    graph = SIM_STATE['graph']
    nodes = []
    for bank in graph:
        attrs = contagion.bank_states[bank]
        nodes.append({
            'id': bank,
            'name': bank,
            'val': max(3, attrs['Total_Assets'] / 200),
            'total_assets': round(attrs['Total_Assets'], 2),
            'health': contagion.get_bank_health(bank),
            'failed': bank in contagion.failed_banks,
        })

    links = []
    seen = set()
    for from_bank in graph:
        for to_bank in graph[from_bank]['neighbors']:
            key = (from_bank, to_bank)
            if key not in seen:
                seen.add(key)
                exposure = SIM_STATE['interbank_matrix'].get(from_bank, {}).get(to_bank, 0)
                links.append({
                    'source': from_bank,
                    'target': to_bank,
                    'value': round(exposure, 2),
                })

    return {'nodes': nodes, 'links': links}


@app.route('/api/simulate/reset', methods=['POST'])
def reset_simulation():
    """Re-initialize the simulation with fresh random stock selection & holdings."""
    init_simulation()
    return jsonify({'status': 'ok', 'message': 'Simulation reset with new random stocks'})


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_simulation()
    app.run(debug=True, port=5000)
