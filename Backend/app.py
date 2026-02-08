"""
Flask backend for the Banking Network Contagion Simulation.
Exposes REST API endpoints for the React frontend.
Adapted from SpankBank simulation - includes shock simulation and CCP payoff calculation.
"""

import os
import csv
import random
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ML model integration
try:
    from ML.ccp_model import CCPMarginPredictor, get_margin_predictor, MARGIN_LEVELS
    ML_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ML model integration not available: {e}")
    ML_MODEL_AVAILABLE = False

# Import shared algorithm functions
from ML.algorithm import (
    load_bank_attributes,
    load_interbank_matrix,
    load_stock_prices,
    distribute_shares,
    generate_margin_requirements,
)
# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Time horizon
    'T': 100,
    'NUM_EPISODES': 1000,

    # RL parameters
    'GAMMA': 0.99,
    'EPSILON_START': 1.0,
    'EPSILON_MIN': 0.05,
    'EPSILON_DECAY': 0.995,
    'MEMORY_SIZE': 5000,
    'BATCH_SIZE': 64,

    # Bank payoff weights
    'LAMBDA_RISK': 0.3,
    'MU_MARGIN': 0.1,
    'PHI_DEFAULT': 100.0,

    # CCP payoff weights
    'ALPHA_SYSTEMIC': 1.0,
    'BETA_DF_LOSS': 0.5,
    'GAMMA_VOLUME': 0.01,

    # Market parameters
    'FUNDING_RATE': 0.05,
    'CRASH_SEVERITY': 1.5,
    'SHOCK_RANGE': (0.01, 0.10),

    # Model paths
    'BANK_MODEL_PATH': 'bank_policies.pkl',
    'CCP_MODEL_PATH': 'ccp_policy.pkl',
}

# Paths
BACKEND_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BACKEND_DIR, 'ml/dataset')

app = Flask(__name__)
CORS(app)

# Load Featherless API Key
FLESS_KEY = os.getenv('FLESS')
if not FLESS_KEY:
    # Try loading from .env file manually if not in env
    try:
        with open(os.path.join(BACKEND_DIR, '.env'), 'r') as f:
            for line in f:
                if line.startswith('FLESS='):
                    FLESS_KEY = line.strip().split('=', 1)[1]
                    break
    except Exception:
        pass

import requests
import datetime

@app.route('/api/greeting', methods=['GET'])
def get_greeting():
    """Get a dynamic greeting from Featherless.ai based on time of day."""
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        time_of_day = "morning"
    elif 12 <= current_hour < 17:
        time_of_day = "afternoon"
    elif 17 <= current_hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    if not FLESS_KEY:
        return jsonify({'greeting': f"Good {time_of_day}!"})

    try:
        # Using Featherless.ai API (Assuming OpenAI-compatible format as is common)
        # If specific endpoint differs, this will need adjustment.
        # Based on standard interference API patterns.
        response = requests.post(
            "https://api.featherless.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {FLESS_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-ai/DeepSeek-V3-0324", # Using model from provided docs
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Keep it very short."},
                    {"role": "user", "content": f"Give me a short, professional greeting for a financial simulation dashboard user. It is currently {time_of_day}. Max 10 words."}
                ],
                "temperature": 0.7,
                "max_tokens": 50
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            greeting = data['choices'][0]['message']['content'].strip().replace('"', '')
            return jsonify({'greeting': greeting})
        else:
            print(f"Featherless API Error: {response.text}")
            return jsonify({'greeting': f"Good {time_of_day}!"})
            
    except Exception as e:
        print(f"Error fetching greeting: {e}")
        return jsonify({'greeting': f"Good {time_of_day}!"})

# ─── Global simulation state ─────────────────────────────────────────────────
SIM_STATE = {
    'bank_attrs': None,
    'graph': None,
    'stock_prices': None,
    'stock_timeseries': None,
    'holdings': None,
    'contagion': None,
    'interbank_matrix': None,
    'network_simulator': None,
    # ML model integration
    'margin_requirements': None,
    'margin_predictor': None,
    'use_ml_margins': False,
}


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================
class TradeDecision(Enum):
    APPROVED = "APPROVED"
    REQUIRE_MARGIN = "REQUIRE_MARGIN"
    REJECTED = "REJECTED"


@dataclass
class SystemState:
    """Global system state S_t = (E, L, A, X, G)"""
    equity: Dict[str, float]
    liquidity: Dict[str, float]
    assets: Dict[str, float]
    exposures: Dict[str, Dict[str, float]]
    liabilities: Dict[str, float]
    stock_volatility: Dict[str, float]
    failed_banks: Set[str] = field(default_factory=set)


# ============================================================================
# NETWORK SIMULATOR
# ============================================================================
class NetworkSimulator:
    """Simulates the banking network state evolution."""

    def __init__(self, banks_file: str, stocks_file: str, matrix_file: str):
        self.banks_df = pd.read_csv(banks_file)
        self.interbank_matrix = pd.read_csv(matrix_file, index_col=0)

        # Calculate stock volatility
        stocks_df = pd.read_csv(stocks_file)
        stocks_df['Return'] = stocks_df.groupby('Ticker')['Close'].pct_change()
        vol_df = stocks_df.groupby('Ticker')['Return'].std().reset_index()
        vol_df.columns = ['Ticker', 'Volatility']
        self.stock_volatility = vol_df.set_index('Ticker')['Volatility'].to_dict()

        self.bank_list = self.banks_df['Bank'].tolist()
        self.initial_state = self._create_initial_state()
        self.current_state = None
        self.reset()

        self.scaler = StandardScaler()
        features = ['Total_Assets', 'Equity', 'HQLA', 'Net_Outflows_30d', 'Interbank_Liabilities']
        self.scaler.fit(self.banks_df[features].fillna(0))

        print(f"✅ NetworkSimulator loaded: {len(self.bank_list)} banks, "
              f"{len(self.stock_volatility)} stocks")

    def _create_initial_state(self) -> SystemState:
        equity = {}
        liquidity = {}
        assets = {}
        liabilities = {}
        exposures = {}

        for _, row in self.banks_df.iterrows():
            bank = row['Bank']
            equity[bank] = row['Equity']
            liquidity[bank] = row['HQLA']
            assets[bank] = row['Total_Assets']
            liabilities[bank] = row['Total_Liabilities']

            exposures[bank] = {}
            for other in self.bank_list:
                if bank != other:
                    exp = self.interbank_matrix.loc[bank, other] if bank in self.interbank_matrix.index else 0
                    exposures[bank][other] = exp

        return SystemState(
            equity=equity,
            liquidity=liquidity,
            assets=assets,
            exposures=exposures,
            liabilities=liabilities,
            stock_volatility=self.stock_volatility
        )

    def reset(self) -> SystemState:
        self.current_state = SystemState(
            equity=self.initial_state.equity.copy(),
            liquidity=self.initial_state.liquidity.copy(),
            assets=self.initial_state.assets.copy(),
            exposures={b: e.copy() for b, e in self.initial_state.exposures.items()},
            liabilities=self.initial_state.liabilities.copy(),
            stock_volatility=self.initial_state.stock_volatility.copy(),
            failed_banks=set()
        )
        return self.current_state

    def get_state(self) -> SystemState:
        return self.current_state

    def get_bank_health(self, bank: str) -> float:
        state = self.current_state
        if bank in state.failed_banks:
            return 0.0

        assets = state.assets.get(bank, 0)
        equity = state.equity.get(bank, 0)

        if assets <= 0:
            return 0.0

        equity_ratio = equity / assets
        capital_score = min(40.0, (equity_ratio / 0.15) * 40.0)

        lcr = state.liquidity.get(bank, 0) / max(1, state.liabilities.get(bank, 1) * 0.1)
        liquidity_score = min(30.0, (lcr / 1.5) * 30.0)

        bank_row = self.banks_df[self.banks_df['Bank'] == bank]
        if not bank_row.empty:
            cds = bank_row['Est_CDS_Spread'].values[0]
            vol = bank_row['Stock_Volatility'].values[0]
            risk_penalty = min(15.0, cds / 400.0 * 15.0) + min(10.0, vol / 0.5 * 10.0)
        else:
            risk_penalty = 10.0

        health = capital_score + liquidity_score - risk_penalty
        return max(0.0, min(100.0, health))

    def run_contagion(self, failure_threshold: float = 20.0) -> Dict:
        state = self.current_state
        initial_failed = state.failed_banks.copy()

        newly_failed = set()
        for bank in self.bank_list:
            if bank not in state.failed_banks:
                if self.get_bank_health(bank) < failure_threshold:
                    state.failed_banks.add(bank)
                    newly_failed.add(bank)

        rounds = 0
        max_rounds = 50

        while newly_failed and rounds < max_rounds:
            rounds += 1
            previous_failed = newly_failed.copy()
            newly_failed = set()

            dampening = 0.7 ** rounds

            for failed_bank in previous_failed:
                exposures = state.exposures.get(failed_bank, {})

                for creditor, exposure in exposures.items():
                    if creditor not in state.failed_banks and exposure > 0:
                        loss = exposure * dampening * 0.5
                        loss = min(loss, state.assets[creditor] * 0.15)

                        state.assets[creditor] -= loss
                        state.equity[creditor] -= loss

                        if self.get_bank_health(creditor) < failure_threshold:
                            state.failed_banks.add(creditor)
                            newly_failed.add(creditor)

        systemic_loss = sum(
            max(0, state.liabilities[b] - state.assets[b])
            for b in state.failed_banks
        )

        return {
            'new_failures': len(state.failed_banks) - len(initial_failed),
            'total_failures': len(state.failed_banks),
            'systemic_loss': systemic_loss,
            'rounds': rounds
        }


# ============================================================================
# CCP PAYOFF CALCULATOR
# ============================================================================
class CCPPayoffCalculator:
    """Calculates the CCP's payoff (loss absorption) when banks fail."""
    
    def __init__(self, recovery_rate: float = 0.40):
        self.recovery_rate = recovery_rate
    
    def calculate_payoff(self, failed_banks: Set[str], state: SystemState) -> float:
        total_payoff = 0.0
        
        for bank in failed_banks:
            liabilities = state.liabilities.get(bank, 0)
            assets = state.assets.get(bank, 0)
            shortfall = max(0, liabilities - assets * self.recovery_rate)
            total_payoff += shortfall
        
        return total_payoff
    
    def get_detailed_breakdown(self, failed_banks: Set[str], state: SystemState) -> List[Dict]:
        breakdown = []
        
        for bank in failed_banks:
            liabilities = state.liabilities.get(bank, 0)
            assets = state.assets.get(bank, 0)
            equity = state.equity.get(bank, 0)
            recoverable = assets * self.recovery_rate
            shortfall = max(0, liabilities - recoverable)
            
            breakdown.append({
                'bank': bank,
                'total_liabilities_B': round(liabilities, 2),
                'total_assets_B': round(assets, 2),
                'final_equity_B': round(equity, 2),
                'recoverable_assets_B': round(recoverable, 2),
                'ccp_payoff_B': round(shortfall, 2)
            })
        
        return breakdown


# ============================================================================
# BANK SHOCK SIMULATOR
# ============================================================================
class BankShockSimulator:
    """Simulates direct shocks to specific banks."""
    
    def __init__(self, network: NetworkSimulator, payoff_calculator: CCPPayoffCalculator = None):
        self.network = network
        self.payoff_calculator = payoff_calculator or CCPPayoffCalculator()
    
    def simulate_bank_shock(self, bank_name: str, shock_percent: float, 
                            reset_first: bool = True, failure_threshold: float = 20.0) -> Dict:
        if reset_first:
            self.network.reset()
        
        state = self.network.get_state()
        
        if bank_name not in self.network.bank_list:
            raise ValueError(f"Bank '{bank_name}' not found in network")
        
        initial_equity = state.equity.get(bank_name, 0)
        initial_assets = state.assets.get(bank_name, 0)
        
        # Apply shock (shock_percent is 0-100 from frontend)
        shock_fraction = shock_percent / 100.0
        shock_amount = initial_assets * shock_fraction
        state.equity[bank_name] -= shock_amount
        state.assets[bank_name] -= shock_amount
        
        direct_failure = state.equity[bank_name] <= 0
        if direct_failure:
            state.failed_banks.add(bank_name)
        
        bank_health_after = self.network.get_bank_health(bank_name)
        
        # Run contagion cascade
        cascade_result = self.network.run_contagion(failure_threshold=failure_threshold)
        
        final_state = self.network.get_state()
        
        # Calculate CCP payoff
        ccp_payoff = self.payoff_calculator.calculate_payoff(
            final_state.failed_banks, final_state
        )
        
        payoff_breakdown = self.payoff_calculator.get_detailed_breakdown(
            final_state.failed_banks, final_state
        )
        
        return {
            'shock_type': 'BANK_SHOCK',
            'target_bank': bank_name,
            'shock_percent': shock_percent,
            'shock_amount_B': round(shock_amount, 2),
            'initial_equity_B': round(initial_equity, 2),
            'final_equity_B': round(final_state.equity.get(bank_name, 0), 2),
            'bank_health_after': round(bank_health_after, 2),
            'direct_failure': direct_failure,
            'failed_banks': list(final_state.failed_banks),
            'total_failures': len(final_state.failed_banks),
            'cascade_rounds': cascade_result['rounds'],
            'systemic_loss_B': round(cascade_result['systemic_loss'], 2),
            'ccp_payoff_B': round(ccp_payoff, 2),
            'payoff_breakdown': payoff_breakdown
        }


# ─── Data loading functions ───────────────────────────────────────────────────
# Note: load_bank_attributes, load_interbank_matrix, load_stock_prices, 
# and distribute_shares are imported from ML.algorithm


def build_graph(bank_attributes, interbank_matrix):
    bank_list = list(bank_attributes.keys())

    graph = {}
    for bank in bank_list:
        graph[bank] = {
            'neighbors': [],
            'attributes': bank_attributes[bank].copy(),
            'holdings': {},
        }

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


# ─── Contagion engine ─────────────────────────────────────────────────────────

class BankingNetworkContagion:
    def __init__(self, graph, stock_prices=None, interbank_matrix=None, margin_requirements=None):
        self.graph = graph
        self.stock_prices = stock_prices.copy() if stock_prices else {}
        self.current_stock_prices = stock_prices.copy() if stock_prices else {}
        self.interbank_matrix = interbank_matrix or {}
        self.margin_requirements = margin_requirements.copy() if margin_requirements else {}
        self.bank_states = {}
        self.margin_states = {}  # Track remaining margin for each bank
        self.failed_banks = set()
        self.history = []
        self.initialize_states()

    def initialize_states(self):
        for bank in self.graph:
            self.bank_states[bank] = self.graph[bank]['attributes'].copy()
            # Initialize margin: locked amount that can be used as buffer during stress
            self.margin_states[bank] = self.margin_requirements.get(bank, 0)
            # Reduce available HQLA by margin (margin is locked)
            if self.margin_states[bank] > 0:
                self.bank_states[bank]['HQLA'] = max(0, self.bank_states[bank]['HQLA'] - self.margin_states[bank])
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
        return self.interbank_matrix.get(from_bank, {}).get(to_bank, 0)

    def propagate_bank_shock(self, initial_bank, devaluation_shock,
                             max_rounds=100, failure_threshold=20.0):
        self.initialize_states()
        self.failed_banks = set()
        self.history = []

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

                        # Use margin buffer to absorb losses
                        actual_loss, _ = self._use_margin_buffer(neighbor, loss)
                        self.bank_states[neighbor]['Total_Assets'] -= actual_loss
                        self.bank_states[neighbor]['Equity'] -= actual_loss

                        if self.get_bank_health(neighbor) < failure_threshold:
                            self.mark_bank_failed(neighbor)
                            newly_failed.add(neighbor)

                for borrower in self.graph:
                    if failed_bank in self.graph[borrower]['neighbors']:
                        if borrower not in self.failed_banks:
                            exposure = self._get_interbank_exposure(borrower, failed_bank)
                            loss = asset_loss_ratio * exposure * 0.3 * round_dampening
                            b_assets = self.bank_states[borrower]['Total_Assets']
                            loss = min(loss, b_assets * 0.10)

                            # Use margin buffer to absorb losses
                            actual_loss, _ = self._use_margin_buffer(borrower, loss)
                            self.bank_states[borrower]['Total_Assets'] -= actual_loss
                            self.bank_states[borrower]['Equity'] -= actual_loss

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
                        # Use margin buffer to absorb losses
                        actual_loss, _ = self._use_margin_buffer(bank, loss_b)
                        self.bank_states[bank]['Total_Assets'] -= actual_loss
                        self.bank_states[bank]['Equity'] -= actual_loss

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
                        # Use margin buffer to absorb losses
                        actual_loss, _ = self._use_margin_buffer(neighbor, loss)
                        self.bank_states[neighbor]['Total_Assets'] -= actual_loss
                        self.bank_states[neighbor]['Equity'] -= actual_loss
                        if self.get_bank_health(neighbor) < failure_threshold:
                            self.mark_bank_failed(neighbor)
                            newly_failed.add(neighbor)

                for borrower in self.graph:
                    if failed_bank in self.graph[borrower]['neighbors']:
                        if borrower not in self.failed_banks:
                            exposure = self._get_interbank_exposure(borrower, failed_bank)
                            loss = asset_loss_ratio * exposure * 0.3 * round_dampening
                            loss = min(loss, self.bank_states[borrower]['Total_Assets'] * 0.10)
                            # Use margin buffer to absorb losses
                            actual_loss, _ = self._use_margin_buffer(borrower, loss)
                            self.bank_states[borrower]['Total_Assets'] -= actual_loss
                            self.bank_states[borrower]['Equity'] -= actual_loss
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

        # Calculate CCP payoff
        payoff_calc = CCPPayoffCalculator()
        # Build a temporary SystemState-like object for payoff calculation
        class TempState:
            def __init__(self, bank_states, failed):
                self.liabilities = {b: bank_states[b]['Total_Liabilities'] for b in bank_states}
                self.assets = {b: bank_states[b]['Total_Assets'] for b in bank_states}
                self.equity = {b: bank_states[b]['Equity'] for b in bank_states}
                self.failed_banks = failed
        
        temp_state = TempState(self.bank_states, self.failed_banks)
        ccp_payoff = payoff_calc.calculate_payoff(self.failed_banks, temp_state)
        payoff_breakdown = payoff_calc.get_detailed_breakdown(self.failed_banks, temp_state)

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
            'num_failed': int(num_failed),
            'total_banks': int(num_total),
            'collapse_ratio': float(round(num_failed / num_total, 4)),
            'total_asset_loss': float(round(self._total_loss(), 2)),
            'avg_survivor_health': float(round(avg_health, 2)),
            'rounds': int(len(self.history)),
            'system_collapsed': bool(num_failed / num_total > 0.5),
            'failed_banks': list(self.failed_banks),
            'contagion_history': self.history,
            'banks': banks_detail,
            'ccp_payoff_B': float(round(ccp_payoff, 2)),
            'payoff_breakdown': payoff_breakdown,
        }


# ─── Initialization ──────────────────────────────────────────────────────────
# Note: generate_margin_requirements is imported from ML.algorithm


def init_simulation(use_ml_margins=True):
    """
    Load all data and build the simulation state once at startup.
    
    Args:
        use_ml_margins: If True, use ML model for margin requirements.
                        If False, use random margins.
    """
    print("[INIT] Loading bank attributes...")
    SIM_STATE['bank_attrs'] = load_bank_attributes(
        os.path.join(DATASET_DIR, 'us_banks_top50_nodes_final.csv'))

    print("[INIT] Loading interbank matrix...")
    SIM_STATE['interbank_matrix'] = load_interbank_matrix(
        os.path.join(DATASET_DIR, 'us_banks_interbank_matrix.csv'))

    print("[INIT] Loading stock prices (selecting 30 random)...")
    prices, ts = load_stock_prices(
        os.path.join(DATASET_DIR, 'stocks_data_long.csv'), num_stocks=30)
    SIM_STATE['stock_prices'] = prices
    SIM_STATE['stock_timeseries'] = ts

    print("[INIT] Distributing shares among banks...")
    SIM_STATE['holdings'] = distribute_shares(SIM_STATE['bank_attrs'], prices)

    print("[INIT] Building interbank graph...")
    SIM_STATE['graph'] = build_graph(SIM_STATE['bank_attrs'], SIM_STATE['interbank_matrix'])

    for bank in SIM_STATE['graph']:
        if bank in SIM_STATE['holdings']:
            SIM_STATE['graph'][bank]['holdings'] = SIM_STATE['holdings'][bank]

    # --- ML Model Integration for Margin Requirements ---
    print("[INIT] Setting up margin requirements...")
    SIM_STATE['use_ml_margins'] = use_ml_margins and ML_MODEL_AVAILABLE
    
    if SIM_STATE['use_ml_margins']:
        try:
            print("[INIT] Loading CCP ML model for margin predictions...")
            SIM_STATE['margin_predictor'] = get_margin_predictor(
                os.path.join(BACKEND_DIR, 'ML', 'ccp_policy.pt')
            )
            SIM_STATE['margin_requirements'] = SIM_STATE['margin_predictor'].generate_margin_requirements(
                SIM_STATE['bank_attrs']
            )
            print(f"[INIT] ✅ ML-based margins generated for {len(SIM_STATE['margin_requirements'])} banks")
        except Exception as e:
            print(f"[INIT] ⚠️  ML margin generation failed: {e}")
            print("[INIT] Falling back to random margins...")
            SIM_STATE['use_ml_margins'] = False
            SIM_STATE['margin_requirements'] = generate_margin_requirements(SIM_STATE['bank_attrs'])
    else:
        print("[INIT] Using random margin requirements (ML not available or disabled)")
        SIM_STATE['margin_requirements'] = generate_margin_requirements(SIM_STATE['bank_attrs'])

    # Create contagion simulator with margins
    SIM_STATE['contagion'] = BankingNetworkContagion(
        SIM_STATE['graph'], 
        prices, 
        SIM_STATE['interbank_matrix'],
        SIM_STATE['margin_requirements']
    )

    # Also initialize NetworkSimulator for advanced shock simulation
    SIM_STATE['network_simulator'] = NetworkSimulator(
        os.path.join(DATASET_DIR, 'us_banks_top50_nodes_final.csv'),
        os.path.join(DATASET_DIR, 'stocks_data_long.csv'),
        os.path.join(DATASET_DIR, 'us_banks_interbank_matrix.csv')
    )

    print(f"[INIT] Ready — {len(SIM_STATE['bank_attrs'])} banks, "
          f"{len(prices)} stocks, ML margins: {SIM_STATE['use_ml_margins']}")


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/banks', methods=['GET'])
def get_banks():
    """Return list of banks with attributes, health scores, and margin requirements."""
    contagion = SIM_STATE['contagion']
    contagion.initialize_states()
    contagion.failed_banks = set()

    margins = SIM_STATE.get('margin_requirements', {})
    
    banks = []
    for name, attrs in SIM_STATE['bank_attrs'].items():
        health = contagion.get_bank_health(name)
        margin = margins.get(name, 0)
        margin_pct = (margin / attrs['Total_Assets'] * 100) if attrs['Total_Assets'] > 0 else 0
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
            'margin_B': round(margin, 3),
            'margin_pct': round(margin_pct, 2),
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
            'val': max(3, attrs['Total_Assets'] / 200),
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
            'timeseries': series[-30:] if len(series) > 30 else series,
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
    data = request.get_json() or {}
    use_ml = data.get('use_ml_margins', True)
    init_simulation(use_ml_margins=use_ml)
    return jsonify({
        'status': 'ok', 
        'message': 'Simulation reset with new random stocks',
        'ml_margins_enabled': SIM_STATE['use_ml_margins']
    })


# ─── ML Model API Routes ──────────────────────────────────────────────────────

@app.route('/api/ml/status', methods=['GET'])
def get_ml_status():
    """Get the status of the ML model integration."""
    return jsonify({
        'ml_available': ML_MODEL_AVAILABLE,
        'ml_margins_enabled': SIM_STATE.get('use_ml_margins', False),
        'predictor_loaded': SIM_STATE.get('margin_predictor') is not None,
        'model_loaded': (
            SIM_STATE.get('margin_predictor') is not None and 
            SIM_STATE['margin_predictor'].model_loaded
        )
    })


@app.route('/api/ml/margins', methods=['GET'])
def get_margin_requirements():
    """
    Get current margin requirements for all banks.
    Shows whether ML or random margins are being used.
    """
    margins = SIM_STATE.get('margin_requirements', {})
    bank_attrs = SIM_STATE.get('bank_attrs', {})
    
    result = []
    for bank_name, margin in margins.items():
        attrs = bank_attrs.get(bank_name, {})
        total_assets = attrs.get('Total_Assets', 0)
        margin_pct = (margin / total_assets * 100) if total_assets > 0 else 0
        result.append({
            'bank': bank_name,
            'margin_B': round(margin, 3),
            'margin_pct': round(margin_pct, 2),
            'total_assets_B': round(total_assets, 2)
        })
    
    result.sort(key=lambda x: -x['margin_B'])
    
    return jsonify({
        'ml_margins_enabled': SIM_STATE.get('use_ml_margins', False),
        'total_margin_B': round(sum(margins.values()), 2),
        'margins': result
    })


@app.route('/api/ml/margins/<bank_name>', methods=['GET'])
def get_bank_margin_details(bank_name):
    """
    Get detailed margin decision for a specific bank.
    Includes Q-values if ML model is used.
    """
    if bank_name not in SIM_STATE.get('bank_attrs', {}):
        return jsonify({'error': f'Bank {bank_name} not found'}), 404
    
    predictor = SIM_STATE.get('margin_predictor')
    if predictor is None:
        margin = SIM_STATE.get('margin_requirements', {}).get(bank_name, 0)
        attrs = SIM_STATE.get('bank_attrs', {}).get(bank_name, {})
        return jsonify({
            'bank': bank_name,
            'margin_B': round(margin, 3),
            'margin_ratio': margin / max(attrs.get('Total_Assets', 1), 1),
            'model_used': False,
            'reason': 'ML model not available'
        })
    
    # Get detailed decision from ML model
    details = predictor.get_margin_decision_details(
        SIM_STATE['bank_attrs'],
        bank_name,
        set(),  # No failed banks for initial assessment
        None
    )
    
    return jsonify(details)


@app.route('/api/ml/regenerate-margins', methods=['POST'])
def regenerate_margins():
    """
    Regenerate margin requirements using the ML model (or fallback).
    Optionally can specify to use ML or random margins.
    
    Body: { "use_ml": true }
    """
    data = request.get_json() or {}
    use_ml = data.get('use_ml', True)
    
    if use_ml and ML_MODEL_AVAILABLE:
        try:
            predictor = SIM_STATE.get('margin_predictor')
            if predictor is None:
                predictor = get_margin_predictor(
                    os.path.join(BACKEND_DIR, 'ML', 'ccp_policy.pt')
                )
                SIM_STATE['margin_predictor'] = predictor
            
            SIM_STATE['margin_requirements'] = predictor.generate_margin_requirements(
                SIM_STATE['bank_attrs']
            )
            SIM_STATE['use_ml_margins'] = True
            method = 'ML model'
        except Exception as e:
            SIM_STATE['margin_requirements'] = generate_margin_requirements(
                SIM_STATE['bank_attrs']
            )
            SIM_STATE['use_ml_margins'] = False
            method = f'random (ML failed: {e})'
    else:
        SIM_STATE['margin_requirements'] = generate_margin_requirements(
            SIM_STATE['bank_attrs']
        )
        SIM_STATE['use_ml_margins'] = False
        method = 'random'
    
    # Update contagion simulator with new margins
    SIM_STATE['contagion'].margin_requirements = SIM_STATE['margin_requirements'].copy()
    SIM_STATE['contagion'].initialize_states()
    
    return jsonify({
        'status': 'ok',
        'method': method,
        'ml_margins_enabled': SIM_STATE['use_ml_margins'],
        'total_margin_B': round(sum(SIM_STATE['margin_requirements'].values()), 2),
        'num_banks': len(SIM_STATE['margin_requirements'])
    })


@app.route('/api/ml/simulate-with-margins', methods=['POST'])
def simulate_with_custom_margins():
    """
    Run simulation with custom margin requirements.
    Allows testing different margin scenarios.
    
    Body: {
        "margins": {"JPM": 100.0, "BAC": 50.0, ...},  # Optional: custom margins
        "shock_type": "bank" | "stock",
        "target": "JPM" | {"AAPL": 30},  # Bank name or stock shocks
        "shock_pct": 50,  # For bank shocks
        "failure_threshold": 20
    }
    """
    data = request.get_json()
    
    shock_type = data.get('shock_type', 'bank')
    target = data.get('target')
    shock_pct = data.get('shock_pct', 50)
    threshold = data.get('failure_threshold', 20)
    custom_margins = data.get('margins')
    
    if target is None:
        return jsonify({'error': 'target is required'}), 400
    
    # Use custom margins if provided, otherwise use current
    if custom_margins:
        margins = custom_margins
    else:
        margins = SIM_STATE['margin_requirements']
    
    # Create temporary contagion simulator with specified margins
    temp_contagion = BankingNetworkContagion(
        SIM_STATE['graph'],
        SIM_STATE['stock_prices'],
        SIM_STATE['interbank_matrix'],
        margins
    )
    
    if shock_type == 'bank':
        if target not in SIM_STATE['bank_attrs']:
            return jsonify({'error': f'Bank {target} not found'}), 400
        result = temp_contagion.propagate_bank_shock(target, shock_pct, failure_threshold=threshold)
    elif shock_type == 'stock':
        if not isinstance(target, dict):
            return jsonify({'error': 'For stock shock, target must be a dict of {ticker: pct}'}), 400
        result = temp_contagion.propagate_stock_shock(target, failure_threshold=threshold)
    else:
        return jsonify({'error': f'Unknown shock_type: {shock_type}'}), 400
    
    network = _get_post_sim_network(temp_contagion)
    result['network'] = network
    result['margins_used'] = {k: round(v, 3) for k, v in margins.items()}
    result['total_margin_B'] = round(sum(margins.values()), 2)
    
    return jsonify(result)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_simulation()
    app.run(debug=True, port=5000)
