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
DATASET_DIR = os.path.join(BACKEND_DIR, 'dataset')

app = Flask(__name__)
CORS(app)

# ─── Global simulation state ─────────────────────────────────────────────────
SIM_STATE = {
    'bank_attrs': None,
    'graph': None,
    'stock_prices': None,
    'stock_timeseries': None,
    'all_stock_prices': None,      # All 965 stocks with latest prices
    'all_stock_timeseries': None,  # All 965 stocks time series
    'holdings': None,
    'contagion': None,
    'interbank_matrix': None,
    'network_simulator': None,
    'forecast_simulator': None,    # Forecast simulator for time-based projections
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


# ============================================================================
# FORECAST SIMULATOR
# ============================================================================
class ForecastSimulator:
    """
    Simulates shock impacts over time using historical stock data.
    Forecasts bank health for 1 month, 3 months, 6 months, and 1 year.
    """
    
    # Trading days approximation
    FORECAST_PERIODS = {
        '1_month': 21,      # ~21 trading days
        '3_months': 63,     # ~63 trading days  
        '6_months': 126,    # ~126 trading days
        '1_year': 252       # ~252 trading days
    }
    
    def __init__(self, network: NetworkSimulator, all_stock_timeseries: dict,
                 payoff_calculator: CCPPayoffCalculator = None):
        self.network = network
        self.all_stock_timeseries = all_stock_timeseries
        self.payoff_calculator = payoff_calculator or CCPPayoffCalculator()
        
        # Pre-compute stock statistics for forecasting
        self._compute_stock_statistics()
    
    def _compute_stock_statistics(self):
        """Compute historical volatility and trend for each stock."""
        self.stock_stats = {}
        
        for ticker, ts in self.all_stock_timeseries.items():
            if len(ts) < 10:
                continue
            
            # Sort by date
            sorted_ts = sorted(ts, key=lambda x: x['Date'])
            closes = [d['Close'] for d in sorted_ts]
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(closes)):
                if closes[i-1] > 0:
                    returns.append((closes[i] - closes[i-1]) / closes[i-1])
            
            if len(returns) < 5:
                continue
            
            # Calculate statistics
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(closes))
            coeffs = np.polyfit(x, closes, 1)
            trend_slope = coeffs[0] / closes[0]  # Normalize by initial price
            
            self.stock_stats[ticker] = {
                'mean_return': mean_return,
                'volatility': volatility,
                'trend_slope': trend_slope,
                'last_price': closes[-1],
                'first_price': closes[0],
                'total_return': (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
            }
    
    def _project_stock_prices(self, shock_tickers: dict, days_ahead: int, 
                              num_simulations: int = 100) -> dict:
        """
        Monte Carlo simulation to project stock prices forward.
        Returns distribution of prices for each stock.
        """
        projections = {}
        
        for ticker, shock_pct in shock_tickers.items():
            if ticker not in self.stock_stats:
                continue
            
            stats = self.stock_stats[ticker]
            initial_price = stats['last_price'] * (1 - shock_pct / 100.0)  # Apply shock
            
            # Run Monte Carlo simulations
            final_prices = []
            for _ in range(num_simulations):
                price = initial_price
                for _ in range(days_ahead):
                    # Geometric Brownian Motion with trend
                    daily_return = np.random.normal(stats['mean_return'], stats['volatility'])
                    price *= (1 + daily_return)
                    price = max(price, 0.01)  # Floor at $0.01
                final_prices.append(price)
            
            projections[ticker] = {
                'initial_price': stats['last_price'],
                'shocked_price': initial_price,
                'projected_mean': np.mean(final_prices),
                'projected_median': np.median(final_prices),
                'projected_p10': np.percentile(final_prices, 10),
                'projected_p90': np.percentile(final_prices, 90),
                'recovery_pct': (np.mean(final_prices) - initial_price) / initial_price * 100 if initial_price > 0 else 0
            }
        
        return projections
    
    def _simulate_bank_evolution(self, bank_name: str, shock_pct: float, 
                                 days_ahead: int, failure_threshold: float = 20.0) -> dict:
        """
        Simulate bank health evolution over time after a shock.
        Models gradual contagion and recovery dynamics.
        """
        # Reset network
        self.network.reset()
        state = self.network.get_state()
        
        if bank_name not in self.network.bank_list:
            return None
        
        # Apply initial shock
        initial_equity = state.equity.get(bank_name, 0)
        initial_assets = state.assets.get(bank_name, 0)
        
        shock_fraction = shock_pct / 100.0
        shock_amount = initial_assets * shock_fraction
        state.equity[bank_name] -= shock_amount
        state.assets[bank_name] -= shock_amount
        
        # Track health over time
        health_timeline = []
        failed_banks_timeline = []
        systemic_loss_timeline = []
        
        # Simulate day by day (with acceleration)
        step_size = max(1, days_ahead // 50)  # Max 50 steps
        current_day = 0
        
        while current_day <= days_ahead:
            # Calculate current state
            all_health = {}
            for bank in self.network.bank_list:
                all_health[bank] = self.network.get_bank_health(bank)
            
            # Check for failures
            for bank in self.network.bank_list:
                if bank not in state.failed_banks and all_health[bank] < failure_threshold:
                    state.failed_banks.add(bank)
            
            # Record state
            health_timeline.append({
                'day': current_day,
                'target_bank_health': all_health.get(bank_name, 0),
                'avg_system_health': np.mean(list(all_health.values())),
                'min_health': min(all_health.values()),
                'num_failed': len(state.failed_banks)
            })
            
            # Gradual contagion effects (small daily propagation)
            if len(state.failed_banks) > 0 and current_day > 0:
                # Daily contagion factor (small)
                daily_contagion = 0.001 * len(state.failed_banks)
                for bank in self.network.bank_list:
                    if bank not in state.failed_banks:
                        # Small daily loss from failed counterparties
                        counterparty_exposure = sum(
                            state.exposures.get(bank, {}).get(fb, 0) 
                            for fb in state.failed_banks
                        )
                        if counterparty_exposure > 0:
                            loss = counterparty_exposure * daily_contagion
                            loss = min(loss, state.assets[bank] * 0.001)  # Cap at 0.1%
                            state.assets[bank] -= loss
                            state.equity[bank] -= loss
            
            # Recovery dynamics (healthy banks slowly recover)
            if current_day > 0:
                for bank in self.network.bank_list:
                    if bank not in state.failed_banks:
                        # Daily recovery (0.01-0.05% based on health)
                        health = all_health.get(bank, 50)
                        if health > 40:  # Only healthy banks recover
                            recovery_rate = 0.0001 * (health / 100.0)
                            recovery = state.assets[bank] * recovery_rate
                            state.assets[bank] += recovery
                            state.equity[bank] += recovery
            
            current_day += step_size
        
        # Calculate final CCP payoff
        ccp_payoff = self.payoff_calculator.calculate_payoff(state.failed_banks, state)
        
        return {
            'health_timeline': health_timeline,
            'final_failed_banks': list(state.failed_banks),
            'ccp_payoff_B': round(ccp_payoff, 2)
        }
    
    def forecast_bank_shock(self, bank_name: str, shock_pct: float,
                            failure_threshold: float = 20.0) -> dict:
        """
        Forecast the full impact of a bank shock over all time horizons.
        """
        results = {
            'shock_type': 'BANK_SHOCK_FORECAST',
            'target_bank': bank_name,
            'shock_percent': shock_pct,
            'forecasts': {}
        }
        
        for period_name, days in self.FORECAST_PERIODS.items():
            forecast = self._simulate_bank_evolution(
                bank_name, shock_pct, days, failure_threshold
            )
            if forecast:
                results['forecasts'][period_name] = {
                    'days': days,
                    'period_label': period_name.replace('_', ' ').title(),
                    'final_target_health': forecast['health_timeline'][-1]['target_bank_health'] if forecast['health_timeline'] else 0,
                    'final_avg_health': forecast['health_timeline'][-1]['avg_system_health'] if forecast['health_timeline'] else 0,
                    'total_failures': len(forecast['final_failed_banks']),
                    'failed_banks': forecast['final_failed_banks'],
                    'ccp_payoff_B': forecast['ccp_payoff_B'],
                    'health_timeline': forecast['health_timeline']
                }
        
        return results
    
    def forecast_stock_shock(self, stock_shocks: dict, 
                             failure_threshold: float = 20.0) -> dict:
        """
        Forecast the full impact of stock shocks over all time horizons.
        Uses Monte Carlo to project stock recovery and bank health evolution.
        """
        results = {
            'shock_type': 'STOCK_SHOCK_FORECAST',
            'shocks': stock_shocks,
            'num_stocks_shocked': len(stock_shocks),
            'forecasts': {}
        }
        
        for period_name, days in self.FORECAST_PERIODS.items():
            # Project stock prices
            stock_projections = self._project_stock_prices(stock_shocks, days)
            
            # For stock shocks, we need to apply the shock and then simulate
            self.network.reset()
            state = self.network.get_state()
            
            # Apply initial stock shock impact to banks
            total_bank_losses = {}
            for ticker, shock_pct in stock_shocks.items():
                if ticker not in self.stock_stats:
                    continue
                
                stats = self.stock_stats[ticker]
                old_price = stats['last_price']
                new_price = old_price * (1 - shock_pct / 100.0)
                
                # Impact on all banks (systemic effect)
                for bank in self.network.bank_list:
                    if bank not in total_bank_losses:
                        total_bank_losses[bank] = 0
                    
                    # Systemic loss from market shock
                    market_correlation = 0.1  # 10% correlation to overall market
                    systemic_loss = state.assets[bank] * (shock_pct / 100.0) * market_correlation * 0.01
                    total_bank_losses[bank] += systemic_loss
            
            # Apply losses
            for bank, loss in total_bank_losses.items():
                state.assets[bank] -= loss
                state.equity[bank] -= loss
            
            # Check for immediate failures
            for bank in self.network.bank_list:
                if bank not in state.failed_banks:
                    if self.network.get_bank_health(bank) < failure_threshold:
                        state.failed_banks.add(bank)
            
            # Simulate evolution for this period
            health_timeline = []
            step_size = max(1, days // 30)
            current_day = 0
            
            while current_day <= days:
                all_health = {b: self.network.get_bank_health(b) for b in self.network.bank_list}
                
                health_timeline.append({
                    'day': current_day,
                    'avg_system_health': np.mean(list(all_health.values())),
                    'min_health': min(all_health.values()),
                    'num_failed': len(state.failed_banks)
                })
                
                # Small daily recovery based on projected stock recovery
                if current_day > 0:
                    avg_recovery = np.mean([
                        proj.get('recovery_pct', 0) / days
                        for proj in stock_projections.values()
                    ]) if stock_projections else 0
                    
                    for bank in self.network.bank_list:
                        if bank not in state.failed_banks:
                            health = all_health.get(bank, 50)
                            if health > 30:
                                recovery = state.assets[bank] * max(0, avg_recovery / 100) * 0.1
                                state.assets[bank] += recovery
                                state.equity[bank] += recovery
                
                current_day += step_size
            
            # Calculate CCP payoff
            ccp_payoff = self.payoff_calculator.calculate_payoff(state.failed_banks, state)
            
            results['forecasts'][period_name] = {
                'days': days,
                'period_label': period_name.replace('_', ' ').title(),
                'stock_projections': {
                    ticker: {
                        'initial': round(proj['initial_price'], 2),
                        'shocked': round(proj['shocked_price'], 2),
                        'projected_mean': round(proj['projected_mean'], 2),
                        'recovery_pct': round(proj['recovery_pct'], 2)
                    }
                    for ticker, proj in stock_projections.items()
                },
                'final_avg_health': health_timeline[-1]['avg_system_health'] if health_timeline else 0,
                'total_failures': len(state.failed_banks),
                'failed_banks': list(state.failed_banks),
                'ccp_payoff_B': round(ccp_payoff, 2),
                'health_timeline': health_timeline
            }
        
        return results


# ─── Data loading functions ───────────────────────────────────────────────────

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
    matrix = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        bank_names = header[1:]
        for row in reader:
            from_bank = row[0]
            matrix[from_bank] = {}
            for i, to_bank in enumerate(bank_names):
                matrix[from_bank][to_bank] = float(row[i + 1])
    return matrix


def load_all_stock_data(csv_path):
    """Load ALL stock data from CSV - returns all tickers with their prices and timeseries."""
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
    
    all_prices = {}
    all_timeseries = {}
    for ticker, records in all_data.items():
        ts = sorted(records, key=lambda x: x['Date'])
        all_prices[ticker] = ts[-1]['Close']
        all_timeseries[ticker] = ts
    
    return all_prices, all_timeseries


def load_stock_prices(csv_path, num_stocks=10, all_data=None):
    """Select a subset of stocks for simulation.
    If all_data is provided, use it instead of re-reading the CSV.
    """
    if all_data is None:
        all_prices, all_timeseries = load_all_stock_data(csv_path)
    else:
        all_prices, all_timeseries = all_data

    # Filter viable tickers (price >= 5.0)
    viable_tickers = [t for t in all_prices if all_prices[t] >= 5.0]
    selected_tickers = random.sample(viable_tickers, min(num_stocks, len(viable_tickers)))

    selected_prices = {ticker: all_prices[ticker] for ticker in selected_tickers}
    selected_timeseries = {ticker: all_timeseries[ticker] for ticker in selected_tickers}

    return selected_prices, selected_timeseries


def distribute_shares(bank_attributes, stock_prices):
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
    def __init__(self, graph, stock_prices=None, interbank_matrix=None, all_stock_prices=None):
        self.graph = graph
        self.stock_prices = stock_prices.copy() if stock_prices else {}
        # Store all stock prices for shock simulation (can shock any of 965 stocks)
        self.all_stock_prices = all_stock_prices.copy() if all_stock_prices else self.stock_prices.copy()
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
        
        # Track which shocks were applied (for response)
        applied_shocks = {}
        skipped_shocks = []

        for ticker, pct in stock_devaluations.items():
            # Check against ALL stock prices (not just the 10 selected)
            if ticker not in self.all_stock_prices:
                skipped_shocks.append(ticker)
                continue
            
            applied_shocks[ticker] = pct
            old_price = self.all_stock_prices[ticker]
            new_price = old_price * (1 - pct / 100.0)
            self.current_stock_prices[ticker] = new_price

            # Check if any banks hold this stock
            banks_affected = False
            for bank in self.graph:
                holdings = self.graph[bank].get('holdings', {})
                if ticker in holdings:
                    banks_affected = True
                    shares = holdings[ticker]
                    loss_b = shares * (old_price - new_price) / 1e9
                    if loss_b > 0:
                        self.bank_states[bank]['Total_Assets'] -= loss_b
                        self.bank_states[bank]['Equity'] -= loss_b
            
            # If no banks hold this stock directly, apply a smaller systemic shock
            # based on market correlation (stocks not held still affect market sentiment)
            if not banks_affected:
                # Systemic correlation factor (market-wide impact)
                market_impact_factor = (pct / 100.0) * 0.1  # 10% of shock as systemic impact
                for bank in self.graph:
                    total_assets = self.bank_states[bank]['Total_Assets']
                    systemic_loss = total_assets * market_impact_factor * 0.01  # Small % loss
                    self.bank_states[bank]['Total_Assets'] -= systemic_loss
                    self.bank_states[bank]['Equity'] -= systemic_loss

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
            'ccp_payoff_B': round(ccp_payoff, 2),
            'payoff_breakdown': payoff_breakdown,
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

    print("[INIT] Loading ALL stock data...")
    all_prices, all_ts = load_all_stock_data(
        os.path.join(DATASET_DIR, 'stocks_data_long.csv'))
    SIM_STATE['all_stock_prices'] = all_prices
    SIM_STATE['all_stock_timeseries'] = all_ts
    print(f"[INIT] Loaded {len(all_prices)} total stocks from dataset")

    print("[INIT] Selecting 10 random stocks for simulation...")
    prices, ts = load_stock_prices(
        os.path.join(DATASET_DIR, 'stocks_data_long.csv'), 
        num_stocks=10, 
        all_data=(all_prices, all_ts))
    SIM_STATE['stock_prices'] = prices
    SIM_STATE['stock_timeseries'] = ts

    print("[INIT] Distributing shares among banks...")
    SIM_STATE['holdings'] = distribute_shares(SIM_STATE['bank_attrs'], prices)

    print("[INIT] Building interbank graph...")
    SIM_STATE['graph'] = build_graph(SIM_STATE['bank_attrs'], SIM_STATE['interbank_matrix'])

    for bank in SIM_STATE['graph']:
        if bank in SIM_STATE['holdings']:
            SIM_STATE['graph'][bank]['holdings'] = SIM_STATE['holdings'][bank]

    SIM_STATE['contagion'] = BankingNetworkContagion(
        SIM_STATE['graph'], prices, SIM_STATE['interbank_matrix'], all_prices)

    # Also initialize NetworkSimulator for advanced shock simulation
    SIM_STATE['network_simulator'] = NetworkSimulator(
        os.path.join(DATASET_DIR, 'us_banks_top50_nodes_final.csv'),
        os.path.join(DATASET_DIR, 'stocks_data_long.csv'),
        os.path.join(DATASET_DIR, 'us_banks_interbank_matrix.csv')
    )
    
    # Initialize ForecastSimulator for time-based projections
    print("[INIT] Initializing forecast simulator...")
    SIM_STATE['forecast_simulator'] = ForecastSimulator(
        SIM_STATE['network_simulator'],
        all_ts
    )
    print(f"[INIT] Forecast simulator ready with {len(SIM_STATE['forecast_simulator'].stock_stats)} stock statistics")

    print(f"[INIT] Ready — {len(SIM_STATE['bank_attrs'])} banks, "
          f"{len(prices)} selected stocks, {len(all_prices)} total stocks available for shock")


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/banks', methods=['GET'])
def get_banks():
    """Return list of banks with attributes and health scores."""
    contagion = SIM_STATE['contagion']
    contagion.initialize_states()
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


@app.route('/api/all-stocks', methods=['GET'])
def get_all_stocks():
    """Return ALL available stocks (965) for stock shock selection.
    Only returns ticker and price (no timeseries to keep response small).
    """
    all_prices = SIM_STATE['all_stock_prices']
    
    stocks = []
    for ticker, price in all_prices.items():
        stocks.append({
            'ticker': ticker,
            'price': round(price, 2),
        })
    
    # Sort by ticker name for easier searching
    stocks.sort(key=lambda s: s['ticker'])
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


@app.route('/api/simulate/bank-shock-forecast', methods=['POST'])
def simulate_bank_shock_forecast():
    """
    Run a bank-level shock forecast simulation over multiple time horizons.
    Body: { "bank": "JPM", "shock_pct": 50, "failure_threshold": 20 }
    Returns forecasts for 1 month, 3 months, 6 months, and 1 year.
    """
    data = request.get_json()
    bank = data.get('bank')
    shock_pct = data.get('shock_pct', 50)
    threshold = data.get('failure_threshold', 20)

    if bank not in SIM_STATE['bank_attrs']:
        return jsonify({'error': f'Bank {bank} not found'}), 400
    
    if SIM_STATE['forecast_simulator'] is None:
        return jsonify({'error': 'Forecast simulator not initialized'}), 500

    result = SIM_STATE['forecast_simulator'].forecast_bank_shock(
        bank, shock_pct, failure_threshold=threshold
    )

    return jsonify(result)


@app.route('/api/simulate/stock-shock-forecast', methods=['POST'])
def simulate_stock_shock_forecast():
    """
    Run a stock-level shock forecast simulation over multiple time horizons.
    Body: { "shocks": {"AAPL": 30, "MSFT": 20}, "failure_threshold": 20 }
    Returns forecasts for 1 month, 3 months, 6 months, and 1 year.
    """
    data = request.get_json()
    shocks = data.get('shocks', {})
    threshold = data.get('failure_threshold', 20)

    if not shocks:
        return jsonify({'error': 'No stock shocks provided'}), 400
    
    if SIM_STATE['forecast_simulator'] is None:
        return jsonify({'error': 'Forecast simulator not initialized'}), 500

    result = SIM_STATE['forecast_simulator'].forecast_stock_shock(
        shocks, failure_threshold=threshold
    )

    return jsonify(result)


@app.route('/api/simulate/reset', methods=['POST'])
def reset_simulation():
    """Re-initialize the simulation with fresh random stock selection & holdings."""
    init_simulation()
    return jsonify({'status': 'ok', 'message': 'Simulation reset with new random stocks'})


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_simulation()
    app.run(debug=True, port=5000)
