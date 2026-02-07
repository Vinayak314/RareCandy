"""
================================================================================
MODEL_FINAL.PY - Unified Game-Theoretic CCP Risk Simulation
================================================================================
Merges MODEL 1 (RL-based CCP) and MODEL 2 (Contagion Simulation) into a
dynamic stochastic game where:

Players:
  - Banks: i = 1, 2, ..., N (strategic traders)
  - CCP: Single mechanism designer (sets margins, approves/rejects trades)

Game Type:
  - Dynamic stochastic game over discrete time t = 1, 2, ..., T
  - Incomplete information (banks don't fully observe others' balance sheets)
  - RL converges to approximate Markov Perfect Equilibrium

Architecture:
  - State Space: S_t = (E, L, A, X, G) from MODEL 2 simulation
  - Actions: Banks choose (trade_size, asset, counterparty)
           CCP chooses (margin, approve/reject)
  - Payoffs: Banks = profit - risk - margin_cost - default_penalty
            CCP = -systemic_loss - default_fund_loss + market_volume

================================================================================
"""

import numpy as np
import pandas as pd
import random
import pickle
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Time horizon
    'T': 100,  # Number of time steps per episode
    'NUM_EPISODES': 1000,  # Training episodes
    
    # RL parameters
    'GAMMA': 0.99,  # Discount factor
    'EPSILON_START': 1.0,
    'EPSILON_MIN': 0.05,
    'EPSILON_DECAY': 0.995,
    'MEMORY_SIZE': 5000,
    'BATCH_SIZE': 64,
    
    # Bank payoff weights
    'LAMBDA_RISK': 0.3,  # Risk aversion
    'MU_MARGIN': 0.1,    # Margin cost sensitivity
    'PHI_DEFAULT': 100.0, # Default penalty
    
    # CCP payoff weights
    'ALPHA_SYSTEMIC': 1.0,   # Systemic loss weight
    'BETA_DF_LOSS': 0.5,     # Default fund loss weight
    'GAMMA_VOLUME': 0.01,    # Market volume weight (encourage trading)
    
    # Market parameters
    'FUNDING_RATE': 0.05,  # Cost of locked margin capital
    'CRASH_SEVERITY': 3.5,  # Sigma for worst-case scenarios
    'SHOCK_RANGE': (0.05, 0.25),  # Random shock range
    
    # Model paths
    'BANK_MODEL_PATH': 'bank_policies.pkl',
    'CCP_MODEL_PATH': 'ccp_policy.pkl',
}


class TradeDecision(Enum):
    APPROVED = "APPROVED"
    REQUIRE_MARGIN = "REQUIRE_MARGIN"
    REJECTED = "REJECTED"


@dataclass
class BankAction:
    """Bank's action at time t"""
    trade_size: float  # q_i(t) - size in billions
    asset: str  # Which asset to trade
    counterparty: str  # 'CCP' or specific bank name
    direction: str  # 'BUY' or 'SELL'


@dataclass
class CCPAction:
    """CCP's action for a bank at time t"""
    margin_requirement: float  # m_i(t) - margin ratio (0-1)
    decision: TradeDecision  # approve/reject/require_margin


@dataclass
class SystemState:
    """Global system state S_t = (E, L, A, X, G)"""
    equity: Dict[str, float]  # E_i(t) for each bank
    liquidity: Dict[str, float]  # L_i(t) - HQLA
    assets: Dict[str, float]  # A_i(t) - total assets
    exposures: Dict[str, Dict[str, float]]  # X_ij(t) - interbank exposures
    liabilities: Dict[str, float]  # Total liabilities
    stock_volatility: Dict[str, float]  # Per-stock volatility
    failed_banks: Set[str] = field(default_factory=set)


# ============================================================================
# NETWORK SIMULATOR (MODEL 2 - State Evolution)
# ============================================================================
class NetworkSimulator:
    """
    Simulates the banking network state evolution.
    This is MODEL 2 - provides state to MODEL 1's RL agents.
    """
    
    def __init__(self, banks_file: str, stocks_file: str, matrix_file: str):
        # Load bank data
        self.banks_df = pd.read_csv(banks_file)
        self.interbank_matrix = pd.read_csv(matrix_file, index_col=0)
        
        # Calculate stock volatility
        stocks_df = pd.read_csv(stocks_file)
        stocks_df['Return'] = stocks_df.groupby('Ticker')['Close'].pct_change()
        vol_df = stocks_df.groupby('Ticker')['Return'].std().reset_index()
        vol_df.columns = ['Ticker', 'Volatility']
        self.stock_volatility = vol_df.set_index('Ticker')['Volatility'].to_dict()
        
        # Initialize bank attributes
        self.bank_list = self.banks_df['Bank'].tolist()
        self.initial_state = self._create_initial_state()
        self.current_state = None
        self.reset()
        
        # Feature scaler for RL
        self.scaler = StandardScaler()
        features = ['Total_Assets', 'Equity', 'HQLA', 'Net_Outflows_30d', 'Interbank_Liabilities']
        self.scaler.fit(self.banks_df[features].fillna(0))
        
        print(f"✅ NetworkSimulator loaded: {len(self.bank_list)} banks, "
              f"{len(self.stock_volatility)} stocks")
    
    def _create_initial_state(self) -> SystemState:
        """Create initial system state from data"""
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
            
            # Build exposure matrix
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
        """Reset to initial state"""
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
        """Get current system state"""
        return self.current_state
    
    def get_bank_features(self, bank: str) -> np.ndarray:
        """Get normalized feature vector for a bank (for RL)"""
        state = self.current_state
        features = np.array([
            state.assets.get(bank, 0),
            state.equity.get(bank, 0),
            state.liquidity.get(bank, 0),
            self.banks_df[self.banks_df['Bank'] == bank]['Net_Outflows_30d'].values[0],
            sum(state.exposures.get(bank, {}).values())
        ]).reshape(1, -1)
        return self.scaler.transform(features).flatten()
    
    def get_bank_health(self, bank: str) -> float:
        """Calculate bank health score (0-100)"""
        state = self.current_state
        if bank in state.failed_banks:
            return 0.0
        
        assets = state.assets.get(bank, 0)
        equity = state.equity.get(bank, 0)
        
        if assets <= 0:
            return 0.0
        
        # Capital adequacy (0-40 points)
        equity_ratio = equity / assets
        capital_score = min(40.0, (equity_ratio / 0.15) * 40.0)
        
        # Liquidity (0-30 points)
        lcr = state.liquidity.get(bank, 0) / max(1, state.liabilities.get(bank, 1) * 0.1)
        liquidity_score = min(30.0, (lcr / 1.5) * 30.0)
        
        # Risk penalty (0-25 points)
        bank_row = self.banks_df[self.banks_df['Bank'] == bank]
        if not bank_row.empty:
            cds = bank_row['Est_CDS_Spread'].values[0]
            vol = bank_row['Stock_Volatility'].values[0]
            risk_penalty = min(15.0, cds / 400.0 * 15.0) + min(10.0, vol / 0.5 * 10.0)
        else:
            risk_penalty = 10.0
        
        health = capital_score + liquidity_score - risk_penalty
        return max(0.0, min(100.0, health))
    
    def apply_trade(self, bank: str, action: BankAction, ccp_action: CCPAction) -> float:
        """
        Apply a trade and return the immediate profit/loss.
        Modifies current_state.
        """
        if ccp_action.decision == TradeDecision.REJECTED:
            return 0.0  # No trade executed
        
        state = self.current_state
        trade_size = action.trade_size
        
        # Get asset volatility
        vol = state.stock_volatility.get(action.asset, 0.02)
        
        # Simulate price movement (random walk)
        price_change = np.random.normal(0, vol)  # Daily return
        
        # Calculate profit
        if action.direction == 'BUY':
            profit = trade_size * price_change
            # Reduce liquidity by trade size (use HQLA to buy)
            state.liquidity[bank] = max(0, state.liquidity[bank] - trade_size)
        else:  # SELL
            profit = -trade_size * price_change
            # Increase liquidity
            state.liquidity[bank] += trade_size * 0.95  # 5% haircut
        
        # Update assets and equity with profit
        state.assets[bank] += profit
        state.equity[bank] += profit
        
        return profit
    
    def apply_market_shock(self, shock_pct: float = None):
        """Apply random market shock to all banks"""
        if shock_pct is None:
            shock_pct = random.uniform(*CONFIG['SHOCK_RANGE'])
        
        state = self.current_state
        for bank in self.bank_list:
            if bank not in state.failed_banks:
                loss = state.assets[bank] * shock_pct * random.uniform(0.5, 1.5)
                state.assets[bank] -= loss
                state.equity[bank] -= loss
    
    def run_contagion(self, failure_threshold: float = 20.0) -> Dict:
        """
        Run contagion cascade (MODEL 2 core logic).
        Returns cascade results.
        """
        state = self.current_state
        initial_failed = state.failed_banks.copy()
        
        # Check for new failures
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
                # Get interbank exposure
                exposures = state.exposures.get(failed_bank, {})
                
                for creditor, exposure in exposures.items():
                    if creditor not in state.failed_banks and exposure > 0:
                        # Loss transmission
                        loss = exposure * dampening * 0.5
                        loss = min(loss, state.assets[creditor] * 0.15)
                        
                        state.assets[creditor] -= loss
                        state.equity[creditor] -= loss
                        
                        if self.get_bank_health(creditor) < failure_threshold:
                            state.failed_banks.add(creditor)
                            newly_failed.add(creditor)
        
        # Calculate systemic loss
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
# BANK AGENT (MODEL 1 - RL Policy)
# ============================================================================
class BankAgent:
    """
    RL agent for a single bank.
    Learns policy π_i(a_i | S_t) to maximize utility.
    """
    
    def __init__(self, bank_name: str, state_size: int = 8, action_size: int = 9):
        self.bank_name = bank_name
        self.state_size = state_size
        self.action_size = action_size  # 3 trade sizes x 3 directions/assets
        
        # Q-network
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1,
            warm_start=True,
            learning_rate_init=0.001
        )
        self._init_model()
        
        # Experience replay
        self.memory = deque(maxlen=CONFIG['MEMORY_SIZE'])
        self.epsilon = CONFIG['EPSILON_START']
        
        # Action mapping
        self.trade_sizes = [0.01, 0.05, 0.10]  # Fraction of HQLA
        self.directions = ['BUY', 'SELL', 'HOLD']
    
    def _init_model(self):
        """Initialize with dummy data"""
        X = np.zeros((1, self.state_size))
        y = np.zeros((1, self.action_size))
        self.model.fit(X, y)
    
    def get_action(self, state_vec: np.ndarray, hqla: float, 
                   available_assets: List[str], explore: bool = True) -> BankAction:
        """Choose action using epsilon-greedy policy"""
        
        if explore and random.random() < self.epsilon:
            # Random action
            size_idx = random.randint(0, 2)
            dir_idx = random.randint(0, 2)
        else:
            # Greedy action
            q_values = self.model.predict(state_vec.reshape(1, -1))[0]
            best_action = np.argmax(q_values)
            size_idx = best_action // 3
            dir_idx = best_action % 3
        
        direction = self.directions[dir_idx]
        
        if direction == 'HOLD':
            return BankAction(
                trade_size=0,
                asset=random.choice(available_assets) if available_assets else 'NONE',
                counterparty='CCP',
                direction='HOLD'
            )
        
        trade_size = hqla * self.trade_sizes[size_idx]
        asset = random.choice(available_assets) if available_assets else 'AAPL'
        
        return BankAction(
            trade_size=trade_size,
            asset=asset,
            counterparty='CCP',
            direction=direction
        )
    
    def remember(self, state, action_idx, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self, batch_size: int = None):
        """Train on batch"""
        if batch_size is None:
            batch_size = CONFIG['BATCH_SIZE']
        
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        X = np.array([s for s, _, _, _, _ in batch])
        
        # Get current Q-values
        current_q = self.model.predict(X)
        
        # Update Q-values with rewards
        for i, (state, action_idx, reward, next_state, done) in enumerate(batch):
            if done:
                target = reward
            else:
                next_q = self.model.predict(next_state.reshape(1, -1))[0]
                target = reward + CONFIG['GAMMA'] * np.max(next_q)
            
            current_q[i, action_idx] = target
        
        self.model.fit(X, current_q)
        
        # Decay epsilon
        if self.epsilon > CONFIG['EPSILON_MIN']:
            self.epsilon *= CONFIG['EPSILON_DECAY']


# ============================================================================
# CCP AGENT (MODEL 1 - Mechanism Designer)
# ============================================================================
class CCPAgent:
    """
    CCP agent that sets margins and approves/rejects trades.
    Optimizes for systemic stability while allowing market activity.
    """
    
    def __init__(self, state_size: int = 10, num_banks: int = 50):
        self.state_size = state_size
        self.num_banks = num_banks
        
        # Margin levels (must be defined before _init_model)
        self.margin_levels = [0.05, 0.10, 0.20, 0.30, 0.50]
        
        # Thresholds
        self.approve_threshold = 0.5  # B$ systemic loss
        self.margin_threshold = 5.0   # B$ systemic loss
        
        self.memory = deque(maxlen=CONFIG['MEMORY_SIZE'])
        self.epsilon = CONFIG['EPSILON_START']
        
        # Q-network for margin policy
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1,
            warm_start=True,
            learning_rate_init=0.001
        )
        self._init_model()
    
    def _init_model(self):
        X = np.zeros((1, self.state_size))
        y = np.zeros((1, len(self.margin_levels) + 1))  # margins + reject
        self.model.fit(X, y)
    
    def get_action(self, bank_state: np.ndarray, bank_action: BankAction,
                   predicted_risk: float, explore: bool = True) -> CCPAction:
        """Decide on margin and approval for a trade"""
        
        # Build input features
        features = np.concatenate([
            bank_state,
            [bank_action.trade_size, predicted_risk, 
             1.0 if bank_action.direction == 'BUY' else 0.0]
        ])
        
        if explore and random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.margin_levels))
        else:
            q_values = self.model.predict(features.reshape(1, -1))[0]
            action_idx = np.argmax(q_values)
        
        if action_idx == len(self.margin_levels):
            # Reject
            return CCPAction(margin_requirement=1.0, decision=TradeDecision.REJECTED)
        
        margin = self.margin_levels[action_idx]
        
        # Determine decision based on predicted risk
        if predicted_risk < self.approve_threshold:
            decision = TradeDecision.APPROVED
        elif predicted_risk < self.margin_threshold:
            decision = TradeDecision.REQUIRE_MARGIN
        else:
            decision = TradeDecision.REJECTED
        
        return CCPAction(margin_requirement=margin, decision=decision)
    
    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self, batch_size: int = None):
        if batch_size is None:
            batch_size = CONFIG['BATCH_SIZE']
        
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        X = np.array([s for s, _, _, _, _ in batch])
        current_q = self.model.predict(X)
        
        for i, (state, action_idx, reward, next_state, done) in enumerate(batch):
            if done:
                target = reward
            else:
                next_q = self.model.predict(next_state.reshape(1, -1))[0]
                target = reward + CONFIG['GAMMA'] * np.max(next_q)
            current_q[i, action_idx] = target
        
        self.model.fit(X, current_q)
        
        if self.epsilon > CONFIG['EPSILON_MIN']:
            self.epsilon *= CONFIG['EPSILON_DECAY']


# ============================================================================
# PAYOFF FUNCTIONS
# ============================================================================
def compute_bank_payoff(bank: str, profit: float, state_before: SystemState,
                        state_after: SystemState, margin: float) -> float:
    """
    Bank i's utility:
    U_i(t) = E[Π_i(t)] - λ·Risk_i(t) - μ·MarginCost_i(t) - φ·1_default
    """
    # 1. Trading profit
    expected_profit = profit
    
    # 2. Risk term: P(E_i(t+1) < 0 | S_t, a_i(t))
    equity_before = state_before.equity.get(bank, 0)
    equity_after = state_after.equity.get(bank, 0)
    
    if equity_before > 0:
        risk = max(0, -equity_after) / equity_before
    else:
        risk = 1.0
    
    # 3. Margin cost: r · m_i(t)
    margin_cost = CONFIG['FUNDING_RATE'] * margin
    
    # 4. Default penalty
    default_penalty = 1.0 if bank in state_after.failed_banks else 0.0
    
    # Combine
    utility = (
        expected_profit
        - CONFIG['LAMBDA_RISK'] * risk
        - CONFIG['MU_MARGIN'] * margin_cost
        - CONFIG['PHI_DEFAULT'] * default_penalty
    )
    
    return utility


def compute_ccp_payoff(state_before: SystemState, state_after: SystemState,
                       margins: Dict[str, float], trade_volumes: Dict[str, float]) -> float:
    """
    CCP's utility:
    U_CCP(t) = -α·SystemicLoss(t) - β·DefaultFundLoss(t) + γ·MarketVolume(t)
    """
    # 1. Systemic loss
    systemic_loss = sum(
        max(0, state_after.liabilities.get(b, 0) - state_after.assets.get(b, 0))
        for b in state_after.failed_banks
    )
    
    # 2. Default fund loss (losses beyond margin)
    df_loss = 0
    for bank in state_after.failed_banks:
        exposure = abs(trade_volumes.get(bank, 0))
        margin_held = exposure * margins.get(bank, 0)
        actual_loss = max(0, exposure - margin_held)
        df_loss += actual_loss
    
    # 3. Market volume
    market_volume = sum(abs(v) for v in trade_volumes.values())
    
    # Combine
    utility = (
        - CONFIG['ALPHA_SYSTEMIC'] * systemic_loss
        - CONFIG['BETA_DF_LOSS'] * df_loss
        + CONFIG['GAMMA_VOLUME'] * market_volume
    )
    
    return utility


# ============================================================================
# MAIN GAME LOOP
# ============================================================================
class CCPGame:
    """
    Main game controller that runs the simulation.
    Merges MODEL 1 (RL) and MODEL 2 (Network Simulation).
    """
    
    def __init__(self, banks_file: str, stocks_file: str, matrix_file: str):
        # Initialize network simulator (MODEL 2)
        self.network = NetworkSimulator(banks_file, stocks_file, matrix_file)
        
        # Initialize bank agents (MODEL 1)
        self.bank_agents = {
            bank: BankAgent(bank) for bank in self.network.bank_list
        }
        
        # Initialize CCP agent (MODEL 1)
        self.ccp_agent = CCPAgent(num_banks=len(self.network.bank_list))
        
        # Get available assets (top 20 by volume)
        self.available_assets = list(self.network.stock_volatility.keys())[:20]
        
        # Metrics
        self.episode_rewards = []
        self.systemic_losses = []
        
        print(f"✅ CCPGame initialized with {len(self.bank_agents)} banks")
    
    def run_episode(self, T: int = None, train: bool = True) -> Dict:
        """Run one episode of the game"""
        if T is None:
            T = CONFIG['T']
        
        # Reset state
        state = self.network.reset()
        
        episode_bank_rewards = {b: 0 for b in self.bank_agents}
        episode_ccp_reward = 0
        total_trades = 0
        total_volume = 0
        
        for t in range(T):
            # Store state before actions
            state_before = SystemState(
                equity=state.equity.copy(),
                liquidity=state.liquidity.copy(),
                assets=state.assets.copy(),
                exposures={b: e.copy() for b, e in state.exposures.items()},
                liabilities=state.liabilities.copy(),
                stock_volatility=state.stock_volatility.copy(),
                failed_banks=state.failed_banks.copy()
            )
            
            margins = {}
            trade_volumes = {}
            bank_profits = {}
            
            # ===== BANK ACTIONS =====
            for bank in self.network.bank_list:
                if bank in state.failed_banks:
                    continue
                
                # Get bank state features
                bank_features = self.network.get_bank_features(bank)
                hqla = state.liquidity.get(bank, 0)
                
                # Bank chooses action
                bank_action = self.bank_agents[bank].get_action(
                    bank_features, hqla, self.available_assets, explore=train
                )
                
                if bank_action.direction == 'HOLD' or bank_action.trade_size == 0:
                    continue
                
                # ===== CCP EVALUATES =====
                # Predict risk using simple heuristic (or could use RL prediction)
                vol = state.stock_volatility.get(bank_action.asset, 0.02)
                predicted_risk = bank_action.trade_size * vol * CONFIG['CRASH_SEVERITY']
                
                ccp_action = self.ccp_agent.get_action(
                    bank_features, bank_action, predicted_risk, explore=train
                )
                
                margins[bank] = ccp_action.margin_requirement
                
                # ===== EXECUTE TRADE =====
                profit = self.network.apply_trade(bank, bank_action, ccp_action)
                
                if ccp_action.decision != TradeDecision.REJECTED:
                    trade_volumes[bank] = bank_action.trade_size
                    bank_profits[bank] = profit
                    total_trades += 1
                    total_volume += bank_action.trade_size
            
            # ===== MARKET SHOCK (random) =====
            if random.random() < 0.1:  # 10% chance of shock each step
                self.network.apply_market_shock()
            
            # ===== CONTAGION CASCADE (MODEL 2) =====
            cascade_result = self.network.run_contagion()
            
            # Get state after
            state = self.network.get_state()
            
            # ===== COMPUTE REWARDS =====
            # Bank rewards
            for bank in bank_profits:
                reward = compute_bank_payoff(
                    bank, bank_profits[bank], state_before, state, margins.get(bank, 0)
                )
                episode_bank_rewards[bank] += reward
                
                # Store experience for RL
                if train:
                    bank_features_before = self.network.get_bank_features(bank)
                    bank_features_after = self.network.get_bank_features(bank)
                    action_idx = 0  # Simplified
                    done = bank in state.failed_banks
                    self.bank_agents[bank].remember(
                        bank_features_before, action_idx, reward, bank_features_after, done
                    )
            
            # CCP reward
            ccp_reward = compute_ccp_payoff(state_before, state, margins, trade_volumes)
            episode_ccp_reward += ccp_reward
            
            if train:
                # Train CCP
                ccp_state = np.mean([
                    self.network.get_bank_features(b) for b in self.network.bank_list
                    if b not in state.failed_banks
                ], axis=0) if len(state.failed_banks) < len(self.network.bank_list) else np.zeros(8)
                
                self.ccp_agent.remember(ccp_state, 0, ccp_reward, ccp_state, False)
            
            # Check for system collapse
            if len(state.failed_banks) > len(self.network.bank_list) * 0.5:
                break  # End episode early
        
        # ===== EXPERIENCE REPLAY (TRAINING) =====
        if train:
            for bank_agent in self.bank_agents.values():
                bank_agent.replay()
            self.ccp_agent.replay()
        
        # Return episode metrics
        return {
            'total_trades': total_trades,
            'total_volume': total_volume,
            'failed_banks': len(state.failed_banks),
            'systemic_loss': sum(max(0, state.liabilities.get(b, 0) - state.assets.get(b, 0))
                                 for b in state.failed_banks),
            'avg_bank_reward': np.mean(list(episode_bank_rewards.values())),
            'ccp_reward': episode_ccp_reward
        }
    
    def train(self, num_episodes: int = None):
        """Train all agents"""
        if num_episodes is None:
            num_episodes = CONFIG['NUM_EPISODES']
        
        print(f"\n🚀 Training for {num_episodes} episodes...")
        print("=" * 60)
        
        for episode in range(num_episodes):
            result = self.run_episode(train=True)
            
            self.episode_rewards.append(result['avg_bank_reward'])
            self.systemic_losses.append(result['systemic_loss'])
            
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_loss = np.mean(self.systemic_losses[-50:])
                eps = self.bank_agents[self.network.bank_list[0]].epsilon
                print(f"Episode {episode:4d} | Trades: {result['total_trades']:3d} | "
                      f"Failed: {result['failed_banks']:2d} | Loss: ${avg_loss:.1f}B | "
                      f"ε: {eps:.3f}")
        
        print("=" * 60)
        print("✅ Training complete!")
        self.save_models()
    
    def save_models(self):
        """Save all trained models"""
        bank_policies = {b: agent.model for b, agent in self.bank_agents.items()}
        with open(CONFIG['BANK_MODEL_PATH'], 'wb') as f:
            pickle.dump(bank_policies, f)
        
        with open(CONFIG['CCP_MODEL_PATH'], 'wb') as f:
            pickle.dump(self.ccp_agent.model, f)
        
        print(f"✅ Models saved to {CONFIG['BANK_MODEL_PATH']} and {CONFIG['CCP_MODEL_PATH']}")
    
    def load_models(self):
        """Load trained models"""
        if os.path.exists(CONFIG['BANK_MODEL_PATH']):
            with open(CONFIG['BANK_MODEL_PATH'], 'rb') as f:
                bank_policies = pickle.load(f)
            for b, model in bank_policies.items():
                if b in self.bank_agents:
                    self.bank_agents[b].model = model
            print(f"✅ Loaded bank policies from {CONFIG['BANK_MODEL_PATH']}")
        
        if os.path.exists(CONFIG['CCP_MODEL_PATH']):
            with open(CONFIG['CCP_MODEL_PATH'], 'rb') as f:
                self.ccp_agent.model = pickle.load(f)
            print(f"✅ Loaded CCP policy from {CONFIG['CCP_MODEL_PATH']}")
    
    def evaluate_trade(self, bank: str, asset: str, size: float, direction: str) -> Dict:
        """
        Evaluate a single trade proposal (for live use).
        Returns CCP decision and risk assessment.
        """
        state = self.network.get_state()
        bank_features = self.network.get_bank_features(bank)
        
        action = BankAction(
            trade_size=size,
            asset=asset,
            counterparty='CCP',
            direction=direction
        )
        
        vol = state.stock_volatility.get(asset, 0.02)
        predicted_risk = size * vol * CONFIG['CRASH_SEVERITY']
        
        ccp_action = self.ccp_agent.get_action(
            bank_features, action, predicted_risk, explore=False
        )
        
        return {
            'bank': bank,
            'asset': asset,
            'size': size,
            'direction': direction,
            'decision': ccp_action.decision.value,
            'margin_requirement': ccp_action.margin_requirement,
            'predicted_risk': predicted_risk,
            'bank_health': self.network.get_bank_health(bank)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("CCP GAME-THEORETIC RISK SIMULATION")
    print("Unified MODEL 1 (RL) + MODEL 2 (Contagion)")
    print("=" * 70)
    
    # Initialize game
    game = CCPGame(
        banks_file='phase1_engineered_data.csv',
        stocks_file='stocks_data_long.csv',
        matrix_file='us_banks_interbank_matrix.csv'
    )
    
    # Check for existing models
    if os.path.exists(CONFIG['BANK_MODEL_PATH']) and os.path.exists(CONFIG['CCP_MODEL_PATH']):
        print("\n📂 Found existing models. Loading...")
        game.load_models()
        
        # Quick evaluation
        print("\n🔍 Running evaluation episode...")
        result = game.run_episode(T=50, train=False)
        print(f"  Trades: {result['total_trades']} | Failed: {result['failed_banks']} | "
              f"Loss: ${result['systemic_loss']:.2f}B")
    else:
        # Train from scratch
        game.train(num_episodes=200)  # Reduced for demo
    
    # Demo: Evaluate sample trades
    print("\n" + "=" * 70)
    print("SAMPLE TRADE EVALUATIONS")
    print("=" * 70)
    
    test_trades = [
        ('JPM', 'AAPL', 5.0, 'BUY'),
        ('GS', 'NVDA', 50.0, 'BUY'),
        ('IBOC', 'SMCI', 2.0, 'BUY'),
        ('BAC', 'MSFT', 100.0, 'SELL'),
    ]
    
    for bank, asset, size, direction in test_trades:
        result = game.evaluate_trade(bank, asset, size, direction)
        symbol = {'APPROVED': '✅', 'REQUIRE_MARGIN': '⚠️', 'REJECTED': '❌'}
        print(f"\n{symbol.get(result['decision'], '?')} {bank} {direction} ${size}B of {asset}")
        print(f"   Decision: {result['decision']}")
        print(f"   Margin: {result['margin_requirement']*100:.0f}%")
        print(f"   Predicted Risk: ${result['predicted_risk']:.2f}B")
        print(f"   Bank Health: {result['bank_health']:.1f}/100")
    
    print("\n" + "=" * 70)
    print("✅ MODEL_FINAL.PY READY")
    print("   - Use CCPGame for full simulation")
    print("   - Use game.evaluate_trade() for live CCP checks")
    print("=" * 70)
