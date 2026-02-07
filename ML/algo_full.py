"""
================================================================================
ALGO.PY - Reinforcement Learning CCP & Bank Risk Management
================================================================================
Integrates with train.py's BankingNetworkContagion simulation to train:
  1. Bank Agents: Learn optimal trading/risk decisions
  2. CCP Agent: Learns optimal margin requirements to minimize systemic risk

Uses the existing network graph, stock prices, holdings, and margin system.

Supports:
  - GPU acceleration via PyTorch (auto-detected)
  - Multi-threaded experience collection
  - Parallel episode execution
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
import pickle
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading

# Try to import PyTorch for GPU support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    
    # Auto-detect GPU
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')  # Apple Silicon
        print("🚀 Apple MPS (Metal) detected")
    else:
        DEVICE = torch.device('cpu')
        print("💻 Using CPU (no GPU detected)")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not found - using sklearn (slower, CPU only)")
    print("   Install with: pip install torch")

# Fallback to sklearn if PyTorch not available
if not TORCH_AVAILABLE:
    from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Import from algorithm.py (make sure algorithm.py is in same directory)
from algorithm import (
    load_bank_attributes,
    load_stock_prices,
    distribute_shares,
    generate_margin_requirements,
    generate_random_graph_with_sccs,
    BankingNetworkContagion
)


# ============================================================================
# CONFIGURATION
# ============================================================================
RL_CONFIG = {
    # Training parameters
    'NUM_EPISODES': 500,
    'STEPS_PER_EPISODE': 50,
    
    # Parallelization
    'NUM_WORKERS': min(4, mp.cpu_count()),  # Parallel episode collectors
    'USE_GPU': TORCH_AVAILABLE and DEVICE is not None and DEVICE.type != 'cpu',
    
    # RL hyperparameters
    'GAMMA': 0.95,           # Discount factor
    'EPSILON_START': 1.0,
    'EPSILON_MIN': 0.05,
    'EPSILON_DECAY': 0.995,
    'MEMORY_SIZE': 10000,
    'BATCH_SIZE': 64 if not TORCH_AVAILABLE else 128,  # Larger batch for GPU
    'LEARNING_RATE': 0.001,
    
    # Reward weights for BANKS
    'REWARD_PROFIT': 1.0,           # Profit from trading
    'REWARD_HEALTH_BONUS': 0.5,     # Bonus for maintaining health
    'PENALTY_HEALTH_LOSS': 2.0,     # Penalty for health degradation
    'PENALTY_FAILURE': 100.0,       # Large penalty for bank failure
    'PENALTY_MARGIN_COST': 0.3,     # Cost of locked margin
    
    # Reward weights for CCP
    'CCP_PENALTY_SYSTEMIC': 10.0,   # Penalty for systemic loss
    'CCP_PENALTY_FAILURES': 5.0,    # Penalty per failed bank
    'CCP_REWARD_STABILITY': 2.0,    # Reward for system stability
    'CCP_REWARD_VOLUME': 0.1,       # Small reward for enabling trades
    'CCP_PENALTY_OVER_MARGIN': 0.5, # Penalty for margins that are too high
    
    # Market shock parameters
    'SHOCK_PROBABILITY': 0.1,       # 10% chance of market shock per step
    'SHOCK_RANGE': (0.05, 0.25),    # 5-25% shock magnitude
    
    # Failure threshold
    'FAILURE_THRESHOLD': 20.0,
    
    # Model save paths
    'BANK_MODEL_PATH': 'models/bank_policies.pkl',
    'CCP_MODEL_PATH': 'models/ccp_policy.pkl',
}


class TradeDecision(Enum):
    APPROVED = "APPROVED"
    REQUIRE_MARGIN = "REQUIRE_MARGIN"
    REJECTED = "REJECTED"


@dataclass
class BankAction:
    """Bank's action at time t"""
    trade_size: float      # Size in billions (fraction of HQLA)
    asset: str             # Stock ticker to trade
    direction: str         # 'BUY', 'SELL', or 'HOLD'


@dataclass
class CCPAction:
    """CCP's action for a bank trade"""
    margin_multiplier: float  # Multiplier on base margin (0.5x to 3x)
    decision: TradeDecision   # Approve/reject/require margin


# ============================================================================
# LEGACY SCORING FUNCTIONS (kept for backward compatibility)
# ============================================================================
def calcScore(bank):
    """
    Calculate bank risk score (legacy function).
    Higher score = higher risk.
    """
    # Handle both dict and object-style access
    if hasattr(bank, 'asset'):
        asset = bank.asset
        equity = bank.equity
        hqla = bank.hqla
        net_cash = bank.net_cash_outflows_30d
        cds = bank.cds_spread
        stock_vol = bank.stock_volatility
        interbank = getattr(bank, 'interbank_borrowing', 0)
        correlation = getattr(bank, 'asset_correlation', 0.05)
    else:
        asset = bank.get('Total_Assets', 0)
        equity = bank.get('Equity', 1)
        hqla = bank.get('HQLA', 0)
        net_cash = bank.get('Net_Outflows_30d', 1)
        cds = bank.get('Est_CDS_Spread', 100)
        stock_vol = bank.get('Stock_Volatility', 0.3)
        interbank = bank.get('Interbank_Liabilities', 0)
        correlation = 0.05
    
    LR = asset / max(equity, 1)  # Leverage Ratio
    LCR = hqla / max(net_cash, 1)  # Liquidity Coverage Ratio
    
    CDS_MAX, CDS_MIN = 488.98, 62.57
    VOL_MAX, VOL_MIN = 0.64, 0.01
    
    LR_Score = min(LR / 20, 1)  # Normalized
    LCR_Score = min(1 / max(LCR, 0.1), 1)  # Inverse (lower LCR = higher risk)
    CDS_Score = (cds - CDS_MIN) / (CDS_MAX - CDS_MIN)
    Vol_Score = (stock_vol - VOL_MIN) / (VOL_MAX - VOL_MIN)
    
    # Weighted score (higher = riskier)
    score = (
        0.25 * LR_Score +
        0.20 * LCR_Score +
        0.25 * CDS_Score +
        0.15 * Vol_Score +
        0.15 * (interbank / max(asset, 1))
    )
    
    return min(max(score, 0), 1)  # Clamp to [0, 1]


def updateState(bankList, state):
    """Update state ranking based on bank scores."""
    bankRank = []
    for x in bankList:
        bankRank.append((calcScore(x), x))
    bankRank.sort(key=lambda x: x[0], reverse=True)
    return bankRank


# ============================================================================
# PYTORCH Q-NETWORK (GPU-accelerated)
# ============================================================================
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """PyTorch Q-Network for GPU-accelerated training"""
        
        def __init__(self, state_size: int, action_size: int, hidden_sizes: Tuple[int, ...] = (64, 32)):
            super(QNetwork, self).__init__()
            
            layers = []
            prev_size = state_size
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, action_size))
            
            self.network = nn.Sequential(*layers)
            self.to(DEVICE)
        
        def forward(self, x):
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x).to(DEVICE)
            return self.network(x)
        
        def predict(self, x):
            """Sklearn-compatible predict method"""
            self.eval()
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(DEVICE)
                return self.forward(x).cpu().numpy()


# ============================================================================
# BANK RL AGENT
# ============================================================================
class BankRLAgent:
    """
    RL agent for a single bank.
    Learns to maximize: E[profit] - risk - margin_cost - failure_penalty
    
    State features:
        - Bank's own health score (0-100)
        - Equity ratio (equity / assets)
        - Liquidity ratio (HQLA / liabilities)
        - Margin available
        - System health (avg health of all banks)
        - Number of failed banks
        - Recent profit/loss
        - Stock volatility of held assets
    
    Actions (discretized):
        - Trade size: [0%, 2%, 5%, 10%] of HQLA
        - Direction: [BUY, SELL, HOLD]
        Total: 4 * 3 = 12 actions
    """
    
    def __init__(self, bank_name: str, state_size: int = 10, action_size: int = 12):
        self.bank_name = bank_name
        self.state_size = state_size
        self.action_size = action_size
        self.lock = threading.Lock()  # Thread safety
        
        # Q-network (PyTorch for GPU, sklearn fallback)
        if TORCH_AVAILABLE:
            self.model = QNetwork(state_size, action_size, hidden_sizes=(64, 32))
            self.optimizer = optim.Adam(self.model.parameters(), lr=RL_CONFIG['LEARNING_RATE'])
            self.loss_fn = nn.MSELoss()
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=1,
                warm_start=True,
                learning_rate_init=RL_CONFIG['LEARNING_RATE']
            )
            self._init_sklearn_model()
        
        # Experience replay buffer
        self.memory = deque(maxlen=RL_CONFIG['MEMORY_SIZE'])
        self.epsilon = RL_CONFIG['EPSILON_START']
        
        # Action mappings
        self.trade_sizes = [0.0, 0.02, 0.05, 0.10]  # Fraction of HQLA
        self.directions = ['HOLD', 'BUY', 'SELL']
        
        # Track performance
        self.cumulative_reward = 0
        self.episode_rewards = []
    
    def _init_sklearn_model(self):
        """Initialize sklearn Q-network with dummy data"""
        X = np.random.randn(10, self.state_size)
        y = np.random.randn(10, self.action_size)
        self.model.fit(X, y)
    
    def get_state_features(self, contagion: BankingNetworkContagion, 
                          bank_name: str) -> np.ndarray:
        """Extract state features for this bank from the simulation"""
        state = contagion.bank_states.get(bank_name, {})
        graph = contagion.graph
        
        # Bank-specific features
        health = contagion.get_bank_health(bank_name)
        assets = state.get('Total_Assets', 0)
        equity = state.get('Equity', 0)
        hqla = state.get('HQLA', 0)
        liabilities = state.get('Total_Liabilities', 1)
        margin_available = contagion.margin_states.get(bank_name, 0)
        
        equity_ratio = equity / max(assets, 1)
        liquidity_ratio = hqla / max(liabilities * 0.1, 1)
        
        # System-wide features
        all_banks = list(graph.keys())
        num_failed = len(contagion.failed_banks)
        system_health = np.mean([contagion.get_bank_health(b) for b in all_banks 
                                 if b not in contagion.failed_banks]) if num_failed < len(all_banks) else 0
        
        # Holdings volatility (weighted avg of held stocks)
        holdings = graph[bank_name].get('holdings', {})
        stock_vols = contagion.graph[bank_name]['attributes'].get('Stock_Volatility', 0.3)
        
        features = np.array([
            health / 100.0,                           # Normalized health
            equity_ratio,                              # Equity / Assets
            liquidity_ratio,                           # Liquidity coverage
            margin_available / max(assets * 0.1, 1),  # Margin buffer ratio
            system_health / 100.0,                    # System health
            num_failed / len(all_banks),              # Failure ratio
            assets / 1000.0,                          # Normalized assets (in trillions)
            stock_vols,                               # Stock volatility
            len(holdings) / 10.0,                     # Portfolio diversity
            hqla / max(assets, 1)                     # Cash ratio
        ])
        
        return features
    
    def action_to_index(self, size_idx: int, dir_idx: int) -> int:
        """Convert (size, direction) to action index"""
        return size_idx * len(self.directions) + dir_idx
    
    def index_to_action(self, action_idx: int) -> Tuple[int, int]:
        """Convert action index to (size_idx, dir_idx)"""
        size_idx = action_idx // len(self.directions)
        dir_idx = action_idx % len(self.directions)
        return size_idx, dir_idx
    
    def get_action(self, state: np.ndarray, hqla: float, 
                  available_assets: List[str], explore: bool = True) -> BankAction:
        """Choose action using epsilon-greedy policy"""
        
        if explore and random.random() < self.epsilon:
            # Random exploration
            size_idx = random.randint(0, len(self.trade_sizes) - 1)
            dir_idx = random.randint(0, len(self.directions) - 1)
        else:
            # Greedy action from Q-network
            q_values = self.model.predict(state.reshape(1, -1))[0]
            action_idx = np.argmax(q_values)
            size_idx, dir_idx = self.index_to_action(action_idx)
        
        direction = self.directions[dir_idx]
        
        if direction == 'HOLD' or size_idx == 0:
            return BankAction(
                trade_size=0,
                asset=available_assets[0] if available_assets else 'NONE',
                direction='HOLD'
            )
        
        trade_size = hqla * self.trade_sizes[size_idx]
        asset = random.choice(available_assets) if available_assets else 'AAPL'
        
        return BankAction(
            trade_size=trade_size,
            asset=asset,
            direction=direction
        )
    
    def remember(self, state: np.ndarray, action_idx: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer (thread-safe)"""
        with self.lock:
            self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self, batch_size: int = None):
        """Train on batch from replay buffer (GPU-accelerated if available)"""
        if batch_size is None:
            batch_size = RL_CONFIG['BATCH_SIZE']
        
        with self.lock:
            if len(self.memory) < batch_size:
                return
            batch = random.sample(self.memory, batch_size)
        
        if TORCH_AVAILABLE:
            self._replay_torch(batch)
        else:
            self._replay_sklearn(batch)
        
        # Decay epsilon
        if self.epsilon > RL_CONFIG['EPSILON_MIN']:
            self.epsilon *= RL_CONFIG['EPSILON_DECAY']
    
    def _replay_torch(self, batch):
        """GPU-accelerated training with PyTorch"""
        states = torch.FloatTensor(np.array([s for s, _, _, _, _ in batch])).to(DEVICE)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(DEVICE)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(DEVICE)
        next_states = torch.FloatTensor(np.array([ns for _, _, _, ns, _ in batch])).to(DEVICE)
        dones = torch.BoolTensor([d for _, _, _, _, d in batch]).to(DEVICE)
        
        # Get current Q-values
        self.model.train()
        current_q = self.model(states)
        current_q_actions = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.model(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + RL_CONFIG['GAMMA'] * max_next_q * (~dones)
        
        # Update network
        loss = self.loss_fn(current_q_actions, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _replay_sklearn(self, batch):
        """CPU training with sklearn"""
        X = np.array([s for s, _, _, _, _ in batch])
        current_q = self.model.predict(X)
        
        for i, (state, action_idx, reward, next_state, done) in enumerate(batch):
            if done:
                target = reward
            else:
                next_q = self.model.predict(next_state.reshape(1, -1))[0]
                target = reward + RL_CONFIG['GAMMA'] * np.max(next_q)
            current_q[i, action_idx] = target
        
        self.model.fit(X, current_q)
    
    def save(self, path: str):
        """Save model to file"""
        if TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'cumulative_reward': self.cumulative_reward
            }, path.replace('.pkl', '.pt'))
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'epsilon': self.epsilon,
                    'cumulative_reward': self.cumulative_reward
                }, f)
    
    def load(self, path: str):
        """Load model from file"""
        if TORCH_AVAILABLE:
            pt_path = path.replace('.pkl', '.pt')
            if os.path.exists(pt_path):
                checkpoint = torch.load(pt_path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', RL_CONFIG['EPSILON_MIN'])
        elif os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.epsilon = data.get('epsilon', RL_CONFIG['EPSILON_MIN'])


# ============================================================================
# CCP RL AGENT
# ============================================================================
class CCPRLAgent:
    """
    RL agent for the Central Counterparty (CCP).
    Learns optimal margin requirements to minimize systemic risk.
    
    State features (for each trade request):
        - Requesting bank's health
        - Requesting bank's size (assets)
        - Trade size relative to bank's HQLA
        - Stock volatility of traded asset
        - Current system stress level
        - Number of already failed banks
        - Total system margin buffer
        - Trade direction (buy/sell encoded)
    
    Actions (discretized margin levels):
        - [APPROVE_LOW, APPROVE_MEDIUM, APPROVE_HIGH, REQUIRE_EXTRA, REJECT]
        - Margins: [5%, 10%, 20%, 35%, REJECT]
    """
    
    def __init__(self, state_size: int = 12, action_size: int = 5):
        self.state_size = state_size
        self.action_size = action_size
        self.lock = threading.Lock()  # Thread safety
        
        # Margin levels
        self.margin_levels = [0.05, 0.10, 0.20, 0.35]  # Last action = REJECT
        
        # Q-network (PyTorch for GPU, sklearn fallback)
        if TORCH_AVAILABLE:
            self.model = QNetwork(state_size, action_size, hidden_sizes=(128, 64, 32))
            self.optimizer = optim.Adam(self.model.parameters(), lr=RL_CONFIG['LEARNING_RATE'])
            self.loss_fn = nn.MSELoss()
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=1,
                warm_start=True,
                learning_rate_init=RL_CONFIG['LEARNING_RATE']
            )
            self._init_sklearn_model()
        
        # Experience replay
        self.memory = deque(maxlen=RL_CONFIG['MEMORY_SIZE'])
        self.epsilon = RL_CONFIG['EPSILON_START']
        
        # Performance tracking
        self.cumulative_reward = 0
        self.episode_rewards = []
        self.trades_approved = 0
        self.trades_rejected = 0
    
    def _init_sklearn_model(self):
        """Initialize sklearn Q-network"""
        X = np.random.randn(10, self.state_size)
        y = np.random.randn(10, self.action_size)
        self.model.fit(X, y)
    
    def get_state_features(self, contagion: BankingNetworkContagion,
                          bank_name: str, trade: BankAction) -> np.ndarray:
        """Extract state features for CCP decision"""
        bank_state = contagion.bank_states.get(bank_name, {})
        graph = contagion.graph
        
        # Bank features
        health = contagion.get_bank_health(bank_name)
        assets = bank_state.get('Total_Assets', 0)
        hqla = bank_state.get('HQLA', 0)
        equity = bank_state.get('Equity', 0)
        margin = contagion.margin_states.get(bank_name, 0)
        
        # Trade features
        trade_size_ratio = trade.trade_size / max(hqla, 1) if trade.trade_size > 0 else 0
        stock_vol = graph[bank_name]['attributes'].get('Stock_Volatility', 0.3)
        direction_encoded = 1.0 if trade.direction == 'BUY' else (-1.0 if trade.direction == 'SELL' else 0.0)
        
        # System features
        all_banks = list(graph.keys())
        num_failed = len(contagion.failed_banks)
        system_health = np.mean([contagion.get_bank_health(b) for b in all_banks 
                                 if b not in contagion.failed_banks]) if num_failed < len(all_banks) else 0
        total_margin = sum(contagion.margin_states.values())
        system_assets = sum(contagion.bank_states[b].get('Total_Assets', 0) for b in all_banks)
        
        features = np.array([
            health / 100.0,                             # Bank health
            assets / 1000.0,                            # Bank size (in trillions)
            equity / max(assets, 1),                    # Equity ratio
            trade_size_ratio,                           # Trade size / HQLA
            stock_vol,                                  # Stock volatility
            direction_encoded,                          # Trade direction
            system_health / 100.0,                      # System health
            num_failed / max(len(all_banks), 1),        # Failure ratio
            margin / max(assets * 0.1, 1),              # Bank margin buffer
            total_margin / max(system_assets * 0.1, 1), # System margin buffer
            hqla / max(assets, 1),                      # Bank liquidity ratio
            trade.trade_size / 100.0                    # Absolute trade size
        ])
        
        return features
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> CCPAction:
        """Choose margin level using epsilon-greedy policy"""
        
        if explore and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            q_values = self.model.predict(state.reshape(1, -1))[0]
            action_idx = np.argmax(q_values)
        
        if action_idx == self.action_size - 1:
            # REJECT
            self.trades_rejected += 1
            return CCPAction(
                margin_multiplier=1.0,
                decision=TradeDecision.REJECTED
            )
        
        margin = self.margin_levels[action_idx]
        
        # Determine decision based on margin level
        if margin <= 0.10:
            decision = TradeDecision.APPROVED
        else:
            decision = TradeDecision.REQUIRE_MARGIN
        
        self.trades_approved += 1
        return CCPAction(
            margin_multiplier=margin,
            decision=decision
        )
    
    def remember(self, state: np.ndarray, action_idx: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience (thread-safe)"""
        with self.lock:
            self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self, batch_size: int = None):
        """Train on batch (GPU-accelerated if available)"""
        if batch_size is None:
            batch_size = RL_CONFIG['BATCH_SIZE']
        
        with self.lock:
            if len(self.memory) < batch_size:
                return
            batch = random.sample(self.memory, batch_size)
        
        if TORCH_AVAILABLE:
            self._replay_torch(batch)
        else:
            self._replay_sklearn(batch)
        
        if self.epsilon > RL_CONFIG['EPSILON_MIN']:
            self.epsilon *= RL_CONFIG['EPSILON_DECAY']
    
    def _replay_torch(self, batch):
        """GPU-accelerated training"""
        states = torch.FloatTensor(np.array([s for s, _, _, _, _ in batch])).to(DEVICE)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(DEVICE)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(DEVICE)
        next_states = torch.FloatTensor(np.array([ns for _, _, _, ns, _ in batch])).to(DEVICE)
        dones = torch.BoolTensor([d for _, _, _, _, d in batch]).to(DEVICE)
        
        self.model.train()
        current_q = self.model(states)
        current_q_actions = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.model(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + RL_CONFIG['GAMMA'] * max_next_q * (~dones)
        
        loss = self.loss_fn(current_q_actions, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _replay_sklearn(self, batch):
        """CPU training with sklearn"""
        X = np.array([s for s, _, _, _, _ in batch])
        current_q = self.model.predict(X)
        
        for i, (state, action_idx, reward, next_state, done) in enumerate(batch):
            if done:
                target = reward
            else:
                next_q = self.model.predict(next_state.reshape(1, -1))[0]
                target = reward + RL_CONFIG['GAMMA'] * np.max(next_q)
            current_q[i, action_idx] = target
        
        self.model.fit(X, current_q)
    
    def save(self, path: str):
        if TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'trades_approved': self.trades_approved,
                'trades_rejected': self.trades_rejected
            }, path.replace('.pkl', '.pt'))
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'epsilon': self.epsilon,
                    'trades_approved': self.trades_approved,
                    'trades_rejected': self.trades_rejected
                }, f)
    
    def load(self, path: str):
        if TORCH_AVAILABLE:
            pt_path = path.replace('.pkl', '.pt')
            if os.path.exists(pt_path):
                checkpoint = torch.load(pt_path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', RL_CONFIG['EPSILON_MIN'])
        elif os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.epsilon = data.get('epsilon', RL_CONFIG['EPSILON_MIN'])


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================
def compute_bank_reward(bank_name: str, 
                       health_before: float, health_after: float,
                       profit: float, margin_used: float,
                       failed: bool, 
                       ccp_decision: TradeDecision) -> float:
    """
    Compute reward for bank agent.
    
    Reward = profit_reward + health_bonus - health_penalty - margin_cost - failure_penalty
    """
    reward = 0.0
    
    # 1. Profit component (normalized)
    reward += RL_CONFIG['REWARD_PROFIT'] * (profit / 10.0)  # Normalize by $10B
    
    # 2. Health maintenance bonus
    if health_after >= health_before:
        reward += RL_CONFIG['REWARD_HEALTH_BONUS'] * (health_after / 100.0)
    else:
        # Penalty for health degradation
        health_drop = (health_before - health_after) / 100.0
        reward -= RL_CONFIG['PENALTY_HEALTH_LOSS'] * health_drop
    
    # 3. Margin cost (opportunity cost)
    reward -= RL_CONFIG['PENALTY_MARGIN_COST'] * (margin_used / 10.0)
    
    # 4. Failure penalty (catastrophic)
    if failed:
        reward -= RL_CONFIG['PENALTY_FAILURE']
    
    # 5. Penalty for rejected trades (wasted opportunity)
    if ccp_decision == TradeDecision.REJECTED:
        reward -= 0.5
    
    return reward


def compute_ccp_reward(contagion: BankingNetworkContagion,
                      num_trades: int, total_volume: float,
                      failures_before: int, failures_after: int,
                      system_health_before: float, system_health_after: float,
                      total_margin_collected: float) -> float:
    """
    Compute reward for CCP agent.
    
    CCP wants to:
    - Minimize systemic risk (bank failures)
    - Enable healthy market activity (volume)
    - Maintain appropriate margins (not too high, not too low)
    """
    reward = 0.0
    
    # 1. Penalty for bank failures
    new_failures = failures_after - failures_before
    if new_failures > 0:
        reward -= RL_CONFIG['CCP_PENALTY_FAILURES'] * new_failures
        
        # Extra penalty for cascade failures (more than 1)
        if new_failures > 1:
            reward -= RL_CONFIG['CCP_PENALTY_SYSTEMIC'] * (new_failures - 1)
    
    # 2. System stability reward
    if system_health_after >= system_health_before:
        reward += RL_CONFIG['CCP_REWARD_STABILITY']
    else:
        health_drop = (system_health_before - system_health_after) / 100.0
        reward -= RL_CONFIG['CCP_PENALTY_SYSTEMIC'] * health_drop
    
    # 3. Volume reward (encourage healthy trading)
    reward += RL_CONFIG['CCP_REWARD_VOLUME'] * (total_volume / 10.0)
    
    # 4. Margin efficiency (penalize over-margining that kills trading)
    total_assets = sum(contagion.bank_states[b].get('Total_Assets', 0) 
                       for b in contagion.graph)
    margin_ratio = total_margin_collected / max(total_assets * 0.05, 1)
    if margin_ratio > 2.0:  # Over 2x normal margins
        reward -= RL_CONFIG['CCP_PENALTY_OVER_MARGIN'] * (margin_ratio - 2.0)
    
    # 5. Bonus for zero-failure episodes
    if failures_after == 0:
        reward += RL_CONFIG['CCP_REWARD_STABILITY'] * 2.0
    
    return reward


# ============================================================================
# TRAINING ENVIRONMENT
# ============================================================================
class RLTrainingEnvironment:
    """
    Training environment that integrates RL agents with BankingNetworkContagion.
    """
    
    def __init__(self, data_dir: str = './dataset'):
        """Initialize environment with data"""
        print("=" * 70)
        print("INITIALIZING RL TRAINING ENVIRONMENT")
        print("=" * 70)
        
        # Load data
        self.bank_attrs = load_bank_attributes(f'{data_dir}/us_banks_top50_nodes_final.csv')
        self.stock_prices, self.stock_timeseries = load_stock_prices(
            f'{data_dir}/stocks_data_long.csv', num_stocks=15
        )
        
        self.bank_list = list(self.bank_attrs.keys())
        self.available_assets = list(self.stock_prices.keys())
        
        # Initialize agents
        print(f"\nInitializing {len(self.bank_list)} bank agents...")
        self.bank_agents = {
            bank: BankRLAgent(bank) for bank in self.bank_list
        }
        
        print("Initializing CCP agent...")
        self.ccp_agent = CCPRLAgent()
        
        # Create model directory
        os.makedirs('models', exist_ok=True)
        
        # Training metrics
        self.episode_metrics = []
        
        # Calculate training instances
        num_banks = len(self.bank_list)
        num_episodes = RL_CONFIG['NUM_EPISODES']
        steps_per_ep = RL_CONFIG['STEPS_PER_EPISODE']
        # Max potential experiences per episode: banks * steps (each bank acts each step)
        max_experiences_per_ep = num_banks * steps_per_ep
        total_training_instances = num_episodes * max_experiences_per_ep
        
        print(f"✅ Environment ready: {num_banks} banks, {len(self.available_assets)} stocks")
        print(f"\n📊 TRAINING INSTANCES:")
        print(f"   Episodes: {num_episodes}")
        print(f"   Steps/Episode: {steps_per_ep}")
        print(f"   Banks: {num_banks}")
        print(f"   Max experiences/episode: {max_experiences_per_ep:,}")
        print(f"   Total training instances: {total_training_instances:,}")
    
    def reset_episode(self) -> BankingNetworkContagion:
        """Reset environment for new episode"""
        # Generate fresh network
        graph = generate_random_graph_with_sccs(
            self.bank_attrs, num_sccs=4, prob_intra=0.4, prob_inter=0.05
        )
        
        # Distribute holdings
        holdings = distribute_shares(self.bank_attrs, self.stock_prices)
        for bank in graph:
            if bank in holdings:
                graph[bank]['holdings'] = holdings[bank]
        
        # Generate margins
        margin_requirements = generate_margin_requirements(
            self.bank_attrs, margin_ratio_range=(0.02, 0.08)
        )
        
        # Create contagion simulator
        contagion = BankingNetworkContagion(graph, self.stock_prices, margin_requirements)
        
        return contagion
    
    def run_episode(self, contagion: BankingNetworkContagion, train: bool = True, do_replay: bool = True) -> Dict:
        """
        Run one episode of training.
        
        Args:
            contagion: The simulation environment
            train: Whether to collect experiences (exploration + remember)
            do_replay: Whether to train models after episode (set False for parallel collection)
        """
        episode_rewards = {bank: 0.0 for bank in self.bank_list}
        ccp_reward_total = 0.0
        total_trades = 0
        total_volume = 0.0
        total_margin_collected = 0.0
        
        initial_failures = len(contagion.failed_banks)
        initial_health = np.mean([contagion.get_bank_health(b) for b in self.bank_list])
        
        for step in range(RL_CONFIG['STEPS_PER_EPISODE']):
            step_margin = 0.0
            step_trades = 0
            step_volume = 0.0
            
            failures_before = len(contagion.failed_banks)
            health_before = {b: contagion.get_bank_health(b) for b in self.bank_list}
            
            # ===== BANK ACTIONS =====
            for bank in self.bank_list:
                if bank in contagion.failed_banks:
                    continue
                
                # Get state
                state = self.bank_agents[bank].get_state_features(contagion, bank)
                hqla = contagion.bank_states[bank].get('HQLA', 0)
                
                # Bank chooses action
                holdings = contagion.graph[bank].get('holdings', {})
                owned_assets = list(holdings.keys()) if holdings else self.available_assets[:3]
                
                action = self.bank_agents[bank].get_action(state, hqla, owned_assets, explore=train)
                
                if action.direction == 'HOLD' or action.trade_size <= 0:
                    continue
                
                # ===== CCP EVALUATES =====
                ccp_state = self.ccp_agent.get_state_features(contagion, bank, action)
                ccp_action = self.ccp_agent.get_action(ccp_state, explore=train)
                
                if ccp_action.decision == TradeDecision.REJECTED:
                    # Store experience for rejected trade
                    if train:
                        # Bank learns rejection is bad
                        action_idx = self.bank_agents[bank].action_to_index(
                            self.bank_agents[bank].trade_sizes.index(
                                min(self.bank_agents[bank].trade_sizes, 
                                    key=lambda x: abs(x - action.trade_size / max(hqla, 1)))
                            ) if hqla > 0 else 0,
                            self.bank_agents[bank].directions.index(action.direction)
                        )
                        reward = compute_bank_reward(
                            bank, health_before[bank], health_before[bank],
                            0, 0, False, ccp_action.decision
                        )
                        self.bank_agents[bank].remember(state, action_idx, reward, state, False)
                    continue
                
                # ===== EXECUTE TRADE =====
                profit = self._execute_trade(contagion, bank, action, ccp_action)
                
                step_trades += 1
                step_volume += action.trade_size
                step_margin += action.trade_size * ccp_action.margin_multiplier
                
                # Get new state
                health_after = contagion.get_bank_health(bank)
                failed = bank in contagion.failed_banks
                
                # ===== COMPUTE REWARDS =====
                bank_reward = compute_bank_reward(
                    bank, health_before[bank], health_after,
                    profit, action.trade_size * ccp_action.margin_multiplier,
                    failed, ccp_action.decision
                )
                episode_rewards[bank] += bank_reward
                
                # Store bank experience
                if train:
                    next_state = self.bank_agents[bank].get_state_features(contagion, bank)
                    size_idx = self.bank_agents[bank].trade_sizes.index(
                        min(self.bank_agents[bank].trade_sizes,
                            key=lambda x: abs(x - action.trade_size / max(hqla, 1)))
                    ) if hqla > 0 else 0
                    dir_idx = self.bank_agents[bank].directions.index(action.direction)
                    action_idx = self.bank_agents[bank].action_to_index(size_idx, dir_idx)
                    self.bank_agents[bank].remember(state, action_idx, bank_reward, next_state, failed)
            
            # ===== MARKET SHOCK =====
            if random.random() < RL_CONFIG['SHOCK_PROBABILITY']:
                self._apply_market_shock(contagion)
            
            # ===== RUN CONTAGION =====
            cascade_result = self._run_contagion_step(contagion)
            
            # ===== CCP REWARD =====
            failures_after = len(contagion.failed_banks)
            system_health_after = np.mean([contagion.get_bank_health(b) for b in self.bank_list
                                          if b not in contagion.failed_banks]) if failures_after < len(self.bank_list) else 0
            
            ccp_reward = compute_ccp_reward(
                contagion, step_trades, step_volume,
                failures_before, failures_after,
                initial_health, system_health_after,
                step_margin
            )
            ccp_reward_total += ccp_reward
            
            # Store CCP experience
            if train and step_trades > 0:
                ccp_state_after = self.ccp_agent.get_state_features(
                    contagion, self.bank_list[0], BankAction(0, '', 'HOLD')
                )
                # Use average action index for simplicity
                self.ccp_agent.remember(ccp_state, 1, ccp_reward, ccp_state_after, False)
            
            total_trades += step_trades
            total_volume += step_volume
            total_margin_collected += step_margin
            
            # Early termination if system collapses
            if len(contagion.failed_banks) > len(self.bank_list) * 0.5:
                break
        
        # ===== TRAINING =====
        if train and do_replay:
            for bank_agent in self.bank_agents.values():
                bank_agent.replay()
            self.ccp_agent.replay()
        
        return {
            'total_trades': total_trades,
            'total_volume': total_volume,
            'failed_banks': len(contagion.failed_banks),
            'avg_bank_reward': np.mean(list(episode_rewards.values())),
            'ccp_reward': ccp_reward_total,
            'final_health': np.mean([contagion.get_bank_health(b) for b in self.bank_list
                                    if b not in contagion.failed_banks]) if len(contagion.failed_banks) < len(self.bank_list) else 0
        }
    
    def _execute_trade(self, contagion: BankingNetworkContagion, 
                       bank: str, action: BankAction, ccp_action: CCPAction) -> float:
        """Execute a trade and return profit"""
        vol = contagion.graph[bank]['attributes'].get('Stock_Volatility', 0.02)
        price_change = np.random.normal(0, vol)
        
        if action.direction == 'BUY':
            profit = action.trade_size * price_change
            contagion.bank_states[bank]['HQLA'] = max(0, 
                contagion.bank_states[bank]['HQLA'] - action.trade_size)
        else:  # SELL
            profit = -action.trade_size * price_change
            contagion.bank_states[bank]['HQLA'] += action.trade_size * 0.95
        
        contagion.bank_states[bank]['Total_Assets'] += profit
        contagion.bank_states[bank]['Equity'] += profit
        
        return profit
    
    def _apply_market_shock(self, contagion: BankingNetworkContagion):
        """Apply random market shock"""
        shock_pct = random.uniform(*RL_CONFIG['SHOCK_RANGE'])
        
        for bank in self.bank_list:
            if bank not in contagion.failed_banks:
                loss = contagion.bank_states[bank]['Total_Assets'] * shock_pct * random.uniform(0.3, 1.0)
                actual_loss, _ = contagion._use_margin_buffer(bank, loss)
                contagion.bank_states[bank]['Total_Assets'] -= actual_loss
                contagion.bank_states[bank]['Equity'] -= actual_loss
    
    def _run_contagion_step(self, contagion: BankingNetworkContagion) -> Dict:
        """Run one step of contagion"""
        initial_failed = len(contagion.failed_banks)
        
        # Check for new failures
        for bank in self.bank_list:
            if bank not in contagion.failed_banks:
                if contagion.get_bank_health(bank) < RL_CONFIG['FAILURE_THRESHOLD']:
                    contagion.failed_banks.add(bank)
        
        return {
            'new_failures': len(contagion.failed_banks) - initial_failed
        }
    
    def train(self, num_episodes: int = None, parallel: bool = True):
        """Run training loop with optional parallelization"""
        if num_episodes is None:
            num_episodes = RL_CONFIG['NUM_EPISODES']
        
        num_workers = RL_CONFIG['NUM_WORKERS'] if parallel else 1
        use_gpu = RL_CONFIG['USE_GPU']
        
        print(f"\n🚀 Starting training for {num_episodes} episodes...")
        print(f"   Workers: {num_workers} | GPU: {'Yes' if use_gpu else 'No'}")
        print("=" * 70)
        
        if parallel and num_workers > 1:
            self._train_parallel(num_episodes, num_workers)
        else:
            self._train_sequential(num_episodes)
        
        print("=" * 70)
        print("✅ Training complete!")
        self.save_models()
    
    def _train_sequential(self, num_episodes: int):
        """Sequential training (single-threaded)"""
        for episode in range(num_episodes):
            contagion = self.reset_episode()
            result = self.run_episode(contagion, train=True)
            self.episode_metrics.append(result)
            self._print_progress(episode + 1, result)
    
    def _train_parallel(self, num_episodes: int, num_workers: int):
        """Parallel training with multiple workers collecting experiences"""
        batch_size = num_workers * 2  # Episodes per batch
        
        def collect_episode(_):
            """Collect experiences from one episode (no model updates)"""
            contagion = self.reset_episode()
            # Collect experiences but don't update models (not thread-safe)
            result = self.run_episode(contagion, train=True, do_replay=False)
            return result
        
        episode = 0
        while episode < num_episodes:
            # Collect batch of episodes in parallel
            batch_count = min(batch_size, num_episodes - episode)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(collect_episode, range(batch_count)))
            
            # Store results
            self.episode_metrics.extend(results)
            
            # Batch training on all agents in main thread (thread-safe)
            for bank_agent in self.bank_agents.values():
                bank_agent.replay()
            self.ccp_agent.replay()
            
            episode += batch_count
            
            # Print progress for last result in batch
            if results:
                self._print_progress(episode, results[-1])
    
    def _print_progress(self, episode: int, result: Dict):
        """Print training progress"""
        if episode % 25 == 0:
            recent = self.episode_metrics[-25:] if len(self.episode_metrics) >= 25 else self.episode_metrics
            avg_reward = np.mean([m['avg_bank_reward'] for m in recent])
            avg_failures = np.mean([m['failed_banks'] for m in recent])
            avg_ccp = np.mean([m['ccp_reward'] for m in recent])
            eps = self.bank_agents[self.bank_list[0]].epsilon
            
            print(f"Episode {episode:4d} | "
                  f"Trades: {result['total_trades']:3d} | "
                  f"Failed: {avg_failures:.1f} | "
                  f"Bank R: {avg_reward:+.2f} | "
                  f"CCP R: {avg_ccp:+.2f} | "
                  f"ε: {eps:.3f}")
    
    def save_models(self):
        """Save all models"""
        for bank, agent in self.bank_agents.items():
            agent.save(f'models/bank_{bank}.pkl')
        self.ccp_agent.save(RL_CONFIG['CCP_MODEL_PATH'])
        print(f"✅ Models saved to models/ directory")
    
    def load_models(self):
        """Load all models"""
        for bank, agent in self.bank_agents.items():
            path = f'models/bank_{bank}.pkl'
            if os.path.exists(path):
                agent.load(path)
        self.ccp_agent.load(RL_CONFIG['CCP_MODEL_PATH'])
        print("✅ Models loaded")
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate trained agents"""
        print(f"\n🔍 Evaluating over {num_episodes} episodes...")
        
        results = []
        for _ in range(num_episodes):
            contagion = self.reset_episode()
            result = self.run_episode(contagion, train=False)
            results.append(result)
        
        avg_result = {
            'avg_trades': np.mean([r['total_trades'] for r in results]),
            'avg_failures': np.mean([r['failed_banks'] for r in results]),
            'avg_bank_reward': np.mean([r['avg_bank_reward'] for r in results]),
            'avg_ccp_reward': np.mean([r['ccp_reward'] for r in results]),
            'avg_final_health': np.mean([r['final_health'] for r in results])
        }
        
        print(f"  Avg Trades/Episode: {avg_result['avg_trades']:.1f}")
        print(f"  Avg Failures: {avg_result['avg_failures']:.1f}")
        print(f"  Avg Bank Reward: {avg_result['avg_bank_reward']:.2f}")
        print(f"  Avg CCP Reward: {avg_result['avg_ccp_reward']:.2f}")
        print(f"  Avg Final Health: {avg_result['avg_final_health']:.1f}")
        
        return avg_result


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("CCP REINFORCEMENT LEARNING TRAINING")
    print("=" * 70)
    
    # Initialize environment
    env = RLTrainingEnvironment(data_dir='./dataset')
    
    # Check for existing models
    if os.path.exists(RL_CONFIG['CCP_MODEL_PATH']):
        print("\n📂 Found existing models. Loading...")
        env.load_models()
        print("Running evaluation...")
        env.evaluate(num_episodes=5)
        
        # Continue training
        print("\nContinuing training...")
        env.train(num_episodes=200)
    else:
        # Train from scratch
        env.train(num_episodes=RL_CONFIG['NUM_EPISODES'])
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    env.evaluate(num_episodes=10)
    
    print("\n✅ Training complete! Models saved to models/ directory")