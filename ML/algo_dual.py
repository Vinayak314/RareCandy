"""
================================================================================
ALGO.PY - Dual RL Training: CCP + Bank Policy
================================================================================
Trains TWO models:
  1. CCP Agent: Learns optimal margin requirements to minimize systemic risk
  2. Bank Agent: Learns optimal trading behavior to maximize profit while surviving

Supports:
  - GPU acceleration via PyTorch (auto-detected)
  - Independent experience replay for each agent
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
from typing import Dict, List, Tuple, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import threading

# Try to import PyTorch for GPU support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        print("🚀 Apple MPS (Metal) detected")
    else:
        DEVICE = torch.device('cpu')
        print("💻 Using CPU (no GPU detected)")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not found - using sklearn (slower)")

if not TORCH_AVAILABLE:
    from sklearn.neural_network import MLPRegressor

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
    # Training
    'NUM_EPISODES': 100,
    'STEPS_PER_EPISODE': 100,
    'NUM_WORKERS': min(4, mp.cpu_count()),
    
    # RL hyperparameters
    'GAMMA': 0.95,
    'EPSILON_START': 1.0,
    'EPSILON_MIN': 0.01,
    'EPSILON_DECAY': 0.9995,
    'MEMORY_SIZE': 200000,
    'BATCH_SIZE': 1024 if TORCH_AVAILABLE else 64,
    'LEARNING_RATE': 0.0005,
    'TRAIN_FREQ': 4,
    
    # CCP Rewards
    'CCP_PENALTY_FAILURE': 100.0,
    'CCP_PENALTY_CASCADE': 50.0,
    'CCP_PENALTY_COLLAPSE': 2000.0,
    'CCP_REWARD_STABILITY': 20.0,
    'CCP_REWARD_TRADE': 0.5,
    'CCP_PENALTY_REJECT': 5.0,
    'CCP_PENALTY_HIGH_MARGIN': 1.0,
    
    # Bank Rewards
    'BANK_REWARD_PROFIT': 10.0,        # Per unit profit
    'BANK_PENALTY_LOSS': 15.0,         # Per unit loss (asymmetric)
    'BANK_REWARD_SURVIVAL': 5.0,       # Bonus for surviving each step
    'BANK_PENALTY_FAILURE': 500.0,     # Huge penalty for failing
    'BANK_PENALTY_RISKY': 2.0,         # Trading when unhealthy
    'BANK_REWARD_CONSERVATIVE': 1.0,   # Holding when unhealthy
    
    # Simulation
    'SHOCK_PROBABILITY': 0.15,
    'SHOCK_RANGE': (0.10, 0.40),
    'FAILURE_THRESHOLD': 20.0,
    
    # Save paths
    'CCP_MODEL_PATH': 'models/ccp_policy.pt',
    'BANK_MODEL_PATH': 'models/bank_policy.pt',
}


# ============================================================================
# ACTION ENUMS
# ============================================================================
class MarginDecision(Enum):
    """CCP's margin decisions"""
    LOW = 0       # 5% margin
    MEDIUM = 1    # 10% margin
    HIGH = 2      # 20% margin
    VERY_HIGH = 3 # 35% margin
    REJECT = 4    # Reject trade


class BankAction(Enum):
    """Bank's trading decisions"""
    HOLD = 0           # No trade
    SMALL_BUY = 1      # Buy 2-5% of HQLA
    SMALL_SELL = 2     # Sell 2-5% of HQLA
    MEDIUM_BUY = 3     # Buy 5-10% of HQLA
    MEDIUM_SELL = 4    # Sell 5-10% of HQLA
    LARGE_BUY = 5      # Buy 10-15% of HQLA
    LARGE_SELL = 6     # Sell 10-15% of HQLA


# ============================================================================
# PYTORCH Q-NETWORK
# ============================================================================
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, state_size: int, action_size: int, hidden_sizes=(512, 256, 128, 64)):
            super().__init__()
            
            layers = []
            prev_size = state_size
            for i, hidden_size in enumerate(hidden_sizes):
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                if i < 2:  # Dropout on first two layers
                    layers.append(nn.Dropout(0.1))
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, action_size))
            
            self.network = nn.Sequential(*layers)
            self.to(DEVICE)
        
        def forward(self, x):
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x).to(DEVICE)
            return self.network(x)
        
        def predict(self, x):
            self.eval()
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(DEVICE)
                return self.forward(x).cpu().numpy()


# ============================================================================
# BASE AGENT CLASS
# ============================================================================
class BaseAgent:
    """Base class for RL agents"""
    
    def __init__(self, state_size: int, action_size: int, name: str):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.lock = threading.Lock()
        
        if TORCH_AVAILABLE:
            self.model = QNetwork(state_size, action_size)
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
            self._init_sklearn()
        
        self.memory = deque(maxlen=RL_CONFIG['MEMORY_SIZE'])
        self.epsilon = RL_CONFIG['EPSILON_START']
    
    def _init_sklearn(self):
        X = np.random.randn(10, self.state_size)
        y = np.random.randn(10, self.action_size)
        self.model.fit(X, y)
    
    def get_action_idx(self, state: np.ndarray, explore: bool = True) -> int:
        """Choose action using epsilon-greedy"""
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = self.model.predict(state.reshape(1, -1))[0]
            return np.argmax(q_values)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = None, num_updates: int = 4):
        if batch_size is None:
            batch_size = RL_CONFIG['BATCH_SIZE']
        
        with self.lock:
            if len(self.memory) < batch_size:
                return
        
        for _ in range(num_updates):
            with self.lock:
                batch = random.sample(self.memory, batch_size)
            
            if TORCH_AVAILABLE:
                self._replay_torch(batch)
            else:
                self._replay_sklearn(batch)
        
        if self.epsilon > RL_CONFIG['EPSILON_MIN']:
            self.epsilon *= RL_CONFIG['EPSILON_DECAY']
    
    def _replay_torch(self, batch):
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
        X = np.array([s for s, _, _, _, _ in batch])
        current_q = self.model.predict(X)
        
        for i, (_, action, reward, next_state, done) in enumerate(batch):
            if done:
                target = reward
            else:
                next_q = self.model.predict(next_state.reshape(1, -1))[0]
                target = reward + RL_CONFIG['GAMMA'] * np.max(next_q)
            current_q[i, action] = target
        
        self.model.fit(X, current_q)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
            }, path)
        else:
            with open(path.replace('.pt', '.pkl'), 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'epsilon': self.epsilon,
                }, f)
        print(f"✅ {self.name} saved to {path}")
    
    def load(self, path: str) -> bool:
        if TORCH_AVAILABLE and os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', RL_CONFIG['EPSILON_MIN'])
            print(f"✅ {self.name} loaded from {path}")
            return True
        elif os.path.exists(path.replace('.pt', '.pkl')):
            with open(path.replace('.pt', '.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.epsilon = data.get('epsilon', RL_CONFIG['EPSILON_MIN'])
            print(f"✅ {self.name} loaded")
            return True
        return False


# ============================================================================
# CCP AGENT
# ============================================================================
class CCPAgent(BaseAgent):
    """
    CCP learns optimal margin requirements to minimize systemic risk.
    
    State (15 features):
        - System: avg health, num failed, total assets, total margin, stress
        - Bank: health, size, equity ratio, liquidity, margin ratio, any_failed
        - Trade: size ratio, direction, size, is_failed
    
    Actions (5): LOW, MEDIUM, HIGH, VERY_HIGH, REJECT
    """
    
    MARGIN_LEVELS = [0.05, 0.10, 0.20, 0.35, None]
    STATE_SIZE = 15
    ACTION_SIZE = 5
    
    def __init__(self):
        super().__init__(self.STATE_SIZE, self.ACTION_SIZE, "CCP Model")
        self.decisions = {d: 0 for d in MarginDecision}
    
    def get_state(self, contagion: BankingNetworkContagion, 
                  bank_name: str, trade_size: float, trade_direction: int) -> np.ndarray:
        """Extract state features for CCP decision"""
        graph = contagion.graph
        all_banks = list(graph.keys())
        num_banks = len(all_banks)
        
        num_failed = len(contagion.failed_banks)
        active_banks = [b for b in all_banks if b not in contagion.failed_banks]
        
        if active_banks:
            avg_health = np.mean([contagion.get_bank_health(b) for b in active_banks])
            total_assets = sum(contagion.bank_states[b].get('Total_Assets', 0) for b in active_banks)
            total_margin = sum(contagion.margin_states.get(b, 0) for b in active_banks)
        else:
            avg_health, total_assets, total_margin = 0, 0, 0
        
        bank_state = contagion.bank_states.get(bank_name, {})
        bank_health = contagion.get_bank_health(bank_name)
        bank_assets = bank_state.get('Total_Assets', 0)
        bank_equity = bank_state.get('Equity', 0)
        bank_hqla = bank_state.get('HQLA', 0)
        bank_margin = contagion.margin_states.get(bank_name, 0)
        
        equity_ratio = bank_equity / max(bank_assets, 1)
        liquidity_ratio = bank_hqla / max(bank_assets, 1)
        margin_ratio = bank_margin / max(bank_assets * 0.1, 1)
        trade_ratio = trade_size / max(bank_hqla, 1)
        stress = 1.0 - (avg_health / 100.0) if avg_health > 0 else 1.0
        
        return np.array([
            avg_health / 100.0,
            num_failed / num_banks,
            total_assets / 10000.0,
            total_margin / max(total_assets * 0.05, 1),
            stress,
            bank_health / 100.0,
            bank_assets / 1000.0,
            equity_ratio,
            liquidity_ratio,
            margin_ratio,
            num_failed > 0,
            trade_ratio,
            trade_direction,
            trade_size / 100.0,
            bank_name in contagion.failed_banks,
        ], dtype=np.float32)
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> Tuple[MarginDecision, float]:
        """Get margin decision"""
        action_idx = self.get_action_idx(state, explore)
        decision = MarginDecision(action_idx)
        self.decisions[decision] += 1
        margin = self.MARGIN_LEVELS[action_idx]
        return decision, margin


# ============================================================================
# BANK AGENT
# ============================================================================
class BankAgent(BaseAgent):
    """
    Bank learns optimal trading behavior to maximize profit while surviving.
    
    State (12 features):
        - Own: health, assets, equity_ratio, liquidity_ratio, margin_buffer
        - System: avg_health, num_failed_ratio, stress
        - Market: recent_volatility, shock_indicator
        - Position: current_exposure, pending_margin
    
    Actions (7): HOLD, SMALL_BUY, SMALL_SELL, MEDIUM_BUY, MEDIUM_SELL, LARGE_BUY, LARGE_SELL
    """
    
    TRADE_SIZES = {
        BankAction.HOLD: (0, 0, 0),           # (min, max, direction)
        BankAction.SMALL_BUY: (0.02, 0.05, 1),
        BankAction.SMALL_SELL: (0.02, 0.05, -1),
        BankAction.MEDIUM_BUY: (0.05, 0.10, 1),
        BankAction.MEDIUM_SELL: (0.05, 0.10, -1),
        BankAction.LARGE_BUY: (0.10, 0.15, 1),
        BankAction.LARGE_SELL: (0.10, 0.15, -1),
    }
    
    STATE_SIZE = 12
    ACTION_SIZE = 7
    
    def __init__(self):
        super().__init__(self.STATE_SIZE, self.ACTION_SIZE, "Bank Model")
        self.decisions = {a: 0 for a in BankAction}
    
    def get_state(self, contagion: BankingNetworkContagion, bank_name: str) -> np.ndarray:
        """Extract state for bank trading decision"""
        graph = contagion.graph
        all_banks = list(graph.keys())
        num_banks = len(all_banks)
        
        num_failed = len(contagion.failed_banks)
        active_banks = [b for b in all_banks if b not in contagion.failed_banks]
        
        if active_banks:
            avg_health = np.mean([contagion.get_bank_health(b) for b in active_banks])
        else:
            avg_health = 0
        
        bank_state = contagion.bank_states.get(bank_name, {})
        bank_health = contagion.get_bank_health(bank_name)
        bank_assets = bank_state.get('Total_Assets', 0)
        bank_equity = bank_state.get('Equity', 0)
        bank_hqla = bank_state.get('HQLA', 0)
        bank_margin = contagion.margin_states.get(bank_name, 0)
        
        equity_ratio = bank_equity / max(bank_assets, 1)
        liquidity_ratio = bank_hqla / max(bank_assets, 1)
        margin_ratio = bank_margin / max(bank_assets * 0.1, 1)
        stress = 1.0 - (avg_health / 100.0) if avg_health > 0 else 1.0
        
        # Estimate recent volatility (simplified)
        vol = contagion.graph[bank_name]['attributes'].get('Stock_Volatility', 0.02)
        
        return np.array([
            # Own bank (5)
            bank_health / 100.0,
            bank_assets / 1000.0,
            equity_ratio,
            liquidity_ratio,
            margin_ratio,
            # System (3)
            avg_health / 100.0,
            num_failed / num_banks,
            stress,
            # Market (2)
            vol * 10,  # Scale volatility
            num_failed > num_banks * 0.1,  # Crisis indicator
            # Position (2)
            bank_margin / max(bank_hqla, 1),  # Margin exposure
            bank_hqla / max(bank_assets * 0.1, 1),  # Liquidity headroom
        ], dtype=np.float32)
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> Tuple[BankAction, float, int]:
        """Get trading decision: action, size, direction"""
        action_idx = self.get_action_idx(state, explore)
        action = BankAction(action_idx)
        self.decisions[action] += 1
        
        min_pct, max_pct, direction = self.TRADE_SIZES[action]
        if action == BankAction.HOLD:
            return action, 0.0, 0
        
        size_pct = random.uniform(min_pct, max_pct)
        return action, size_pct, direction


# ============================================================================
# DUAL TRAINING ENVIRONMENT
# ============================================================================
class DualTrainingEnvironment:
    """Training environment for both CCP and Bank agents"""
    
    def __init__(self, data_dir: str = './dataset'):
        print("=" * 70)
        print("DUAL RL TRAINING: CCP + BANK POLICY")
        print("=" * 70)
        
        # Load data
        self.bank_attrs = load_bank_attributes(f'{data_dir}/us_banks_top50_nodes_final.csv')
        self.stock_prices, self.stock_timeseries = load_stock_prices(
            f'{data_dir}/stocks_data_long.csv', num_stocks=15
        )
        
        self.bank_list = list(self.bank_attrs.keys())
        self.num_banks = len(self.bank_list)
        
        # Initialize BOTH agents
        self.ccp = CCPAgent()
        self.bank_agent = BankAgent()
        
        # Metrics
        self.episode_metrics = []
        
        print(f"\n✅ Environment ready:")
        print(f"   Banks: {self.num_banks}")
        print(f"   Stocks: {len(self.stock_prices)}")
        print(f"   CCP State Size: {CCPAgent.STATE_SIZE}, Actions: {CCPAgent.ACTION_SIZE}")
        print(f"   Bank State Size: {BankAgent.STATE_SIZE}, Actions: {BankAgent.ACTION_SIZE}")
        print(f"   Episodes: {RL_CONFIG['NUM_EPISODES']}")
    
    def reset_episode(self) -> BankingNetworkContagion:
        """Create fresh simulation"""
        graph = generate_random_graph_with_sccs(
            self.bank_attrs, num_sccs=4, prob_intra=0.4, prob_inter=0.05
        )
        
        holdings = distribute_shares(self.bank_attrs, self.stock_prices)
        for bank in graph:
            if bank in holdings:
                graph[bank]['holdings'] = holdings[bank]
        
        margin_reqs = generate_margin_requirements(self.bank_attrs)
        return BankingNetworkContagion(graph, self.stock_prices, margin_reqs)
    
    def run_episode(self, train: bool = True) -> Dict:
        """Run one episode with both agents learning"""
        contagion = self.reset_episode()
        
        ccp_total_reward = 0.0
        bank_total_reward = 0.0
        total_trades = 0
        approved_trades = 0
        rejected_trades = 0
        total_profit = 0.0
        
        # Track bank experiences for deferred learning
        bank_experiences = []
        
        for step in range(RL_CONFIG['STEPS_PER_EPISODE']):
            ccp_step_reward = 0.0
            failures_before = len(contagion.failed_banks)
            
            for bank in self.bank_list:
                if bank in contagion.failed_banks:
                    continue
                
                bank_state_before = contagion.bank_states[bank].copy()
                assets_before = bank_state_before.get('Total_Assets', 0)
                hqla = bank_state_before.get('HQLA', 0)
                
                if hqla <= 0:
                    continue
                
                # BANK AGENT decides trade
                bank_state = self.bank_agent.get_state(contagion, bank)
                bank_action, size_pct, direction = self.bank_agent.get_action(bank_state, explore=train)
                
                bank_reward = 0.0
                
                if bank_action == BankAction.HOLD:
                    # Small reward for holding when unhealthy
                    bank_health = contagion.get_bank_health(bank)
                    if bank_health < 40:
                        bank_reward += RL_CONFIG['BANK_REWARD_CONSERVATIVE']
                    
                    # Store hold experience
                    if train:
                        bank_experiences.append({
                            'bank': bank,
                            'state': bank_state,
                            'action': bank_action.value,
                            'reward': bank_reward,
                            'step': step
                        })
                    bank_total_reward += bank_reward
                    continue
                
                # Calculate trade size
                trade_size = hqla * size_pct
                total_trades += 1
                
                # CCP AGENT decides margin
                ccp_state = self.ccp.get_state(contagion, bank, trade_size, direction)
                ccp_decision, margin = self.ccp.get_action(ccp_state, explore=train)
                
                if ccp_decision == MarginDecision.REJECT or margin is None:
                    rejected_trades += 1
                    bank_health = contagion.get_bank_health(bank)
                    if bank_health > 50:
                        ccp_step_reward -= RL_CONFIG['CCP_PENALTY_REJECT']
                    # Bank gets small penalty for rejected trade
                    bank_reward -= 1.0
                else:
                    approved_trades += 1
                    ccp_step_reward += RL_CONFIG['CCP_REWARD_TRADE']
                    
                    # Execute trade
                    profit = self._execute_trade(contagion, bank, trade_size, direction, margin)
                    total_profit += profit
                    
                    # Bank reward based on profit/loss
                    if profit > 0:
                        bank_reward += RL_CONFIG['BANK_REWARD_PROFIT'] * min(profit / 10, 5)
                    else:
                        bank_reward -= RL_CONFIG['BANK_PENALTY_LOSS'] * min(abs(profit) / 10, 5)
                    
                    # Penalty for risky trading when unhealthy
                    bank_health = contagion.get_bank_health(bank)
                    if bank_health < 40 and size_pct > 0.05:
                        bank_reward -= RL_CONFIG['BANK_PENALTY_RISKY']
                    
                    # CCP penalty for high margin on healthy bank
                    if margin >= 0.20 and bank_health > 60:
                        ccp_step_reward -= RL_CONFIG['CCP_PENALTY_HIGH_MARGIN']
                
                # Store CCP experience
                if train:
                    ccp_next_state = self.ccp.get_state(contagion, bank, 0, 0)
                    self.ccp.remember(ccp_state, ccp_decision.value, ccp_step_reward, ccp_next_state, False)
                
                # Store bank experience (deferred until we know if bank failed)
                if train:
                    bank_experiences.append({
                        'bank': bank,
                        'state': bank_state,
                        'action': bank_action.value,
                        'reward': bank_reward,
                        'step': step
                    })
                
                bank_total_reward += bank_reward
            
            # Random market shock
            if random.random() < RL_CONFIG['SHOCK_PROBABILITY']:
                self._apply_shock(contagion)
            
            # Check failures
            self._check_failures(contagion)
            
            # CCP failure penalties
            failures_after = len(contagion.failed_banks)
            new_failures = failures_after - failures_before
            
            if new_failures > 0:
                ccp_step_reward -= RL_CONFIG['CCP_PENALTY_FAILURE'] * new_failures
                if new_failures > 1:
                    ccp_step_reward -= RL_CONFIG['CCP_PENALTY_CASCADE'] * (new_failures - 1)
                
                # Huge penalty for banks that failed
                for exp in bank_experiences:
                    if exp['bank'] in contagion.failed_banks and exp['step'] == step:
                        exp['reward'] -= RL_CONFIG['BANK_PENALTY_FAILURE']
            
            elif failures_after == 0:
                ccp_step_reward += RL_CONFIG['CCP_REWARD_STABILITY'] * 0.1
                # Survival bonus for all active banks
                for exp in bank_experiences:
                    if exp['step'] == step:
                        exp['reward'] += RL_CONFIG['BANK_REWARD_SURVIVAL'] * 0.1
            
            ccp_total_reward += ccp_step_reward
            
            # Early termination
            if failures_after > self.num_banks * 0.5:
                break
        
        # Process bank experiences
        if train:
            for i, exp in enumerate(bank_experiences):
                # Get next state (or terminal)
                if exp['bank'] in contagion.failed_banks:
                    next_state = np.zeros(BankAgent.STATE_SIZE, dtype=np.float32)
                    done = True
                else:
                    next_state = self.bank_agent.get_state(contagion, exp['bank'])
                    done = False
                
                self.bank_agent.remember(exp['state'], exp['action'], exp['reward'], next_state, done)
        
        # Train both models
        if train:
            self.ccp.replay()
            self.bank_agent.replay()
        
        # Final rewards - scaled by survival rate
        final_failed = len(contagion.failed_banks)
        survival_rate = (self.num_banks - final_failed) / self.num_banks  # 0.0 to 1.0
        
        # Survival-based reward: more banks survive = more reward
        # Max reward when 0 failures, scales down linearly
        survival_reward = RL_CONFIG['CCP_REWARD_STABILITY'] * survival_rate * 10
        ccp_total_reward += survival_reward
        
        # Bonus tiers
        if final_failed == 0:
            ccp_total_reward += RL_CONFIG['CCP_REWARD_STABILITY'] * 5  # Perfect bonus
        elif final_failed <= 3:
            ccp_total_reward += RL_CONFIG['CCP_REWARD_STABILITY'] * 2  # Excellent
        elif final_failed <= 5:
            ccp_total_reward += RL_CONFIG['CCP_REWARD_STABILITY']      # Good
        
        # Collapse penalties (still apply)
        if final_failed > self.num_banks * 0.5:
            ccp_total_reward -= RL_CONFIG['CCP_PENALTY_COLLAPSE'] * 2
        elif final_failed > self.num_banks * 0.25:
            ccp_total_reward -= RL_CONFIG['CCP_PENALTY_COLLAPSE']
        
        return {
            'ccp_reward': ccp_total_reward,
            'bank_reward': bank_total_reward,
            'failed_banks': final_failed,
            'total_trades': total_trades,
            'approved': approved_trades,
            'rejected': rejected_trades,
            'approval_rate': approved_trades / max(total_trades, 1),
            'total_profit': total_profit,
            'avg_health': np.mean([contagion.get_bank_health(b) for b in self.bank_list 
                                   if b not in contagion.failed_banks]) if final_failed < self.num_banks else 0
        }
    
    def _execute_trade(self, contagion, bank, size, direction, margin) -> float:
        """Execute trade, return profit/loss"""
        vol = contagion.graph[bank]['attributes'].get('Stock_Volatility', 0.02)
        price_change = np.random.normal(0, vol)
        margin_cost = size * margin * 0.01
        
        if direction == 1:  # Buy
            profit = size * price_change - margin_cost
            contagion.bank_states[bank]['HQLA'] -= size
        else:  # Sell
            profit = -size * price_change - margin_cost
            contagion.bank_states[bank]['HQLA'] += size * 0.95
        
        contagion.bank_states[bank]['Total_Assets'] += profit
        contagion.bank_states[bank]['Equity'] += profit
        return profit
    
    def _apply_shock(self, contagion):
        """Apply market shock"""
        shock_pct = random.uniform(*RL_CONFIG['SHOCK_RANGE'])
        
        for bank in self.bank_list:
            if bank not in contagion.failed_banks:
                loss = contagion.bank_states[bank]['Total_Assets'] * shock_pct * random.uniform(0.3, 1.0)
                actual_loss, _ = contagion._use_margin_buffer(bank, loss)
                contagion.bank_states[bank]['Total_Assets'] -= actual_loss
                contagion.bank_states[bank]['Equity'] -= actual_loss
    
    def _check_failures(self, contagion):
        """Check for bank failures"""
        for bank in self.bank_list:
            if bank not in contagion.failed_banks:
                if contagion.get_bank_health(bank) < RL_CONFIG['FAILURE_THRESHOLD']:
                    contagion.failed_banks.add(bank)
    
    def train(self, num_episodes: int = None):
        """Main training loop"""
        if num_episodes is None:
            num_episodes = RL_CONFIG['NUM_EPISODES']
        
        print(f"\n🚀 Training BOTH models for {num_episodes} episodes...")
        print("=" * 70)
        
        for episode in range(num_episodes):
            result = self.run_episode(train=True)
            self.episode_metrics.append(result)
            
            print(f"Ep {episode+1:4d} | "
                  f"CCP: {result['ccp_reward']:+8.1f} | "
                  f"Bank: {result['bank_reward']:+7.1f} | "
                  f"Failed: {result['failed_banks']:2d} | "
                  f"Appr: {result['approval_rate']*100:5.1f}% | "
                  f"ε: {self.ccp.epsilon:.3f}")
        
        print("=" * 70)
        self.ccp.save(RL_CONFIG['CCP_MODEL_PATH'])
        self.bank_agent.save(RL_CONFIG['BANK_MODEL_PATH'])
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary"""
        print("\n📊 TRAINING SUMMARY")
        print("-" * 50)
        
        # CCP decisions
        print("\nCCP Margin Decisions:")
        total_ccp = sum(self.ccp.decisions.values())
        for d in MarginDecision:
            pct = self.ccp.decisions[d] / max(total_ccp, 1) * 100
            margin = CCPAgent.MARGIN_LEVELS[d.value]
            label = f"{margin*100:.0f}%" if margin else "REJECT"
            print(f"  {d.name:10s} ({label:>6s}): {self.ccp.decisions[d]:6d} ({pct:5.1f}%)")
        
        # Bank decisions
        print("\nBank Trading Decisions:")
        total_bank = sum(self.bank_agent.decisions.values())
        for a in BankAction:
            pct = self.bank_agent.decisions[a] / max(total_bank, 1) * 100
            print(f"  {a.name:12s}: {self.bank_agent.decisions[a]:6d} ({pct:5.1f}%)")
        
        if self.episode_metrics:
            final = self.episode_metrics[-100:] if len(self.episode_metrics) >= 100 else self.episode_metrics
            print(f"\nFinal 100 episodes:")
            print(f"  Avg CCP Reward: {np.mean([m['ccp_reward'] for m in final]):+.1f}")
            print(f"  Avg Bank Reward: {np.mean([m['bank_reward'] for m in final]):+.1f}")
            print(f"  Avg Failures: {np.mean([m['failed_banks'] for m in final]):.1f}")
            print(f"  Avg Approval Rate: {np.mean([m['approval_rate'] for m in final])*100:.1f}%")
    
    def evaluate(self, num_episodes: int = 20):
        """Evaluate trained models"""
        print(f"\n🔍 Evaluating over {num_episodes} episodes...")
        
        results = []
        for _ in range(num_episodes):
            result = self.run_episode(train=False)
            results.append(result)
        
        print(f"  Avg CCP Reward: {np.mean([r['ccp_reward'] for r in results]):+.1f}")
        print(f"  Avg Bank Reward: {np.mean([r['bank_reward'] for r in results]):+.1f}")
        print(f"  Avg Failures: {np.mean([r['failed_banks'] for r in results]):.1f}")
        print(f"  Avg Approval Rate: {np.mean([r['approval_rate'] for r in results])*100:.1f}%")
        print(f"  Avg Final Health: {np.mean([r['avg_health'] for r in results]):.1f}")
        
        return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("DUAL REINFORCEMENT LEARNING")
    print("  Model 1: CCP (margin optimization)")
    print("  Model 2: Bank (trading policy)")
    print("=" * 70)
    
    env = DualTrainingEnvironment()
    
    # Check for existing models
    ccp_loaded = env.ccp.load(RL_CONFIG['CCP_MODEL_PATH'])
    bank_loaded = env.bank_agent.load(RL_CONFIG['BANK_MODEL_PATH'])
    
    if ccp_loaded or bank_loaded:
        print("\nFound existing model(s). Evaluating...")
        env.evaluate(10)
        
        response = input("\nContinue training? (y/n): ").strip().lower()
        if response == 'y':
            env.train(num_episodes=200)
    else:
        env.train()
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    env.evaluate(20)
