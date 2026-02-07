"""
================================================================================
ALGO.PY - CCP Reinforcement Learning for Optimal Margin Requirements
================================================================================
Trains a single CCP agent to learn optimal margin requirements that minimize
systemic risk while allowing healthy market activity.

Banks use simple rule-based behavior (no RL) - only the CCP learns.

Supports:
  - GPU acceleration via PyTorch (auto-detected)
  - Parallel episode collection
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
    'NUM_EPISODES': 10000,
    'STEPS_PER_EPISODE': 100,
    'NUM_WORKERS': min(4, mp.cpu_count()),
    
    # RL hyperparameters
    'GAMMA': 0.95,
    'EPSILON_START': 1.0,
    'EPSILON_MIN': 0.01,       # Lower minimum for more exploitation
    'EPSILON_DECAY': 0.9995,   # Much slower decay
    'MEMORY_SIZE': 200000,     # Larger buffer
    'BATCH_SIZE': 1024 if TORCH_AVAILABLE else 64,  # Much bigger batches for GPU
    'LEARNING_RATE': 0.0005,   # Slightly lower for stability
    'TRAIN_FREQ': 4,           # Train every N steps (batch multiple experiences)
    
    # CCP Rewards (tuned to strongly discourage systemic failure)
    'CCP_PENALTY_FAILURE': 100.0,     # Per failed bank (was 10)
    'CCP_PENALTY_CASCADE': 50.0,      # Extra per cascade failure (was 20)
    'CCP_PENALTY_COLLAPSE': 2000.0,   # NEW: When >25% banks fail
    'CCP_REWARD_STABILITY': 20.0,     # Bonus for no failures (was 5)
    'CCP_REWARD_TRADE': 0.5,          # Small reward per approved trade (was 0.1)
    'CCP_PENALTY_REJECT': 5.0,        # Penalty for rejecting healthy trades (was 0.2)
    'CCP_PENALTY_HIGH_MARGIN': 1.0,   # Penalty for excessive margins (was 0.3)
    
    # Simulation
    'SHOCK_PROBABILITY': 0.15,
    'SHOCK_RANGE': (0.10, 0.40),
    'FAILURE_THRESHOLD': 20.0,
    
    # Save path
    'MODEL_PATH': 'models/ccp_policy.pt',
}


class MarginDecision(Enum):
    """CCP's margin decisions"""
    LOW = 0       # 5% margin - risky but enables trading
    MEDIUM = 1    # 10% margin - balanced
    HIGH = 2      # 20% margin - conservative
    VERY_HIGH = 3 # 35% margin - very conservative
    REJECT = 4    # Reject trade entirely


# ============================================================================
# PYTORCH Q-NETWORK (Larger for GPU utilization)
# ============================================================================
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, state_size: int, action_size: int):
            super().__init__()
            # Bigger network to actually use GPU
            self.network = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )
            self.to(DEVICE)
            
            # Verify on GPU
            if DEVICE.type == 'cuda':
                print(f"   Network on GPU: {next(self.parameters()).is_cuda}")
        
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
# CCP RL AGENT
# ============================================================================
class CCPAgent:
    """
    CCP learns optimal margin requirements to minimize systemic risk.
    
    State (15 features):
        - System-level: avg health, num failed, total assets, total margin
        - Bank-level: requesting bank's health, size, equity ratio, liquidity
        - Trade-level: size, direction
        - Market: recent volatility, shock indicator
    
    Actions (5):
        - LOW (5%), MEDIUM (10%), HIGH (20%), VERY_HIGH (35%), REJECT
    """
    
    MARGIN_LEVELS = [0.05, 0.10, 0.20, 0.35, None]  # None = reject
    STATE_SIZE = 15
    ACTION_SIZE = 5
    
    def __init__(self):
        self.lock = threading.Lock()
        
        if TORCH_AVAILABLE:
            self.model = QNetwork(self.STATE_SIZE, self.ACTION_SIZE)
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
        
        # Stats
        self.decisions = {d: 0 for d in MarginDecision}
        self.episode_rewards = []
    
    def _init_sklearn(self):
        X = np.random.randn(10, self.STATE_SIZE)
        y = np.random.randn(10, self.ACTION_SIZE)
        self.model.fit(X, y)
    
    def get_state(self, contagion: BankingNetworkContagion, 
                  bank_name: str, trade_size: float, trade_direction: int) -> np.ndarray:
        """Extract state features for CCP decision"""
        graph = contagion.graph
        all_banks = list(graph.keys())
        num_banks = len(all_banks)
        
        # System features
        num_failed = len(contagion.failed_banks)
        active_banks = [b for b in all_banks if b not in contagion.failed_banks]
        
        if active_banks:
            avg_health = np.mean([contagion.get_bank_health(b) for b in active_banks])
            total_assets = sum(contagion.bank_states[b].get('Total_Assets', 0) for b in active_banks)
            total_margin = sum(contagion.margin_states.get(b, 0) for b in active_banks)
        else:
            avg_health, total_assets, total_margin = 0, 0, 0
        
        # Bank features
        bank_state = contagion.bank_states.get(bank_name, {})
        bank_health = contagion.get_bank_health(bank_name)
        bank_assets = bank_state.get('Total_Assets', 0)
        bank_equity = bank_state.get('Equity', 0)
        bank_hqla = bank_state.get('HQLA', 0)
        bank_margin = contagion.margin_states.get(bank_name, 0)
        
        equity_ratio = bank_equity / max(bank_assets, 1)
        liquidity_ratio = bank_hqla / max(bank_assets, 1)
        margin_ratio = bank_margin / max(bank_assets * 0.1, 1)
        
        # Trade features
        trade_ratio = trade_size / max(bank_hqla, 1)
        
        # Market stress indicator
        stress = 1.0 - (avg_health / 100.0) if avg_health > 0 else 1.0
        
        return np.array([
            # System (5)
            avg_health / 100.0,
            num_failed / num_banks,
            total_assets / 10000.0,  # Normalize to ~1
            total_margin / max(total_assets * 0.05, 1),
            stress,
            # Bank (6)
            bank_health / 100.0,
            bank_assets / 1000.0,
            equity_ratio,
            liquidity_ratio,
            margin_ratio,
            num_failed > 0,  # Any failures yet?
            # Trade (4)
            trade_ratio,
            trade_direction,  # 1=buy, -1=sell, 0=hold
            trade_size / 100.0,
            bank_name in contagion.failed_banks,  # Is bank already failed?
        ], dtype=np.float32)
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> Tuple[MarginDecision, float]:
        """Choose margin level using epsilon-greedy"""
        if explore and random.random() < self.epsilon:
            action_idx = random.randint(0, self.ACTION_SIZE - 1)
        else:
            q_values = self.model.predict(state.reshape(1, -1))[0]
            action_idx = np.argmax(q_values)
        
        decision = MarginDecision(action_idx)
        self.decisions[decision] += 1
        
        margin = self.MARGIN_LEVELS[action_idx]
        return decision, margin
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = None, num_updates: int = 4):
        """Train on batches - multiple updates to utilize GPU"""
        if batch_size is None:
            batch_size = RL_CONFIG['BATCH_SIZE']
        
        with self.lock:
            if len(self.memory) < batch_size:
                return
        
        # Multiple training updates per call to better use GPU
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
    
    def save(self, path: str = None):
        if path is None:
            path = RL_CONFIG['MODEL_PATH']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'decisions': dict(self.decisions),
            }, path)
        else:
            with open(path.replace('.pt', '.pkl'), 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'epsilon': self.epsilon,
                    'decisions': dict(self.decisions),
                }, f)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str = None):
        if path is None:
            path = RL_CONFIG['MODEL_PATH']
        
        if TORCH_AVAILABLE and os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', RL_CONFIG['EPSILON_MIN'])
            print(f"✅ Model loaded from {path}")
            return True
        elif os.path.exists(path.replace('.pt', '.pkl')):
            with open(path.replace('.pt', '.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.epsilon = data.get('epsilon', RL_CONFIG['EPSILON_MIN'])
            print(f"✅ Model loaded")
            return True
        return False


# ============================================================================
# TRAINING ENVIRONMENT
# ============================================================================
class CCPTrainingEnvironment:
    """Training environment for CCP agent only"""
    
    def __init__(self, data_dir: str = './dataset'):
        print("=" * 70)
        print("CCP MARGIN OPTIMIZATION - RL TRAINING")
        print("=" * 70)
        
        # Load data
        self.bank_attrs = load_bank_attributes(f'{data_dir}/us_banks_top50_nodes_final.csv')
        self.stock_prices, self.stock_timeseries = load_stock_prices(
            f'{data_dir}/stocks_data_long.csv', num_stocks=15
        )
        
        self.bank_list = list(self.bank_attrs.keys())
        self.num_banks = len(self.bank_list)
        
        # Initialize CCP agent
        self.ccp = CCPAgent()
        
        # Metrics
        self.episode_metrics = []
        
        # Training stats
        num_episodes = RL_CONFIG['NUM_EPISODES']
        steps = RL_CONFIG['STEPS_PER_EPISODE']
        
        print(f"\n✅ Environment ready:")
        print(f"   Banks: {self.num_banks}")
        print(f"   Stocks: {len(self.stock_prices)}")
        print(f"   Episodes: {num_episodes}")
        print(f"   Steps/Episode: {steps}")
        print(f"   Training decisions: ~{num_episodes * steps * self.num_banks:,}")
    
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
        """Run one episode"""
        contagion = self.reset_episode()
        
        total_reward = 0.0
        total_trades = 0
        approved_trades = 0
        rejected_trades = 0
        initial_failed = 0
        
        for step in range(RL_CONFIG['STEPS_PER_EPISODE']):
            step_reward = 0.0
            failures_before = len(contagion.failed_banks)
            
            # Each bank tries to make a trade
            for bank in self.bank_list:
                if bank in contagion.failed_banks:
                    continue
                
                # Bank decides trade (rule-based, not learned)
                hqla = contagion.bank_states[bank].get('HQLA', 0)
                if hqla <= 0:
                    continue
                
                # Random trade decision
                if random.random() < 0.3:  # 30% chance to trade
                    trade_size = hqla * random.uniform(0.02, 0.10)
                    direction = random.choice([-1, 1])  # Buy or sell
                else:
                    continue  # Hold
                
                # Get CCP state and action
                state = self.ccp.get_state(contagion, bank, trade_size, direction)
                decision, margin = self.ccp.get_action(state, explore=train)
                
                total_trades += 1
                
                if decision == MarginDecision.REJECT or margin is None:
                    rejected_trades += 1
                    # Penalty for rejecting trade from healthy bank
                    bank_health = contagion.get_bank_health(bank)
                    if bank_health > 50:
                        step_reward -= RL_CONFIG['CCP_PENALTY_REJECT']
                else:
                    approved_trades += 1
                    step_reward += RL_CONFIG['CCP_REWARD_TRADE']
                    
                    # Execute trade (simplified)
                    self._execute_trade(contagion, bank, trade_size, direction, margin)
                    
                    # Penalty for very high margins on healthy banks
                    if margin >= 0.20 and contagion.get_bank_health(bank) > 60:
                        step_reward -= RL_CONFIG['CCP_PENALTY_HIGH_MARGIN']
                
                # Store experience
                if train:
                    next_state = self.ccp.get_state(contagion, bank, 0, 0)
                    self.ccp.remember(state, decision.value, step_reward, next_state, False)
            
            # Random market shock
            if random.random() < RL_CONFIG['SHOCK_PROBABILITY']:
                self._apply_shock(contagion)
            
            # Check for failures
            self._check_failures(contagion)
            
            # Failure penalties
            failures_after = len(contagion.failed_banks)
            new_failures = failures_after - failures_before
            
            if new_failures > 0:
                step_reward -= RL_CONFIG['CCP_PENALTY_FAILURE'] * new_failures
                if new_failures > 1:
                    step_reward -= RL_CONFIG['CCP_PENALTY_CASCADE'] * (new_failures - 1)
            elif failures_after == 0:
                step_reward += RL_CONFIG['CCP_REWARD_STABILITY'] * 0.1
            
            total_reward += step_reward
            
            # Early termination
            if failures_after > self.num_banks * 0.5:
                break
        
        # Train after episode
        if train:
            self.ccp.replay()
        
        # Final stability bonus / collapse penalty
        final_failed = len(contagion.failed_banks)
        if final_failed == 0:
            total_reward += RL_CONFIG['CCP_REWARD_STABILITY'] * 10  # Big bonus for perfect run
        elif final_failed <= 5:
            total_reward += RL_CONFIG['CCP_REWARD_STABILITY']  # Small bonus for low failures
        elif final_failed > self.num_banks * 0.25:  # >25% collapse
            total_reward -= RL_CONFIG['CCP_PENALTY_COLLAPSE']  # Catastrophic penalty
        elif final_failed > self.num_banks * 0.5:  # >50% collapse
            total_reward -= RL_CONFIG['CCP_PENALTY_COLLAPSE'] * 2  # Even worse
        
        return {
            'reward': total_reward,
            'failed_banks': final_failed,
            'total_trades': total_trades,
            'approved': approved_trades,
            'rejected': rejected_trades,
            'approval_rate': approved_trades / max(total_trades, 1),
            'avg_health': np.mean([contagion.get_bank_health(b) for b in self.bank_list 
                                   if b not in contagion.failed_banks]) if final_failed < self.num_banks else 0
        }
    
    def _execute_trade(self, contagion, bank, size, direction, margin):
        """Execute trade with margin requirement"""
        vol = contagion.graph[bank]['attributes'].get('Stock_Volatility', 0.02)
        price_change = np.random.normal(0, vol)
        
        # Apply margin cost
        margin_cost = size * margin
        
        if direction == 1:  # Buy
            profit = size * price_change - margin_cost * 0.01
            contagion.bank_states[bank]['HQLA'] -= size
        else:  # Sell
            profit = -size * price_change - margin_cost * 0.01
            contagion.bank_states[bank]['HQLA'] += size * 0.95
        
        contagion.bank_states[bank]['Total_Assets'] += profit
        contagion.bank_states[bank]['Equity'] += profit
    
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
        
        print(f"\n🚀 Training CCP for {num_episodes} episodes...")
        print("=" * 70)
        
        for episode in range(num_episodes):
            result = self.run_episode(train=True)
            self.episode_metrics.append(result)
            
            # Print every episode
            print(f"Ep {episode+1:4d} | "
                  f"Reward: {result['reward']:+8.1f} | "
                  f"Failed: {result['failed_banks']:2d} | "
                  f"Approval: {result['approval_rate']*100:5.1f}% | "
                  f"ε: {self.ccp.epsilon:.3f}")
        
        print("=" * 70)
        self.ccp.save()
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary"""
        print("\n📊 TRAINING SUMMARY")
        print("-" * 40)
        
        decisions = self.ccp.decisions
        total = sum(decisions.values())
        
        print("Margin Decisions:")
        for d in MarginDecision:
            pct = decisions[d] / max(total, 1) * 100
            margin = CCPAgent.MARGIN_LEVELS[d.value]
            label = f"{margin*100:.0f}%" if margin else "REJECT"
            print(f"  {d.name:10s} ({label:>6s}): {decisions[d]:6d} ({pct:5.1f}%)")
        
        if self.episode_metrics:
            final = self.episode_metrics[-100:] if len(self.episode_metrics) >= 100 else self.episode_metrics
            print(f"\nFinal 100 episodes:")
            print(f"  Avg Reward: {np.mean([m['reward'] for m in final]):+.1f}")
            print(f"  Avg Failures: {np.mean([m['failed_banks'] for m in final]):.1f}")
            print(f"  Avg Approval Rate: {np.mean([m['approval_rate'] for m in final])*100:.1f}%")
    
    def evaluate(self, num_episodes: int = 20):
        """Evaluate trained model"""
        print(f"\n🔍 Evaluating over {num_episodes} episodes...")
        
        results = []
        for _ in range(num_episodes):
            result = self.run_episode(train=False)
            results.append(result)
        
        print(f"  Avg Reward: {np.mean([r['reward'] for r in results]):+.1f}")
        print(f"  Avg Failures: {np.mean([r['failed_banks'] for r in results]):.1f}")
        print(f"  Avg Approval Rate: {np.mean([r['approval_rate'] for r in results])*100:.1f}%")
        print(f"  Avg Final Health: {np.mean([r['avg_health'] for r in results]):.1f}")
        
        return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("CCP REINFORCEMENT LEARNING - MARGIN OPTIMIZATION")
    print("=" * 70)
    
    env = CCPTrainingEnvironment()
    
    # Check for existing model
    if env.ccp.load():
        print("\nFound existing model. Evaluating...")
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
