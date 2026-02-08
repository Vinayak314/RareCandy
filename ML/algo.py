"""
================================================================================
ALGO.PY - CCP-Only RL Training using algorithm.py Simulation
================================================================================
Trains a SINGLE CCP agent to learn optimal margin requirements.
Uses the existing BankingNetworkContagion from algorithm.py for simulation.

Banks use the built-in contagion behavior - only the CCP learns.
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
import pickle
import os
from collections import deque
from typing import Dict, List, Tuple, Optional
from enum import Enum
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

# Import existing simulation code from algorithm.py
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
    'NUM_EPISODES': 100000,
    'SHOCKS_PER_EPISODE': 10,  # Number of stock shocks per episode
    
    # RL hyperparameters
    'GAMMA': 0.95,
    'EPSILON_START': 1.0,
    'EPSILON_MIN': 0.01,
    'EPSILON_DECAY': 0.9995,
    'MEMORY_SIZE': 100000,
    'BATCH_SIZE': 512 if TORCH_AVAILABLE else 64,
    'LEARNING_RATE': 0.0003,
    
    # CCP Rewards
    'REWARD_PER_SURVIVOR': 50.0,      # Reward per bank that survives
    'PENALTY_PER_FAILURE': 100.0,     # Penalty per bank that fails
    'BONUS_ZERO_FAILURES': 500.0,     # Bonus if no banks fail
    'PENALTY_OVER_MARGIN': 10.0,      # Penalty for excessive margin (reduces trading)
    
    # Simulation
    'SHOCK_RANGE': (10, 40),          # Stock devaluation range (%)
    'FAILURE_THRESHOLD': 20.0,
    
    # Save path
    'MODEL_PATH': 'models/ccp_policy.pt',
}


class MarginLevel(Enum):
    """CCP margin level decisions"""
    VERY_LOW = 0   # 2% margin
    LOW = 1        # 4% margin
    MEDIUM = 2     # 6% margin
    HIGH = 3       # 8% margin
    VERY_HIGH = 4  # 10% margin


# ============================================================================
# PYTORCH Q-NETWORK
# ============================================================================
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, state_size: int, action_size: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )
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
# CCP AGENT
# ============================================================================
class CCPAgent:
    """
    CCP learns optimal margin requirements to minimize systemic failures.
    
    State features (10):
        - Avg bank health (0-1)
        - Avg equity ratio
        - Avg liquidity ratio  
        - Num banks in distress (health < 40)
        - Total system assets (normalized)
        - Stock volatility indicator
        - Previous failures this episode
        - Current margin level
        - Shock count this episode
        - System stress indicator
    
    Actions (5): VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH margin levels
    """
    
    MARGIN_RATES = [x/100 for x in range(1,101)]  # Margin as % of assets
    STATE_SIZE = 10
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
        self.decisions = {m: 0 for m in MarginLevel}
    
    def _init_sklearn(self):
        X = np.random.randn(10, self.STATE_SIZE)
        y = np.random.randn(10, self.ACTION_SIZE)
        self.model.fit(X, y)
    
    def get_state(self, contagion: BankingNetworkContagion, 
                  shock_count: int, prev_failures: int, current_margin_idx: int) -> np.ndarray:
        """Extract state from the simulation"""
        banks = list(contagion.graph.keys())
        active_banks = [b for b in banks if b not in contagion.failed_banks]
        num_banks = len(banks)
        
        if not active_banks:
            return np.zeros(self.STATE_SIZE, dtype=np.float32)
        
        # Compute stats
        healths = [contagion.get_bank_health(b) for b in active_banks]
        avg_health = np.mean(healths) / 100.0
        
        # Equity and liquidity ratios
        equity_ratios = []
        liquidity_ratios = []
        for b in active_banks:
            state = contagion.bank_states[b]
            assets = state.get('Total_Assets', 1)
            equity_ratios.append(state.get('Equity', 0) / max(assets, 1))
            liquidity_ratios.append(state.get('HQLA', 0) / max(assets, 1))
        
        avg_equity = np.mean(equity_ratios)
        avg_liquidity = np.mean(liquidity_ratios)
        
        # Banks in distress
        distressed = sum(1 for h in healths if h < 40)
        
        # Total assets
        total_assets = sum(contagion.bank_states[b].get('Total_Assets', 0) for b in active_banks)
        
        # Volatility (from attributes)
        vols = [contagion.graph[b]['attributes'].get('Stock_Volatility', 0.02) for b in active_banks]
        avg_vol = np.mean(vols)
        
        # Stress indicator
        stress = 1.0 - avg_health
        
        return np.array([
            avg_health,
            avg_equity,
            avg_liquidity,
            distressed / num_banks,
            total_assets / 10000.0,  # Normalize
            avg_vol * 10,
            prev_failures / num_banks,
            current_margin_idx / 4.0,
            shock_count / RL_CONFIG['SHOCKS_PER_EPISODE'],
            stress,
        ], dtype=np.float32)
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> Tuple[MarginLevel, float]:
        """Choose margin level"""
        if explore and random.random() < self.epsilon:
            action_idx = random.randint(0, self.ACTION_SIZE - 1)
        else:
            q_values = self.model.predict(state.reshape(1, -1))[0]
            action_idx = np.argmax(q_values)
        
        decision = MarginLevel(action_idx)
        self.decisions[decision] += 1
        margin_rate = self.MARGIN_RATES[action_idx]
        
        return decision, margin_rate, action_idx
    
    def remember(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = None):
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
                'decisions': {k.name: v for k, v in self.decisions.items()},
            }, path)
        else:
            with open(path.replace('.pt', '.pkl'), 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'epsilon': self.epsilon,
                    'decisions': {k.name: v for k, v in self.decisions.items()},
                }, f)
        print(f"✅ CCP model saved to {path}")
    
    def load(self, path: str = None) -> bool:
        if path is None:
            path = RL_CONFIG['MODEL_PATH']
        
        if TORCH_AVAILABLE and os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', RL_CONFIG['EPSILON_MIN'])
            print(f"✅ CCP model loaded from {path}")
            return True
        elif os.path.exists(path.replace('.pt', '.pkl')):
            with open(path.replace('.pt', '.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.epsilon = data.get('epsilon', RL_CONFIG['EPSILON_MIN'])
            print(f"✅ CCP model loaded")
            return True
        return False


# ============================================================================
# TRAINING ENVIRONMENT
# ============================================================================
class CCPTrainingEnvironment:
    """Training environment using algorithm.py's BankingNetworkContagion"""
    
    def __init__(self, data_dir: str = './dataset'):
        print("=" * 70)
        print("CCP RL TRAINING (using algorithm.py simulation)")
        print("=" * 70)
        
        # Load data
        self.bank_attrs = load_bank_attributes(f'{data_dir}/us_banks_top50_nodes_final.csv')
        self.stock_prices, self.stock_timeseries = load_stock_prices(
            f'{data_dir}/stocks_data_long.csv', num_stocks=15
        )
        
        self.bank_list = list(self.bank_attrs.keys())
        self.num_banks = len(self.bank_list)
        self.stock_list = list(self.stock_prices.keys())
        
        # Initialize CCP agent
        self.ccp = CCPAgent()
        
        # Metrics
        self.episode_metrics = []
        
        print(f"\n✅ Environment ready:")
        print(f"   Banks: {self.num_banks}")
        print(f"   Stocks: {len(self.stock_prices)}")
        print(f"   Episodes: {RL_CONFIG['NUM_EPISODES']}")
        print(f"   Shocks/Episode: {RL_CONFIG['SHOCKS_PER_EPISODE']}")
    
    def create_simulation(self, margin_rate: float) -> BankingNetworkContagion:
        """Create a fresh simulation with given margin rate"""
        # Generate network
        graph = generate_random_graph_with_sccs(
            self.bank_attrs, num_sccs=4, prob_intra=0.4, prob_inter=0.05
        )
        
        # Distribute stock holdings
        holdings = distribute_shares(self.bank_attrs, self.stock_prices)
        for bank in graph:
            if bank in holdings:
                graph[bank]['holdings'] = holdings[bank]
        
        # Generate margin requirements with specific rate
        margin_reqs = {}
        for bank, attrs in self.bank_attrs.items():
            margin_reqs[bank] = attrs['Total_Assets'] * margin_rate
        
        return BankingNetworkContagion(graph, self.stock_prices, margin_reqs)
    
    def run_episode(self, train: bool = True) -> Dict:
        """Run one episode: CCP sets margin, then shocks occur"""
        total_reward = 0.0
        total_failures = 0
        
        # Start with medium margin
        current_margin_idx = 2
        current_margin_rate = self.ccp.MARGIN_RATES[current_margin_idx]
        
        # Create simulation
        contagion = self.create_simulation(current_margin_rate)
        
        for shock_num in range(RL_CONFIG['SHOCKS_PER_EPISODE']):
            # Get state BEFORE shock
            state = self.ccp.get_state(contagion, shock_num, total_failures, current_margin_idx)
            
            # CCP decides margin level for next period
            decision, new_margin_rate, action_idx = self.ccp.get_action(state, explore=train)
            
            # If margin changed significantly, update the simulation
            if abs(new_margin_rate - current_margin_rate) > 0.01:
                # Update margin requirements
                for bank in self.bank_list:
                    if bank not in contagion.failed_banks:
                        old_margin = contagion.margin_states.get(bank, 0)
                        new_margin = self.bank_attrs[bank]['Total_Assets'] * new_margin_rate
                        delta = new_margin - old_margin
                        contagion.margin_states[bank] = new_margin
                        # Adjust HQLA accordingly
                        contagion.bank_states[bank]['HQLA'] -= delta
                
                current_margin_rate = new_margin_rate
                current_margin_idx = action_idx
            
            # Apply random stock shock
            stock = random.choice(self.stock_list)
            shock_pct = random.uniform(*RL_CONFIG['SHOCK_RANGE'])
            
            failures_before = len(contagion.failed_banks)
            
            # Run the simulation's propagation
            contagion.propagate_stock_devaluation(
                stock, shock_pct, 
                max_rounds=50, 
                failure_threshold=RL_CONFIG['FAILURE_THRESHOLD']
            )
            
            failures_after = len(contagion.failed_banks)
            new_failures = failures_after - failures_before
            total_failures += new_failures
            
            # Calculate reward
            survivors = self.num_banks - failures_after
            reward = survivors * RL_CONFIG['REWARD_PER_SURVIVOR'] / self.num_banks
            reward -= new_failures * RL_CONFIG['PENALTY_PER_FAILURE']
            
            # Penalty for very high margin (restricts economy)
            if current_margin_rate >= 0.08:
                reward -= RL_CONFIG['PENALTY_OVER_MARGIN']
            
            total_reward += reward
            
            # Get next state
            next_state = self.ccp.get_state(contagion, shock_num + 1, total_failures, current_margin_idx)
            done = (shock_num == RL_CONFIG['SHOCKS_PER_EPISODE'] - 1) or (failures_after >= self.num_banks * 0.8)
            
            # Store experience
            if train:
                self.ccp.remember(state, action_idx, reward, next_state, done)
            
            if done:
                break
        
        # Final bonus/penalty
        final_failures = len(contagion.failed_banks)
        survival_rate = (self.num_banks - final_failures) / self.num_banks
        
        if final_failures == 0:
            total_reward += RL_CONFIG['BONUS_ZERO_FAILURES']
        
        # Scale reward by survival rate
        total_reward += survival_rate * 100
        
        # Train
        if train:
            self.ccp.replay()
        
        return {
            'reward': total_reward,
            'failed_banks': final_failures,
            'survival_rate': survival_rate,
            'final_margin': current_margin_rate,
            'avg_health': np.mean([contagion.get_bank_health(b) for b in self.bank_list 
                                   if b not in contagion.failed_banks]) if final_failures < self.num_banks else 0
        }
    
    def train(self, num_episodes: int = None):
        """Main training loop"""
        if num_episodes is None:
            num_episodes = RL_CONFIG['NUM_EPISODES']
        
        print(f"\n🚀 Training CCP for {num_episodes} episodes...")
        print("=" * 70)
        
        for episode in range(num_episodes):
            result = self.run_episode(train=True)
            self.episode_metrics.append(result)
            
            print(f"Ep {episode+1:4d} | "
                  f"Reward: {result['reward']:+7.1f} | "
                  f"Failed: {result['failed_banks']:2d}/{self.num_banks} | "
                  f"Survival: {result['survival_rate']*100:5.1f}% | "
                  f"Margin: {result['final_margin']*100:.1f}% | "
                  f"ε: {self.ccp.epsilon:.3f}")
        
        print("=" * 70)
        self.ccp.save()
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary"""
        print("\n📊 TRAINING SUMMARY")
        print("-" * 50)
        
        print("\nCCP Margin Decisions:")
        total = sum(self.ccp.decisions.values())
        for m in MarginLevel:
            pct = self.ccp.decisions[m] / max(total, 1) * 100
            rate = self.ccp.MARGIN_RATES[m.value] * 100
            print(f"  {m.name:10s} ({rate:.0f}%): {self.ccp.decisions[m]:6d} ({pct:5.1f}%)")
        
        if self.episode_metrics:
            final = self.episode_metrics[-100:] if len(self.episode_metrics) >= 100 else self.episode_metrics
            print(f"\nFinal {len(final)} episodes:")
            print(f"  Avg Reward: {np.mean([m['reward'] for m in final]):+.1f}")
            print(f"  Avg Failures: {np.mean([m['failed_banks'] for m in final]):.1f}")
            print(f"  Avg Survival Rate: {np.mean([m['survival_rate'] for m in final])*100:.1f}%")
    
    def evaluate(self, num_episodes: int = 20):
        """Evaluate trained model"""
        print(f"\n🔍 Evaluating over {num_episodes} episodes...")
        
        results = []
        for _ in range(num_episodes):
            result = self.run_episode(train=False)
            results.append(result)
        
        print(f"  Avg Reward: {np.mean([r['reward'] for r in results]):+.1f}")
        print(f"  Avg Failures: {np.mean([r['failed_banks'] for r in results]):.1f}")
        print(f"  Avg Survival Rate: {np.mean([r['survival_rate'] for r in results])*100:.1f}%")
        print(f"  Avg Final Health: {np.mean([r['avg_health'] for r in results]):.1f}")
        
        return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("CCP MARGIN OPTIMIZATION")
    print("Using algorithm.py BankingNetworkContagion simulation")
    print("=" * 70)
    
    env = CCPTrainingEnvironment()
    
    # Always try to load existing model
    loaded = env.ccp.load()
    
    if loaded:
        print("\n📂 Continuing training from existing model...")
    else:
        print("\n🆕 Training new model from scratch...")
    
    env.train()
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    env.evaluate(20)
