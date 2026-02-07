"""
CCP Risk Simulation with Contagion Modeling and Reinforcement Learning
========================================================================
Central Counterparty (CCP) risk assessment system that:
1. Simulates systemic risk contagion through interbank networks
2. Assesses stock transaction impact on bank stability  
3. Uses Deep Q-Learning to predict systemic losses
4. Provides 3-tier trade approval: APPROVED, REQUIRE_MARGIN, REJECTED
"""

import pandas as pd
import numpy as np
import random
import pickle
import os
from collections import deque
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from enum import Enum

# ==========================================
# CONFIGURATION
# ==========================================
TRAINING_EPISODES = 5000
BATCH_SIZE = 64
MEMORY_SIZE = 2000
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.05
SHOCK_RANGE = (0.10, 0.30)  # 10-30% asset shock
CRASH_SEVERITY = 3.5  # 3.5 sigma worst-case move
MODEL_PATH = "ccp_risk_model.pkl"

class TradeDecision(Enum):
    APPROVED = "APPROVED"
    REQUIRE_MARGIN = "REQUIRE_MARGIN"
    REJECTED = "REJECTED"

# ==========================================
# 1. THE MARKET ENVIRONMENT
# ==========================================
class SystemicRiskEnv:
    """
    Simulates the banking system with interbank connections and stock holdings.
    Supports shock propagation (contagion) and trade impact analysis.
    """
    
    def __init__(self, banks_file, stocks_file, matrix_file):
        # Load Bank Data
        self.banks_df = pd.read_csv(banks_file)
        self.interbank_matrix = pd.read_csv(matrix_file, index_col=0)
        
        # Calculate Stock Volatility from historical data
        stocks_df = pd.read_csv(stocks_file)
        stocks_df['Return'] = stocks_df.groupby('Ticker')['Close'].pct_change()
        self.stock_volatility = stocks_df.groupby('Ticker')['Return'].std().reset_index()
        self.stock_volatility.columns = ['Ticker', 'Daily_Volatility']
        median_vol = self.stock_volatility['Daily_Volatility'].median()
        self.stock_volatility['Daily_Volatility'].fillna(median_vol, inplace=True)
        
        # Create fast lookup dictionaries (values in Billions USD)
        self.assets_map = self.banks_df.set_index('Bank')['Total_Assets'].to_dict()
        self.equity_map = self.banks_df.set_index('Bank')['Equity'].to_dict()
        self.hqla_map = self.banks_df.set_index('Bank')['HQLA'].to_dict()
        self.liab_map = self.banks_df.set_index('Bank')['Total_Liabilities'].to_dict()
        self.outflows_map = self.banks_df.set_index('Bank')['Net_Outflows_30d'].to_dict()
        self.interbank_adj = self.interbank_matrix.to_dict(orient='index')
        self.bank_list = self.banks_df['Bank'].tolist()
        
        # Scale Bank Features for Neural Network
        self.scaler = StandardScaler()
        self.features = ['Total_Assets', 'Equity', 'HQLA', 'Net_Outflows_30d', 'Interbank_Liabilities']
        feature_data = self.banks_df[self.features].fillna(0)
        self.bank_features_scaled = self.scaler.fit_transform(feature_data)
        
        # Stock volatility lookup
        self.vol_lookup = self.stock_volatility.set_index('Ticker')['Daily_Volatility'].to_dict()
        
        print(f"✅ Loaded {len(self.bank_list)} banks, {len(self.vol_lookup)} stocks")

    def get_bank_state(self, bank_name):
        """Returns normalized feature vector for RL agent"""
        idx = self.banks_df[self.banks_df['Bank'] == bank_name].index[0]
        return self.bank_features_scaled[idx]

    def get_stock_volatility(self, ticker):
        """Returns daily volatility for a stock ticker"""
        return self.vol_lookup.get(ticker, 0.02)  # Default 2% if unknown

    def get_bank_health_ratio(self, bank_name):
        """Returns Assets / Liabilities ratio (>1 is solvent)"""
        return self.assets_map[bank_name] / self.liab_map[bank_name]

    def simulate_trade_impact(self, bank_name, ticker, trade_usd_billions, direction='BUY'):
        """
        Simulates the impact of a stock trade on a bank's balance sheet.
        
        BUY: Reduces HQLA, adds market exposure (risky)
        SELL: Reduces market exposure, adds cash (safer)
        
        Returns: (new_assets, new_equity, financial_loss)
        """
        current_assets = self.assets_map[bank_name]
        current_equity = self.equity_map[bank_name]
        current_hqla = self.hqla_map[bank_name]
        
        vol = self.get_stock_volatility(ticker)
        
        # Worst-case crash calculation
        crash_pct = min(1.0, CRASH_SEVERITY * vol)
        
        if direction == 'BUY':
            # Bank uses HQLA to buy stock
            if trade_usd_billions > current_hqla:
                trade_usd_billions = current_hqla  # Cap at available HQLA
            
            # In worst case, stock crashes
            financial_loss = trade_usd_billions * crash_pct
            new_assets = current_assets - financial_loss
            new_equity = current_equity - financial_loss
            
        else:  # SELL
            # Selling reduces exposure - generates cash, minimal loss
            # Only risk: fire sale discount (assume 5% haircut max)
            financial_loss = trade_usd_billions * 0.05  
            new_assets = current_assets - financial_loss
            new_equity = current_equity - financial_loss
        
        return new_assets, new_equity, financial_loss

    def propagate_contagion(self, trigger_bank, initial_loss, current_assets=None, current_equity=None):
        """
        Simulates domino effect of bank failures through interbank network.
        
        Returns: (set of failed banks, total CCP loss)
        """
        # Start with copies of current state
        if current_assets is None:
            current_assets = self.assets_map.copy()
        if current_equity is None:
            current_equity = self.equity_map.copy()
        
        # Apply initial shock
        current_assets[trigger_bank] -= initial_loss
        current_equity[trigger_bank] -= initial_loss
        
        failed_banks = set()
        cascade_queue = []
        
        # Check if trigger bank fails
        if current_assets[trigger_bank] < self.liab_map[trigger_bank]:
            failed_banks.add(trigger_bank)
            cascade_queue.append(trigger_bank)
        
        # Propagate through network
        while cascade_queue:
            failed = cascade_queue.pop(0)
            
            for lender in self.bank_list:
                if lender in failed_banks:
                    continue
                
                # Get exposure from interbank matrix
                exposure = self.interbank_adj.get(lender, {}).get(failed, 0.0)
                
                if exposure > 0:
                    # Lender loses the amount it lent to failed bank
                    current_assets[lender] -= exposure
                    current_equity[lender] -= exposure
                    
                    # Check for secondary failure
                    if current_assets[lender] < self.liab_map[lender]:
                        failed_banks.add(lender)
                        cascade_queue.append(lender)
        
        # Calculate total CCP loss (uncovered liabilities)
        ccp_loss = 0.0
        for bank in failed_banks:
            uncovered = max(0, self.liab_map[bank] - current_assets[bank])
            ccp_loss += uncovered
        
        return failed_banks, ccp_loss

    def run_scenario(self, trigger_bank=None, shock_pct=None, trade_params=None):
        """
        Runs a complete scenario:
        1. Optional stock trade
        2. Random shock to trigger bank
        3. Contagion propagation
        4. Calculate total loss
        
        Returns: (failed_banks, ccp_loss, trade_loss)
        """
        if trigger_bank is None:
            trigger_bank = random.choice(self.bank_list)
        
        if shock_pct is None:
            shock_pct = random.uniform(*SHOCK_RANGE)
        
        current_assets = self.assets_map.copy()
        current_equity = self.equity_map.copy()
        trade_loss = 0.0
        
        # Apply trade impact first
        if trade_params:
            bank, ticker, amount, direction = trade_params
            new_assets, new_equity, trade_loss = self.simulate_trade_impact(
                bank, ticker, amount, direction
            )
            current_assets[bank] = new_assets
            current_equity[bank] = new_equity
        
        # Calculate shock loss
        shock_loss = current_assets[trigger_bank] * shock_pct
        
        # Run contagion
        failed_banks, ccp_loss = self.propagate_contagion(
            trigger_bank, shock_loss, current_assets, current_equity
        )
        
        return failed_banks, ccp_loss, trade_loss


# ==========================================
# 2. THE RL AGENT (Deep Q-Learning)
# ==========================================
class RiskAgent:
    """
    Deep Q-Learning agent that learns to predict systemic risk.
    Features online learning for continuous adaptation.
    """
    
    def __init__(self, state_size, load_pretrained=True):
        self.state_size = state_size
        # Input: bank_state (5) + volatility (1) + trade_amount (1) + direction (1) = 8
        self.input_size = state_size + 3
        
        # Neural Network (Q-function approximator)
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1,
            warm_start=True,  # Enables incremental learning
            learning_rate_init=0.001,
            random_state=42
        )
        
        # Initialize with dummy data
        self._initialize_model()
        
        # Experience replay buffer
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0  # Exploration rate
        
        # Try to load pre-trained weights
        if load_pretrained and os.path.exists(MODEL_PATH):
            self.load_model()
            print(f"✅ Loaded pre-trained model from {MODEL_PATH}")
    
    def _initialize_model(self):
        """Initialize with dummy data to set up the model"""
        dummy_X = np.zeros((1, self.input_size))
        dummy_y = np.zeros((1,))
        self.model.fit(dummy_X, dummy_y)
    
    def remember(self, state, vol, amount, direction, reward):
        """Store experience in replay buffer"""
        dir_encoded = 1.0 if direction == 'BUY' else 0.0
        self.memory.append((state, vol, amount, dir_encoded, reward))
    
    def replay(self, batch_size=BATCH_SIZE):
        """Train on random batch from memory (experience replay)"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        X, y = [], []
        
        for state, vol, amt, direction, reward in batch:
            input_vec = np.hstack((state, [vol, amt, direction]))
            X.append(input_vec)
            y.append(reward)
        
        self.model.fit(np.array(X), np.array(y))
        
        # Decay exploration rate
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
    
    def predict_risk(self, state, vol, amount, direction='BUY'):
        """Predict expected systemic loss for a trade"""
        dir_encoded = 1.0 if direction == 'BUY' else 0.0
        input_vec = np.hstack((state, [vol, amount, dir_encoded])).reshape(1, -1)
        return self.model.predict(input_vec)[0]
    
    def save_model(self):
        """Save model to disk"""
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'epsilon': self.epsilon,
                'memory': list(self.memory)[-500:]  # Save last 500 experiences
            }, f)
        print(f"✅ Model saved to {MODEL_PATH}")
    
    def load_model(self):
        """Load model from disk"""
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.epsilon = max(data.get('epsilon', 0.1), MIN_EPSILON)
            saved_memory = data.get('memory', [])
            self.memory.extend(saved_memory)


# ==========================================
# 3. CCP RISK GATEWAY (Trade Approval)
# ==========================================
class CCPRiskGateway:
    """
    Central Counterparty trade approval system.
    Provides 3-tier response: APPROVED, REQUIRE_MARGIN, REJECTED
    Supports online learning from production trades.
    """
    
    # Thresholds (in billions USD)
    APPROVE_THRESHOLD = 0.5   # Below 0.5B predicted loss -> Approve
    MARGIN_THRESHOLD = 5.0    # 0.5B to 5B -> Require additional margin
    # Above 5B -> Reject
    
    def __init__(self, env, agent, online_learning=True):
        self.env = env
        self.agent = agent
        self.online_learning = online_learning
        self.trade_history = []
    
    def check_trade(self, bank_name, ticker, trade_usd_billions, direction='BUY'):
        """
        Main API for CCP risk check.
        
        Returns: (decision, predicted_loss, explanation)
        """
        # Get inputs for RL model
        state = self.env.get_bank_state(bank_name)
        vol = self.env.get_stock_volatility(ticker)
        hqla = self.env.hqla_map[bank_name]
        
        # Normalize trade amount relative to HQLA
        trade_ratio = trade_usd_billions / hqla if hqla > 0 else 1.0
        
        # Get risk prediction
        predicted_loss = self.agent.predict_risk(state, vol, trade_ratio, direction)
        
        # Determine decision
        if predicted_loss <= self.APPROVE_THRESHOLD:
            decision = TradeDecision.APPROVED
            explanation = f"Low systemic risk. Predicted loss: ${predicted_loss:.2f}B"
        elif predicted_loss <= self.MARGIN_THRESHOLD:
            decision = TradeDecision.REQUIRE_MARGIN
            margin_pct = min(50, 10 + (predicted_loss - 0.5) * 5)  # 10-50%
            explanation = f"Moderate risk. Require {margin_pct:.0f}% margin. Predicted loss: ${predicted_loss:.2f}B"
        else:
            decision = TradeDecision.REJECTED
            explanation = f"HIGH SYSTEMIC RISK. Predicted loss: ${predicted_loss:.2f}B exceeds threshold."
        
        # Log for online learning
        self.trade_history.append({
            'bank': bank_name,
            'ticker': ticker,
            'amount': trade_usd_billions,
            'direction': direction,
            'predicted_loss': predicted_loss,
            'decision': decision.value
        })
        
        return decision, predicted_loss, explanation
    
    def record_actual_outcome(self, trade_idx, actual_loss):
        """
        Record actual outcome for online learning.
        Call this after a trade settles to improve the model.
        """
        if not self.online_learning:
            return
        
        trade = self.trade_history[trade_idx]
        state = self.env.get_bank_state(trade['bank'])
        vol = self.env.get_stock_volatility(trade['ticker'])
        hqla = self.env.hqla_map[trade['bank']]
        trade_ratio = trade['amount'] / hqla if hqla > 0 else 1.0
        
        # Add to agent memory and train
        self.agent.remember(state, vol, trade_ratio, trade['direction'], actual_loss)
        self.agent.replay(batch_size=min(32, len(self.agent.memory)))
    
    def print_trade_result(self, bank_name, ticker, trade_usd, direction):
        """Helper to print formatted trade check result"""
        decision, loss, explanation = self.check_trade(bank_name, ticker, trade_usd, direction)
        
        symbol = {"APPROVED": "✅", "REQUIRE_MARGIN": "⚠️", "REJECTED": "❌"}[decision.value]
        
        print(f"\n{'='*60}")
        print(f"TRADE: {bank_name} {direction} ${trade_usd}B of {ticker}")
        print(f"{'='*60}")
        print(f"Volatility: {self.env.get_stock_volatility(ticker):.2%}")
        print(f"Bank HQLA: ${self.env.hqla_map[bank_name]:.2f}B")
        print(f"Predicted Systemic Loss: ${loss:.2f}B")
        print(f"{symbol} DECISION: {decision.value}")
        print(f"   {explanation}")
        print(f"{'='*60}")
        
        return decision


# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
def train_agent(env, agent, episodes=TRAINING_EPISODES):
    """
    Train the RL agent through simulated scenarios.
    Each episode simulates a random shock with optional trade.
    """
    print(f"\n🚀 Training Risk AI on {episodes} scenarios...")
    print("-" * 50)
    
    for episode in range(episodes):
        # Random trigger bank and scenario
        bank = random.choice(env.bank_list)
        state = env.get_bank_state(bank)
        
        # Random trade parameters
        ticker = random.choice(list(env.vol_lookup.keys()))
        vol = env.get_stock_volatility(ticker)
        
        # Trade amount as fraction of HQLA (0.1 to 0.5)
        trade_ratio = random.uniform(0.1, 0.5)
        direction = random.choice(['BUY', 'SELL'])
        
        # Calculate actual trade in billions
        hqla = env.hqla_map[bank]
        trade_amount = hqla * trade_ratio
        
        trade_params = (bank, ticker, trade_amount, direction)
        
        # Run scenario
        failed_banks, ccp_loss, trade_loss = env.run_scenario(
            trigger_bank=bank,
            trade_params=trade_params
        )
        
        # Total loss is what CCP would bear
        total_loss = ccp_loss + trade_loss
        
        # Store experience
        agent.remember(state, vol, trade_ratio, direction, total_loss)
        
        # Train on batch
        agent.replay()
        
        # Progress logging
        if episode > 0 and episode % 500 == 0:
            print(f"Episode {episode:5d} | Loss: ${total_loss:8.2f}B | "
                  f"Epsilon: {agent.epsilon:.3f} | Failed: {len(failed_banks)}")
    
    # Save trained model
    agent.save_model()
    print(f"\n✅ Training complete! Model saved to {MODEL_PATH}")


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize environment
    print("=" * 60)
    print("CCP SYSTEMIC RISK SIMULATION")
    print("=" * 60)
    
    env = SystemicRiskEnv(
        'phase1_engineered_data.csv',
        'stocks_data_long.csv', 
        'us_banks_interbank_matrix.csv'
    )
    
    # Initialize agent (will load pre-trained if exists)
    agent = RiskAgent(state_size=5, load_pretrained=True)
    
    # Check if we need training
    if not os.path.exists(MODEL_PATH):
        print("\n⚠️ No pre-trained model found. Starting training...")
        train_agent(env, agent, episodes=TRAINING_EPISODES)
    else:
        print(f"\n✅ Using pre-trained model. Epsilon: {agent.epsilon:.3f}")
        # Optional: continue training to refine
        # train_agent(env, agent, episodes=1000)
    
    # Initialize CCP Gateway
    ccp = CCPRiskGateway(env, agent, online_learning=True)
    
    # ==========================================
    # DEMO: Sample Trade Checks
    # ==========================================
    print("\n" + "="*60)
    print("SAMPLE CCP RISK ASSESSMENTS")
    print("="*60)
    
    # Low risk trade
    ccp.print_trade_result('JPM', 'AAPL', 5.0, 'BUY')
    
    # Medium risk trade
    ccp.print_trade_result('GS', 'MRVL', 50.0, 'BUY')
    
    # High risk trade (high volatility stock, large amount)
    ccp.print_trade_result('ZION', 'AAOI', 10.0, 'BUY')
    
    # Sell is usually safer
    ccp.print_trade_result('BAC', 'NVDA', 100.0, 'SELL')
    
    # Stress test: Small bank, big trade
    ccp.print_trade_result('IBOC', 'SMCI', 5.0, 'BUY')
    
    print("\n✅ CCP Risk Gateway ready for production use.")
    print("   Import CCPRiskGateway and call check_trade() for live assessment.")
