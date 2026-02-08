"""
CCP (Central Counterparty) ML Model Integration.
Loads trained PyTorch model (ccp_policy.pt) for margin requirement predictions.
"""

import os
import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not available - ML margin predictions disabled")


# Model architecture matching the trained model
STATE_SIZE = 15
ACTION_SIZE = 5
MARGIN_LEVELS = [0.05, 0.10, 0.20, 0.35, None]  # None = reject

# Default model path relative to this file
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ccp_policy.pt')


if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """Q-Network architecture matching the trained model."""
        
        def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
            super().__init__()
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


class CCPMarginPredictor:
    """
    Wrapper for the trained CCP margin model.
    Uses the ML model to predict optimal margin requirements for banks.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the CCP margin predictor.
        
        Args:
            model_path: Path to the trained model file (ccp_policy.pt)
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        self.model_loaded = False
        
        if TORCH_AVAILABLE:
            self._load_model()
        else:
            print("⚠️  PyTorch not available - using fallback margin calculation")
    
    def _load_model(self):
        """Load the trained model from disk."""
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model file not found at {self.model_path}")
            print("   Using fallback margin calculation")
            return
        
        try:
            self.model = QNetwork(STATE_SIZE, ACTION_SIZE)
            checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.model_loaded = True
            print(f"✅ CCP model loaded from {self.model_path}")
            print(f"   Running on: {DEVICE}")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            print("   Using fallback margin calculation")
            self.model = None
    
    def extract_state_features(
        self,
        bank_states: dict,
        bank_name: str,
        failed_banks: set,
        margin_states: dict = None
    ) -> np.ndarray:
        """
        Extract state features for a bank to feed to the model.
        
        Args:
            bank_states: Current bank states {bank_name: {attrs}}
            bank_name: Name of the bank to extract features for
            failed_banks: Set of failed bank names
            margin_states: Current margin states {bank_name: margin} (optional)
        
        Returns:
            Feature vector of shape (15,)
        """
        all_banks = list(bank_states.keys())
        num_banks = len(all_banks)
        num_failed = len(failed_banks)
        
        # Active banks
        active_banks = [b for b in all_banks if b not in failed_banks]
        
        # System features
        if active_banks:
            # Calculate average health (simplified)
            total_equity = sum(bank_states[b].get('Equity', 0) for b in active_banks)
            total_assets = sum(bank_states[b].get('Total_Assets', 1) for b in active_banks)
            avg_equity_ratio = total_equity / max(total_assets, 1)
            avg_health = min(100, avg_equity_ratio * 1000)  # Rough health proxy
            
            if margin_states:
                total_margin = sum(margin_states.get(b, 0) for b in active_banks)
            else:
                total_margin = 0
        else:
            avg_health = 0
            total_assets = 0
            total_margin = 0
        
        # Bank-specific features
        bank_state = bank_states.get(bank_name, {})
        bank_assets = bank_state.get('Total_Assets', 0)
        bank_equity = bank_state.get('Equity', 0)
        bank_hqla = bank_state.get('HQLA', 0)
        bank_margin = margin_states.get(bank_name, 0) if margin_states else 0
        
        equity_ratio = bank_equity / max(bank_assets, 1)
        liquidity_ratio = bank_hqla / max(bank_assets, 1)
        margin_ratio = bank_margin / max(bank_assets * 0.1, 1)
        
        # Bank health (simplified)
        bank_health = min(100, equity_ratio * 1000)
        
        # Stress indicator
        stress = 1.0 - (avg_health / 100.0) if avg_health > 0 else 1.0
        
        # Create feature vector (matching training format)
        return np.array([
            # System (5)
            avg_health / 100.0,
            num_failed / max(num_banks, 1),
            total_assets / 10000.0,
            total_margin / max(total_assets * 0.05, 1),
            stress,
            # Bank (6)
            bank_health / 100.0,
            bank_assets / 1000.0,
            equity_ratio,
            liquidity_ratio,
            margin_ratio,
            float(num_failed > 0),
            # Trade (4) - for initial margin, use neutral values
            0.0,  # trade_ratio (no specific trade)
            0.0,  # trade_direction (neutral)
            0.0,  # trade_size
            float(bank_name in failed_banks),
        ], dtype=np.float32)
    
    def predict_margin(
        self,
        bank_states: dict,
        bank_name: str,
        failed_banks: set = None,
        margin_states: dict = None
    ) -> tuple:
        """
        Predict optimal margin requirement for a bank.
        
        Args:
            bank_states: Current bank states
            bank_name: Bank to predict margin for
            failed_banks: Set of failed banks (default: empty)
            margin_states: Current margin states (optional)
        
        Returns:
            (margin_ratio, decision_index) - margin ratio (e.g., 0.05) and decision index (0-4)
        """
        if failed_banks is None:
            failed_banks = set()
        
        if not self.model_loaded or self.model is None:
            # Fallback: use risk-based heuristic
            return self._fallback_margin(bank_states, bank_name)
        
        # Extract features
        state = self.extract_state_features(bank_states, bank_name, failed_banks, margin_states)
        
        # Get Q-values from model
        q_values = self.model.predict(state.reshape(1, -1))[0]
        
        # Select best action (highest Q-value)
        action_idx = int(np.argmax(q_values))
        margin = MARGIN_LEVELS[action_idx]
        
        # If model suggests rejection, use fallback
        if margin is None:
            margin = 0.10  # Default to medium margin
        
        return margin, action_idx
    
    def _fallback_margin(self, bank_states: dict, bank_name: str) -> tuple:
        """
        Fallback margin calculation when model is not available.
        Uses risk-based heuristics.
        """
        bank = bank_states.get(bank_name, {})
        
        # Risk factors
        assets = bank.get('Total_Assets', 100)
        equity = bank.get('Equity', 10)
        cds_spread = bank.get('Est_CDS_Spread', 100)
        volatility = bank.get('Stock_Volatility', 0.02)
        
        equity_ratio = equity / max(assets, 1)
        
        # Higher margin for riskier banks
        if equity_ratio < 0.05 or cds_spread > 300 or volatility > 0.06:
            return 0.35, 3  # VERY_HIGH
        elif equity_ratio < 0.08 or cds_spread > 200 or volatility > 0.04:
            return 0.20, 2  # HIGH
        elif equity_ratio < 0.10 or cds_spread > 100:
            return 0.10, 1  # MEDIUM
        else:
            return 0.05, 0  # LOW
    
    def generate_margin_requirements(
        self,
        bank_attributes: dict,
        failed_banks: set = None
    ) -> dict:
        """
        Generate margin requirements for all banks using the ML model.
        
        Args:
            bank_attributes: Dict of {bank_name: {attributes}}
            failed_banks: Set of already failed banks (optional)
        
        Returns:
            Dict of {bank_name: margin_amount_in_billions}
        """
        if failed_banks is None:
            failed_banks = set()
        
        margin_requirements = {}
        
        for bank_name, attrs in bank_attributes.items():
            if bank_name in failed_banks:
                margin_requirements[bank_name] = 0
                continue
            
            # Get margin ratio from model
            margin_ratio, _ = self.predict_margin(bank_attributes, bank_name, failed_banks)
            
            # Calculate margin amount
            total_assets = attrs.get('Total_Assets', 0)
            margin_amount = total_assets * margin_ratio
            margin_requirements[bank_name] = margin_amount
        
        return margin_requirements
    
    def get_margin_decision_details(
        self,
        bank_states: dict,
        bank_name: str,
        failed_banks: set = None,
        margin_states: dict = None
    ) -> dict:
        """
        Get detailed margin decision for a specific bank.
        
        Returns full decision details including Q-values for each action.
        """
        if failed_banks is None:
            failed_banks = set()
        
        decision_labels = ['LOW (5%)', 'MEDIUM (10%)', 'HIGH (20%)', 'VERY_HIGH (35%)', 'REJECT']
        
        if not self.model_loaded or self.model is None:
            margin, action_idx = self._fallback_margin(bank_states, bank_name)
            return {
                'bank': bank_name,
                'margin_ratio': margin,
                'decision': decision_labels[action_idx],
                'margin_amount_B': bank_states.get(bank_name, {}).get('Total_Assets', 0) * margin,
                'model_used': False,
                'fallback_reason': 'Model not loaded'
            }
        
        state = self.extract_state_features(bank_states, bank_name, failed_banks, margin_states)
        q_values = self.model.predict(state.reshape(1, -1))[0]
        action_idx = int(np.argmax(q_values))
        margin = MARGIN_LEVELS[action_idx]
        
        if margin is None:
            margin = 0.10
            action_idx = 1
        
        return {
            'bank': bank_name,
            'margin_ratio': margin,
            'decision': decision_labels[action_idx],
            'margin_amount_B': bank_states.get(bank_name, {}).get('Total_Assets', 0) * margin,
            'q_values': {decision_labels[i]: float(q_values[i]) for i in range(len(q_values))},
            'model_used': True,
            'device': str(DEVICE)
        }


# Singleton instance for use across the app
_predictor_instance = None

def get_margin_predictor(model_path: str = None) -> CCPMarginPredictor:
    """Get or create the singleton CCPMarginPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CCPMarginPredictor(model_path)
    return _predictor_instance
