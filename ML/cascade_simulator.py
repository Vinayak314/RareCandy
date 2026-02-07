"""
Shock Generator and Cascade Simulator for financial contagion modeling.

Simulates how defaults propagate through a network of interconnected banks.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

from bank import Bank, CCP
from loan import InterbankLoan


class ShockType(Enum):
    """Types of financial shocks."""
    IDIOSYNCRATIC = "idiosyncratic"  # Single bank shock
    SYSTEMATIC = "systematic"         # Correlated shock to multiple banks
    LIQUIDITY = "liquidity"           # Market-wide funding shock


@dataclass
class Shock:
    """Represents a financial shock event."""
    shock_type: ShockType
    target_bank_ids: List[str]
    magnitude: float  # Percentage of equity lost (0.0 to 1.0)
    description: str


@dataclass
class CascadeStep:
    """Records one step in a default cascade."""
    round_number: int
    defaulted_bank_id: str
    trigger_reason: str
    equity_before: float
    loss_amount: float


class ShockGenerator:
    """
    Generates various types of financial shocks for simulation.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed."""
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
    
    def idiosyncratic_shock(
        self, 
        bank_id: str, 
        magnitude: float = 0.5
    ) -> Shock:
        """
        Generate a shock to a single bank.
        
        Args:
            bank_id: Target bank
            magnitude: Fraction of equity lost (e.g., 0.5 = 50%)
            
        Returns:
            Shock object
        """
        return Shock(
            shock_type=ShockType.IDIOSYNCRATIC,
            target_bank_ids=[bank_id],
            magnitude=magnitude,
            description=f"Idiosyncratic shock: {bank_id} loses {magnitude*100:.0f}% equity"
        )
    
    def systematic_shock(
        self, 
        bank_ids: List[str], 
        magnitude: float = 0.3
    ) -> Shock:
        """
        Generate a correlated shock affecting multiple banks.
        
        Args:
            bank_ids: List of affected banks
            magnitude: Fraction of equity lost
            
        Returns:
            Shock object
        """
        return Shock(
            shock_type=ShockType.SYSTEMATIC,
            target_bank_ids=bank_ids,
            magnitude=magnitude,
            description=f"Systematic shock: {len(bank_ids)} banks lose {magnitude*100:.0f}% equity"
        )
    
    def random_shock(
        self, 
        banks: Dict[str, Bank],
        magnitude_range: Tuple[float, float] = (0.3, 0.7)
    ) -> Shock:
        """
        Generate a random shock to a random bank.
        
        Args:
            banks: Dictionary of bank_id -> Bank
            magnitude_range: (min, max) magnitude
            
        Returns:
            Shock object
        """
        bank_id = np.random.choice(list(banks.keys()))
        magnitude = np.random.uniform(*magnitude_range)
        return self.idiosyncratic_shock(bank_id, magnitude)


class CascadeSimulator:
    """
    Simulates default cascades through the financial network.
    
    When a bank defaults, its counterparties lose their exposures,
    potentially causing them to default as well.
    """
    
    def __init__(
        self, 
        banks: Dict[str, Bank], 
        loans: List[InterbankLoan],
        ccp: Optional[CCP] = None,
        recovery_rate: float = 0.4
    ):
        """
        Initialize the cascade simulator.
        
        Args:
            banks: Dictionary mapping bank_id -> Bank object
            loans: List of interbank loans
            ccp: Optional CCP (for centralized clearing)
            recovery_rate: Recovery rate on defaulted loans (default: 40%)
        """
        self.banks = {b.bank_id: b for b in banks.values()} if isinstance(list(banks.values())[0], Bank) else banks
        self.loans = loans
        self.ccp = ccp
        self.recovery_rate = recovery_rate
        
        # Build exposure maps
        self._build_exposure_maps()
        
        # Track simulation state
        self.defaulted_banks: Set[str] = set()
        self.cascade_history: List[CascadeStep] = []
    
    def _build_exposure_maps(self):
        """Build maps of who owes what to whom."""
        # exposure_to[bank_id] = list of (counterparty, amount) that owe TO this bank
        self.exposure_to: Dict[str, List[Tuple[str, float]]] = {}
        # exposure_from[bank_id] = list of (counterparty, amount) this bank owes
        self.exposure_from: Dict[str, List[Tuple[str, float]]] = {}
        
        for loan in self.loans:
            # Lender has exposure TO the borrower
            if loan.lender_id not in self.exposure_to:
                self.exposure_to[loan.lender_id] = []
            self.exposure_to[loan.lender_id].append((loan.borrower_id, loan.principal))
            
            # Borrower has exposure FROM the lender
            if loan.borrower_id not in self.exposure_from:
                self.exposure_from[loan.borrower_id] = []
            self.exposure_from[loan.borrower_id].append((loan.lender_id, loan.principal))
    
    def apply_shock(self, shock: Shock) -> List[str]:
        """
        Apply a shock to the network.
        
        Args:
            shock: Shock to apply
            
        Returns:
            List of bank IDs that defaulted from the initial shock
        """
        initial_defaults = []
        
        for bank_id in shock.target_bank_ids:
            if bank_id in self.banks and bank_id not in self.defaulted_banks:
                bank = self.banks[bank_id]
                loss = bank.equity * shock.magnitude
                
                # Record state before
                equity_before = bank.equity
                
                # Apply loss
                bank.update_balance_sheet(delta_assets=-loss)
                
                if not bank.is_solvent():
                    self.defaulted_banks.add(bank_id)
                    initial_defaults.append(bank_id)
                    self.cascade_history.append(CascadeStep(
                        round_number=0,
                        defaulted_bank_id=bank_id,
                        trigger_reason=shock.description,
                        equity_before=equity_before,
                        loss_amount=loss
                    ))
        
        return initial_defaults
    
    def propagate_defaults(self, max_rounds: int = 100) -> int:
        """
        Propagate defaults through the network until no new defaults occur.
        
        Args:
            max_rounds: Maximum propagation rounds (safety limit)
            
        Returns:
            Number of propagation rounds
        """
        round_num = 1
        
        while round_num <= max_rounds:
            new_defaults = []
            
            # For each defaulted bank, calculate losses to counterparties
            for defaulted_id in list(self.defaulted_banks):
                # Find all banks that lent TO the defaulted bank
                exposures = self.exposure_to.get(defaulted_id, [])
                
                for lender_id, exposure_amount in exposures:
                    if lender_id in self.defaulted_banks:
                        continue  # Already defaulted
                    
                    if lender_id not in self.banks:
                        continue
                    
                    lender = self.banks[lender_id]
                    
                    # Calculate loss (exposure minus recovery)
                    loss = exposure_amount * (1 - self.recovery_rate)
                    
                    # Check if already processed this exposure
                    # (simplified: we allow repeated hits for now)
                    equity_before = lender.equity
                    lender.update_balance_sheet(delta_assets=-loss)
                    
                    if not lender.is_solvent() and lender_id not in self.defaulted_banks:
                        self.defaulted_banks.add(lender_id)
                        new_defaults.append(lender_id)
                        self.cascade_history.append(CascadeStep(
                            round_number=round_num,
                            defaulted_bank_id=lender_id,
                            trigger_reason=f"Contagion from {defaulted_id}",
                            equity_before=equity_before,
                            loss_amount=loss
                        ))
            
            if not new_defaults:
                break
            
            round_num += 1
        
        return round_num - 1
    
    def run_simulation(self, shock: Shock) -> Dict:
        """
        Run a complete cascade simulation.
        
        Args:
            shock: Initial shock to apply
            
        Returns:
            Simulation results dictionary
        """
        # Reset state
        self.defaulted_banks = set()
        self.cascade_history = []
        
        # Apply initial shock
        initial_defaults = self.apply_shock(shock)
        
        # Propagate
        num_rounds = self.propagate_defaults()
        
        # Calculate metrics
        total_banks = len(self.banks)
        cascade_size = len(self.defaulted_banks)
        
        # Calculate total losses
        total_equity_lost = sum(step.loss_amount for step in self.cascade_history)
        
        return {
            'shock': shock.description,
            'initial_defaults': len(initial_defaults),
            'cascade_size': cascade_size,
            'cascade_depth': num_rounds,
            'survival_rate': (total_banks - cascade_size) / total_banks,
            'total_equity_lost': total_equity_lost,
            'defaulted_banks': list(self.defaulted_banks),
            'cascade_history': self.cascade_history
        }
    
    def get_systemic_fragility(self) -> float:
        """
        Calculate systemic fragility score (0-1).
        
        Higher = more fragile network.
        
        Returns:
            Fragility score
        """
        cascade_size = len(self.defaulted_banks)
        total_banks = len(self.banks)
        
        if total_banks == 0:
            return 0.0
        
        return cascade_size / total_banks
    
    def get_interbank_exposure(self, bank_id: str) -> Dict[str, float]:
        """
        Get detailed interbank exposure for a specific bank.
        
        C.6 Interbank Exposure: Total amount Bank B has borrowed from other banks.
        If Bank B is already heavily in debt to others, Bank A shouldn't add more.
        
        Args:
            bank_id: Bank identifier
            
        Returns:
            Dictionary with exposure details
        """
        if bank_id not in self.banks:
            return {}
        
        bank = self.banks[bank_id]
        
        # Exposure TO this bank (what others owe this bank)
        exposure_to = self.exposure_to.get(bank_id, [])
        total_lent = sum(amount for _, amount in exposure_to)
        
        # Exposure FROM this bank (what this bank owes others)
        exposure_from = self.exposure_from.get(bank_id, [])
        total_borrowed = sum(amount for _, amount in exposure_from)
        
        return {
            'bank_id': bank_id,
            'total_lent': total_lent,
            'total_borrowed': total_borrowed,
            'net_interbank_position': total_lent - total_borrowed,
            'interbank_borrowing_ratio': total_borrowed / bank.assets if bank.assets > 0 else float('inf'),
            'counterparties_lending_to': len(exposure_from),
            'counterparties_borrowing_from': len(exposure_to),
            'is_net_borrower': total_borrowed > total_lent
        }
    
    def get_all_interbank_exposures(self) -> Dict[str, Dict]:
        """
        Get interbank exposure details for all banks.
        
        Returns:
            Dictionary mapping bank_id -> exposure details
        """
        return {bank_id: self.get_interbank_exposure(bank_id) for bank_id in self.banks}
    
    def calculate_contagion_risk_score(self, bank_id: str) -> Dict[str, float]:
        """
        Calculate comprehensive contagion risk score for a bank.
        
        Combines all metrics from task.txt:
        A. Balance Sheet Fundamentals (Leverage, LCR, NPL)
        B. Market Signals (CDS Spread, Volatility)
        C. Network Position (Interbank Exposure, Asset Correlation)
        
        Args:
            bank_id: Bank identifier
            
        Returns:
            Dictionary with risk breakdown and total score
        """
        if bank_id not in self.banks:
            return {}
        
        bank = self.banks[bank_id]
        exposure = self.get_interbank_exposure(bank_id)
        
        # Get base risk metrics
        risk_metrics = bank.get_risk_metrics()
        counterparty_score = bank.calculate_counterparty_risk_score()
        
        # Calculate network centrality risk (based on borrowing/lending)
        total_banks = len(self.banks)
        connection_ratio = (exposure['counterparties_lending_to'] + 
                          exposure['counterparties_borrowing_from']) / (2 * total_banks) if total_banks > 0 else 0
        
        # Higher interbank borrowing ratio = higher contagion risk
        interbank_risk = min(exposure['interbank_borrowing_ratio'], 1.0)
        
        # Combined contagion risk
        contagion_score = (
            0.4 * counterparty_score +        # Base risk from task.txt metrics
            0.3 * interbank_risk +            # Interbank exposure
            0.3 * connection_ratio            # Network centrality
        )
        
        return {
            'bank_id': bank_id,
            'counterparty_risk_score': counterparty_score,
            'interbank_risk': interbank_risk,
            'network_centrality_risk': connection_ratio,
            'contagion_risk_score': contagion_score,
            'risk_metrics': risk_metrics,
            'interbank_exposure': exposure,
            'risk_level': 'HIGH' if contagion_score > 0.6 else ('MEDIUM' if contagion_score > 0.3 else 'LOW')
        }
    
    def identify_vulnerable_banks(self, threshold: float = 0.5) -> List[str]:
        """
        Identify banks vulnerable to contagion based on risk metrics.
        
        Args:
            threshold: Risk score threshold (default: 0.5)
            
        Returns:
            List of bank IDs with high contagion risk
        """
        vulnerable = []
        for bank_id in self.banks:
            risk = self.calculate_contagion_risk_score(bank_id)
            if risk.get('contagion_risk_score', 0) >= threshold:
                vulnerable.append(bank_id)
        return vulnerable
    
    def get_systemic_risk_report(self) -> Dict:
        """
        Generate comprehensive systemic risk report for the network.
        
        Uses all metrics from task.txt to assess network-wide risk.
        
        Returns:
            Dictionary with systemic risk analysis
        """
        # Calculate risk scores for all banks
        all_risks = {bank_id: self.calculate_contagion_risk_score(bank_id) 
                    for bank_id in self.banks}
        
        contagion_scores = [r['contagion_risk_score'] for r in all_risks.values()]
        counterparty_scores = [r['counterparty_risk_score'] for r in all_risks.values()]
        
        # Count banks by risk level
        high_risk = sum(1 for r in all_risks.values() if r['risk_level'] == 'HIGH')
        medium_risk = sum(1 for r in all_risks.values() if r['risk_level'] == 'MEDIUM')
        low_risk = sum(1 for r in all_risks.values() if r['risk_level'] == 'LOW')
        
        # Identify most vulnerable banks
        sorted_risks = sorted(all_risks.items(), 
                            key=lambda x: x[1]['contagion_risk_score'], 
                            reverse=True)
        top_5_vulnerable = [bank_id for bank_id, _ in sorted_risks[:5]]
        
        # Count banks with specific risk flags
        desperate_banks = sum(1 for bank in self.banks.values() if bank.is_lcr_desperate())
        high_leverage_banks = sum(1 for bank in self.banks.values() if bank.is_leverage_high_risk())
        cds_warning_banks = sum(1 for bank in self.banks.values() if bank.is_cds_warning())
        
        return {
            'network_summary': {
                'total_banks': len(self.banks),
                'high_risk_banks': high_risk,
                'medium_risk_banks': medium_risk,
                'low_risk_banks': low_risk,
                'desperate_banks_lcr': desperate_banks,
                'high_leverage_banks': high_leverage_banks,
                'cds_warning_banks': cds_warning_banks,
                'avg_contagion_risk': np.mean(contagion_scores) if contagion_scores else 0,
                'max_contagion_risk': max(contagion_scores) if contagion_scores else 0,
                'avg_counterparty_risk': np.mean(counterparty_scores) if counterparty_scores else 0,
                'systemic_fragility': self.get_systemic_fragility()
            },
            'top_5_vulnerable_banks': top_5_vulnerable,
            'bank_risk_details': all_risks
        }
