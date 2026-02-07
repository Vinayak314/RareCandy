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
