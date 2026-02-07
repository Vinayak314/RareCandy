"""
Bank and CCP (Central Counterparty) classes for financial network simulation.
"""

from typing import Dict, Optional
import numpy as np


class Bank:
    """
    Represents a financial institution in the network.
    
    Attributes:
        bank_id (str): Unique identifier for the bank
        equity (float): Current equity capital
        assets (float): Total assets
        liabilities (float): Total liabilities
        liquidity (float): Available liquid assets
        risk_weight (float): Systemic importance score (0-1)
    """
    
    def __init__(
        self,
        bank_id: str,
        equity: float,
        assets: float,
        liabilities: float,
        liquidity: float,
        risk_weight: float = 0.5
    ):
        """
        Initialize a Bank instance.
        
        Args:
            bank_id: Unique identifier
            equity: Initial equity capital
            assets: Total assets
            liabilities: Total liabilities
            liquidity: Available liquid assets
            risk_weight: Systemic importance (default: 0.5)
        
        Raises:
            ValueError: If financial values are negative or inconsistent
        """
        if equity < 0 or assets < 0 or liabilities < 0 or liquidity < 0:
            raise ValueError("Financial values must be non-negative")
        
        if not 0 <= risk_weight <= 1:
            raise ValueError("Risk weight must be between 0 and 1")
        
        self.bank_id = bank_id
        self.equity = equity
        self.assets = assets
        self.liabilities = liabilities
        self.liquidity = liquidity
        self.risk_weight = risk_weight
        self._is_defaulted = False
    
    def update_balance_sheet(
        self,
        delta_assets: float = 0.0,
        delta_liabilities: float = 0.0,
        delta_liquidity: float = 0.0
    ) -> None:
        """
        Update the bank's balance sheet.
        
        Args:
            delta_assets: Change in total assets
            delta_liabilities: Change in total liabilities
            delta_liquidity: Change in liquid assets
        """
        self.assets += delta_assets
        self.liabilities += delta_liabilities
        self.liquidity += delta_liquidity
        
        # Update equity based on accounting identity: Equity = Assets - Liabilities
        self.equity = self.assets - self.liabilities
        
        # Check for default
        if self.equity <= 0:
            self._is_defaulted = True
    
    def calculate_leverage(self) -> float:
        """
        Calculate the bank's leverage ratio.
        
        Returns:
            Leverage ratio (Assets / Equity)
        
        Raises:
            ZeroDivisionError: If equity is zero
        """
        if self.equity == 0:
            return float('inf')
        return self.assets / self.equity
    
    def calculate_liquidity_ratio(self) -> float:
        """
        Calculate the bank's liquidity ratio.
        
        Returns:
            Liquidity ratio (Liquidity / Liabilities)
        """
        if self.liabilities == 0:
            return float('inf')
        return self.liquidity / self.liabilities
    
    def is_solvent(self) -> bool:
        """
        Check if the bank is solvent (equity > 0).
        
        Returns:
            True if solvent, False otherwise
        """
        return self.equity > 0 and not self._is_defaulted
    
    def mark_default(self) -> None:
        """Mark the bank as defaulted."""
        self._is_defaulted = True
    
    def is_defaulted(self) -> bool:
        """Check if the bank has defaulted."""
        return self._is_defaulted
    
    def get_balance_sheet(self) -> Dict[str, float]:
        """
        Get a snapshot of the bank's balance sheet.
        
        Returns:
            Dictionary containing all balance sheet items
        """
        return {
            'bank_id': self.bank_id,
            'equity': self.equity,
            'assets': self.assets,
            'liabilities': self.liabilities,
            'liquidity': self.liquidity,
            'leverage': self.calculate_leverage() if self.equity > 0 else float('inf'),
            'liquidity_ratio': self.calculate_liquidity_ratio(),
            'is_solvent': self.is_solvent(),
            'is_defaulted': self._is_defaulted,
            'risk_weight': self.risk_weight
        }
    
    def __repr__(self) -> str:
        return (f"Bank(id={self.bank_id}, equity={self.equity:.2f}, "
                f"assets={self.assets:.2f}, leverage={self.calculate_leverage():.2f})")


class CCP(Bank):
    """
    Central Counterparty (CCP) - a specialized bank that provides clearing services.
    
    Extends Bank with margin requirements, default fund, and netting capabilities.
    
    Additional Attributes:
        margin_requirements (Dict): Margin rules and rates
        default_fund (float): Mutualized loss pool
        netting_positions (Dict): Net positions per member bank
    """
    
    def __init__(
        self,
        ccp_id: str,
        equity: float,
        assets: float = 0.0,
        liabilities: float = 0.0,
        liquidity: float = 0.0,
        initial_margin_rate: float = 0.05,
        default_fund_size: float = 0.0
    ):
        """
        Initialize a CCP instance.
        
        Args:
            ccp_id: Unique identifier for the CCP
            equity: Initial equity capital
            assets: Total assets (default: 0)
            liabilities: Total liabilities (default: 0)
            liquidity: Available liquid assets (default: 0)
            initial_margin_rate: Initial margin as % of position (default: 5%)
            default_fund_size: Size of mutualized default fund
        """
        super().__init__(
            bank_id=ccp_id,
            equity=equity,
            assets=assets,
            liabilities=liabilities,
            liquidity=liquidity,
            risk_weight=1.0  # CCPs are systemically critical
        )
        
        self.margin_requirements = {
            'initial_margin_rate': initial_margin_rate,
            'variation_margin_rate': 0.01,  # 1% for mark-to-market
            'margin_model': 'fixed_percentage'  # Can be 'VaR', 'SIMM', etc.
        }
        
        self.default_fund = default_fund_size
        self.netting_positions: Dict[str, float] = {}  # bank_id -> net_position
        self._total_margin_held = 0.0
    
    def calculate_initial_margin(self, position_value: float, volatility: float = 0.02) -> float:
        """
        Calculate initial margin requirement for a position.
        
        Args:
            position_value: Notional value of the position
            volatility: Estimated volatility (default: 2%)
        
        Returns:
            Required initial margin amount
        """
        if self.margin_requirements['margin_model'] == 'fixed_percentage':
            return abs(position_value) * self.margin_requirements['initial_margin_rate']
        elif self.margin_requirements['margin_model'] == 'VaR':
            # Simplified VaR: position * volatility * confidence_multiplier
            confidence_multiplier = 2.33  # 99% confidence (z-score)
            return abs(position_value) * volatility * confidence_multiplier
        else:
            return abs(position_value) * self.margin_requirements['initial_margin_rate']
    
    def calculate_variation_margin(self, position_value: float, price_change: float) -> float:
        """
        Calculate variation margin (mark-to-market adjustment).
        
        Args:
            position_value: Current position value
            price_change: Percentage price change
        
        Returns:
            Variation margin amount (can be negative)
        """
        return position_value * price_change
    
    def calculate_default_fund_contribution(self, bank: Bank) -> float:
        """
        Calculate a bank's required contribution to the default fund.
        
        Based on systemic importance and position size.
        
        Args:
            bank: The member bank
        
        Returns:
            Required default fund contribution
        """
        # Contribution proportional to risk weight and leverage
        base_contribution = self.default_fund * 0.01  # 1% minimum
        risk_adjusted = base_contribution * (1 + bank.risk_weight)
        
        return risk_adjusted
    
    def add_netting_position(self, bank_id: str, net_position: float) -> None:
        """
        Record a bank's net position after multilateral netting.
        
        Args:
            bank_id: Bank identifier
            net_position: Net position value (positive = owed to bank, negative = owes CCP)
        """
        self.netting_positions[bank_id] = net_position
    
    def get_total_exposure(self) -> float:
        """
        Calculate total CCP exposure across all members.
        
        Returns:
            Sum of absolute net positions
        """
        return sum(abs(pos) for pos in self.netting_positions.values())
    
    def get_margin_coverage_ratio(self) -> float:
        """
        Calculate margin coverage ratio.
        
        Returns:
            Ratio of margin held to total exposure
        """
        total_exposure = self.get_total_exposure()
        if total_exposure == 0:
            return float('inf')
        return self._total_margin_held / total_exposure
    
    def apply_default_waterfall(self, loss_amount: float) -> Dict[str, float]:
        """
        Apply CCP default waterfall to absorb losses.
        
        Waterfall order:
        1. Defaulter's margin
        2. CCP's default fund
        3. CCP's equity
        4. Loss mutualization to surviving members
        
        Args:
            loss_amount: Total loss to absorb
        
        Returns:
            Dictionary showing how loss was allocated
        """
        allocation = {
            'defaulter_margin': 0.0,
            'default_fund': 0.0,
            'ccp_equity': 0.0,
            'mutualized_loss': 0.0,
            'remaining_loss': loss_amount
        }
        
        # Step 1: Use default fund
        if self.default_fund >= allocation['remaining_loss']:
            allocation['default_fund'] = allocation['remaining_loss']
            self.default_fund -= allocation['remaining_loss']
            allocation['remaining_loss'] = 0.0
        else:
            allocation['default_fund'] = self.default_fund
            allocation['remaining_loss'] -= self.default_fund
            self.default_fund = 0.0
        
        # Step 2: Use CCP equity
        if allocation['remaining_loss'] > 0:
            if self.equity >= allocation['remaining_loss']:
                allocation['ccp_equity'] = allocation['remaining_loss']
                self.equity -= allocation['remaining_loss']
                allocation['remaining_loss'] = 0.0
            else:
                allocation['ccp_equity'] = self.equity
                allocation['remaining_loss'] -= self.equity
                self.equity = 0.0
                self._is_defaulted = True
        
        # Step 3: Mutualize remaining loss
        if allocation['remaining_loss'] > 0:
            allocation['mutualized_loss'] = allocation['remaining_loss']
        
        return allocation
    
    def get_ccp_status(self) -> Dict[str, any]:
        """
        Get comprehensive CCP status report.
        
        Returns:
            Dictionary with CCP metrics
        """
        status = self.get_balance_sheet()
        status.update({
            'default_fund': self.default_fund,
            'total_exposure': self.get_total_exposure(),
            'margin_coverage_ratio': self.get_margin_coverage_ratio(),
            'num_members': len(self.netting_positions),
            'margin_requirements': self.margin_requirements
        })
        return status
    
    def __repr__(self) -> str:
        return (f"CCP(id={self.bank_id}, equity={self.equity:.2f}, "
                f"default_fund={self.default_fund:.2f}, members={len(self.netting_positions)})")
