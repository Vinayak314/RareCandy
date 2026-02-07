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
        hqla (float): High Quality Liquid Assets
        net_cash_outflows_30d (float): Net cash outflows over 30 days
        total_loans (float): Total loans on the books
        bad_loans (float): Non-performing loans (defaulted or 90+ days overdue)
        cds_spread (float): Credit Default Swap spread in basis points
        stock_volatility (float): Stock price volatility (std dev over 10 days)
        asset_portfolio (np.ndarray): Normalized asset allocation vector for correlation
        interbank_borrowing (float): Total amount borrowed from other banks
    """
    
    def __init__(
        self,
        bank_id: str,
        equity: float,
        assets: float,
        liabilities: float,
        liquidity: float,
        risk_weight: float = 0.5,
        hqla: float = None,
        net_cash_outflows_30d: float = None,
        total_loans: float = None,
        bad_loans: float = 0.0,
        cds_spread: float = 50.0,
        stock_volatility: float = 0.02,
        asset_portfolio: np.ndarray = None,
        interbank_borrowing: float = 0.0
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
            hqla: High Quality Liquid Assets (defaults to liquidity)
            net_cash_outflows_30d: 30-day net cash outflows (defaults to 10% of liabilities)
            total_loans: Total loans on books (defaults to 60% of assets)
            bad_loans: Non-performing loans (default: 0)
            cds_spread: CDS spread in basis points (default: 50bps)
            stock_volatility: Stock volatility (default: 2%)
            asset_portfolio: Normalized asset allocation vector (default: random)
            interbank_borrowing: Total interbank borrowing (default: 0)
        
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
        
        # New attributes for risk metrics (task.txt)
        # A.2 LCR components
        self.hqla = hqla if hqla is not None else liquidity
        self.net_cash_outflows_30d = net_cash_outflows_30d if net_cash_outflows_30d is not None else liabilities * 0.1
        
        # A.3 NPL components
        self.total_loans = total_loans if total_loans is not None else assets * 0.6
        self.bad_loans = bad_loans
        
        # B.4 & B.5 Market signals
        self.cds_spread = cds_spread  # in basis points
        self.stock_volatility = stock_volatility
        
        # C.6 & C.7 Network position
        self.interbank_borrowing = interbank_borrowing
        # Asset portfolio for correlation calculation (normalized weights across asset classes)
        if asset_portfolio is not None:
            self.asset_portfolio = asset_portfolio
        else:
            # Default: random allocation across 5 asset classes
            portfolio = np.random.dirichlet(np.ones(5))
            self.asset_portfolio = portfolio
    
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
    
    def calculate_lcr(self) -> float:
        """
        Calculate the Liquidity Coverage Ratio (LCR).
        
        Formula: HQLA / Net Cash Outflows (30 days)
        A ratio < 100% indicates the bank may not survive a 30-day run.
        
        Returns:
            LCR as a percentage (e.g., 1.2 = 120%)
        """
        if self.net_cash_outflows_30d == 0:
            return float('inf')
        return self.hqla / self.net_cash_outflows_30d
    
    def calculate_npl_ratio(self) -> float:
        """
        Calculate the Non-Performing Loan Ratio (NPL).
        
        Formula: Bad Loans / Total Loans
        Indicates the quality of the bank's existing business.
        
        Returns:
            NPL ratio as a decimal (e.g., 0.05 = 5%)
        """
        if self.total_loans == 0:
            return 0.0
        return self.bad_loans / self.total_loans
    
    def is_leverage_high_risk(self, threshold: float = 20.0) -> bool:
        """
        Check if leverage ratio exceeds high-risk threshold.
        
        Args:
            threshold: Leverage threshold (default: 20x)
        
        Returns:
            True if leverage > threshold
        """
        return self.calculate_leverage() > threshold
    
    def is_lcr_desperate(self, threshold: float = 1.0) -> bool:
        """
        Check if LCR is below the critical threshold.
        
        Args:
            threshold: LCR threshold (default: 100%)
        
        Returns:
            True if LCR < threshold (bank is desperate)
        """
        return self.calculate_lcr() < threshold
    
    def is_cds_warning(self, threshold: float = 200.0) -> bool:
        """
        Check if CDS spread indicates market fear (warning signal).
        
        A spike from 50bps to 200bps is a screaming warning signal.
        
        Args:
            threshold: CDS spread threshold in basis points (default: 200)
        
        Returns:
            True if CDS spread >= threshold
        """
        return self.cds_spread >= threshold
    
    def update_cds_spread(self, new_spread: float) -> None:
        """
        Update the bank's CDS spread.
        
        Args:
            new_spread: New CDS spread in basis points
        """
        if new_spread < 0:
            raise ValueError("CDS spread cannot be negative")
        self.cds_spread = new_spread
    
    def update_stock_volatility(self, new_volatility: float) -> None:
        """
        Update the bank's stock volatility.
        
        Args:
            new_volatility: New stock volatility (standard deviation)
        """
        if new_volatility < 0:
            raise ValueError("Volatility cannot be negative")
        self.stock_volatility = new_volatility
    
    def update_interbank_borrowing(self, amount: float) -> None:
        """
        Update the bank's total interbank borrowing.
        
        Args:
            amount: New total interbank borrowing amount
        """
        if amount < 0:
            raise ValueError("Interbank borrowing cannot be negative")
        self.interbank_borrowing = amount
    
    def add_bad_loan(self, amount: float) -> None:
        """
        Add to the bank's non-performing loans.
        
        Args:
            amount: Amount to add to bad loans
        """
        if amount < 0:
            raise ValueError("Bad loan amount cannot be negative")
        self.bad_loans += amount
    
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
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get all risk metrics as defined in task.txt.
        
        Returns:
            Dictionary containing all risk metrics:
            - A. Balance Sheet Fundamentals
            - B. Market Signals
            - C. Network Position
        """
        return {
            # A. Balance Sheet Fundamentals
            'leverage_ratio': self.calculate_leverage() if self.equity > 0 else float('inf'),
            'leverage_high_risk': self.is_leverage_high_risk(),
            'lcr': self.calculate_lcr(),
            'lcr_desperate': self.is_lcr_desperate(),
            'npl_ratio': self.calculate_npl_ratio(),
            'total_loans': self.total_loans,
            'bad_loans': self.bad_loans,
            
            # B. Market Signals
            'cds_spread': self.cds_spread,
            'cds_warning': self.is_cds_warning(),
            'stock_volatility': self.stock_volatility,
            
            # C. Network Position
            'interbank_borrowing': self.interbank_borrowing,
        }
    
    def calculate_counterparty_risk_score(self) -> float:
        """
        Calculate a composite risk score for this bank as a potential counterparty.
        
        Higher score = riskier counterparty (should avoid lending to).
        Score is normalized between 0 and 1.
        
        Components (weighted):
        - Leverage risk (20%)
        - LCR risk (20%)
        - NPL risk (15%)
        - CDS spread risk (25%)
        - Stock volatility risk (10%)
        - Interbank exposure (10%)
        
        Returns:
            Risk score between 0 and 1
        """
        score = 0.0
        
        # Leverage risk: normalized, higher leverage = higher risk
        leverage = self.calculate_leverage() if self.equity > 0 else 30.0
        leverage_risk = min(leverage / 30.0, 1.0)  # Cap at 30x
        score += 0.20 * leverage_risk
        
        # LCR risk: lower LCR = higher risk
        lcr = self.calculate_lcr()
        lcr_risk = max(0, min(1.0, 1.5 - lcr)) if lcr != float('inf') else 0.0
        score += 0.20 * lcr_risk
        
        # NPL risk: higher NPL = higher risk
        npl = self.calculate_npl_ratio()
        npl_risk = min(npl / 0.10, 1.0)  # Cap at 10% NPL
        score += 0.15 * npl_risk
        
        # CDS spread risk: higher spread = higher risk
        cds_risk = min(self.cds_spread / 300.0, 1.0)  # Cap at 300bps
        score += 0.25 * cds_risk
        
        # Stock volatility risk: higher vol = higher risk
        vol_risk = min(self.stock_volatility / 0.10, 1.0)  # Cap at 10%
        score += 0.10 * vol_risk
        
        # Interbank exposure risk: normalized by assets
        if self.assets > 0:
            interbank_risk = min(self.interbank_borrowing / self.assets, 1.0)
        else:
            interbank_risk = 1.0
        score += 0.10 * interbank_risk
        
        return score
    
    def calculate_asset_correlation(self, other_bank: 'Bank') -> float:
        """
        Calculate asset correlation between this bank and another bank.
        
        Formula: Pearson correlation coefficient between asset portfolios.
        If both banks hold the same assets (e.g., tech stocks), a crash hurts both.
        Bank A should avoid lending to someone who gets sick at the same time.
        
        Args:
            other_bank: The other bank to compare with
        
        Returns:
            Correlation coefficient between -1 and 1
        """
        if self.asset_portfolio is None or other_bank.asset_portfolio is None:
            return 0.0
        
        # Ensure portfolios are same size
        if len(self.asset_portfolio) != len(other_bank.asset_portfolio):
            return 0.0
        
        # Calculate Pearson correlation
        return float(np.corrcoef(self.asset_portfolio, other_bank.asset_portfolio)[0, 1])
    
    def should_lend_to(self, borrower: 'Bank', 
                       max_leverage: float = 20.0,
                       min_lcr: float = 1.0,
                       max_cds: float = 200.0,
                       max_correlation: float = 0.7,
                       max_risk_score: float = 0.6) -> Dict[str, any]:
        """
        Evaluate whether this bank should lend to another bank.
        
        Based on task.txt criteria:
        - High leverage (>20x) is risky
        - LCR < 100% means they're desperate
        - CDS spread spike is a warning
        - High asset correlation means correlated risk
        
        Args:
            borrower: The potential borrower bank
            max_leverage: Maximum acceptable leverage (default: 20x)
            min_lcr: Minimum acceptable LCR (default: 100%)
            max_cds: Maximum acceptable CDS spread (default: 200bps)
            max_correlation: Maximum acceptable asset correlation (default: 0.7)
            max_risk_score: Maximum acceptable risk score (default: 0.6)
        
        Returns:
            Dictionary with recommendation and detailed risk breakdown
        """
        borrower_metrics = borrower.get_risk_metrics()
        correlation = self.calculate_asset_correlation(borrower)
        risk_score = borrower.calculate_counterparty_risk_score()
        
        # Individual risk flags
        flags = {
            'high_leverage': borrower_metrics['leverage_ratio'] > max_leverage,
            'low_lcr': borrower_metrics['lcr'] < min_lcr,
            'high_cds': borrower_metrics['cds_spread'] > max_cds,
            'high_correlation': correlation > max_correlation,
            'high_risk_score': risk_score > max_risk_score,
            'already_heavily_indebted': borrower.interbank_borrowing > borrower.assets * 0.3
        }
        
        # Count red flags
        num_flags = sum(flags.values())
        
        # Recommendation
        if num_flags == 0:
            recommendation = "APPROVE"
            confidence = "HIGH"
        elif num_flags <= 2:
            recommendation = "CAUTION"
            confidence = "MEDIUM"
        else:
            recommendation = "REJECT"
            confidence = "HIGH"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'num_red_flags': num_flags,
            'risk_flags': flags,
            'borrower_risk_score': risk_score,
            'asset_correlation': correlation,
            'borrower_metrics': borrower_metrics
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
