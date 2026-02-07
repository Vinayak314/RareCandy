"""
InterbankLoan class representing bilateral lending relationships.
"""

from typing import Optional
from datetime import datetime, timedelta


class InterbankLoan:
    """
    Represents a bilateral loan between two banks in the financial network.
    
    Attributes:
        loan_id (str): Unique identifier for the loan
        lender_id (str): ID of the lending bank
        borrower_id (str): ID of the borrowing bank
        principal (float): Loan amount
        interest_rate (float): Annual interest rate
        maturity (int): Days until maturity
        collateral (float): Pledged collateral value
    """
    
    def __init__(
        self,
        loan_id: str,
        lender_id: str,
        borrower_id: str,
        principal: float,
        interest_rate: float,
        maturity: int,
        collateral: float = 0.0,
        origination_date: Optional[datetime] = None
    ):
        """
        Initialize an InterbankLoan instance.
        
        Args:
            loan_id: Unique identifier
            lender_id: Lending bank ID
            borrower_id: Borrowing bank ID
            principal: Loan amount
            interest_rate: Annual interest rate (e.g., 0.05 for 5%)
            maturity: Days until maturity
            collateral: Pledged collateral value (default: 0)
            origination_date: Loan origination date (default: now)
        
        Raises:
            ValueError: If principal, interest_rate, or maturity are invalid
        """
        if principal <= 0:
            raise ValueError("Principal must be positive")
        if interest_rate < 0:
            raise ValueError("Interest rate cannot be negative")
        if maturity <= 0:
            raise ValueError("Maturity must be positive")
        if collateral < 0:
            raise ValueError("Collateral cannot be negative")
        
        self.loan_id = loan_id
        self.lender_id = lender_id
        self.borrower_id = borrower_id
        self.principal = principal
        self.interest_rate = interest_rate
        self.maturity = maturity
        self.collateral = collateral
        self.origination_date = origination_date or datetime.now()
        self._is_defaulted = False
    
    def calculate_interest(self) -> float:
        """
        Calculate total interest over the loan's lifetime.
        
        Returns:
            Total interest amount
        """
        # Simple interest: Principal * Rate * (Days / 365)
        return self.principal * self.interest_rate * (self.maturity / 365.0)
    
    def calculate_total_repayment(self) -> float:
        """
        Calculate total repayment amount (principal + interest).
        
        Returns:
            Total amount to be repaid
        """
        return self.principal + self.calculate_interest()
    
    def get_maturity_date(self) -> datetime:
        """
        Calculate the loan's maturity date.
        
        Returns:
            Maturity date
        """
        return self.origination_date + timedelta(days=self.maturity)
    
    def get_loan_to_value_ratio(self) -> float:
        """
        Calculate loan-to-value ratio (LTV).
        
        Returns:
            LTV ratio (Principal / Collateral)
        """
        if self.collateral == 0:
            return float('inf')
        return self.principal / self.collateral
    
    def is_secured(self) -> bool:
        """
        Check if the loan is secured (has collateral).
        
        Returns:
            True if collateral > 0, False otherwise
        """
        return self.collateral > 0
    
    def mark_default(self) -> None:
        """Mark the loan as defaulted."""
        self._is_defaulted = True
    
    def is_defaulted(self) -> bool:
        """Check if the loan has defaulted."""
        return self._is_defaulted
    
    def calculate_recovery_value(self, recovery_rate: float = 0.4) -> float:
        """
        Calculate expected recovery value in case of default.
        
        Args:
            recovery_rate: Expected recovery rate (default: 40%)
        
        Returns:
            Expected recovery amount
        """
        if self.is_secured():
            # For secured loans, recovery is min(collateral, principal * recovery_rate)
            return min(self.collateral, self.principal * recovery_rate)
        else:
            # For unsecured loans, recovery is principal * recovery_rate
            return self.principal * recovery_rate
    
    def calculate_loss_given_default(self, recovery_rate: float = 0.4) -> float:
        """
        Calculate loss given default (LGD).
        
        Args:
            recovery_rate: Expected recovery rate (default: 40%)
        
        Returns:
            Expected loss amount
        """
        total_exposure = self.calculate_total_repayment()
        recovery = self.calculate_recovery_value(recovery_rate)
        return max(0, total_exposure - recovery)
    
    def get_loan_details(self) -> dict:
        """
        Get comprehensive loan details.
        
        Returns:
            Dictionary containing all loan information
        """
        return {
            'loan_id': self.loan_id,
            'lender_id': self.lender_id,
            'borrower_id': self.borrower_id,
            'principal': self.principal,
            'interest_rate': self.interest_rate,
            'maturity_days': self.maturity,
            'collateral': self.collateral,
            'total_interest': self.calculate_interest(),
            'total_repayment': self.calculate_total_repayment(),
            'ltv_ratio': self.get_loan_to_value_ratio(),
            'is_secured': self.is_secured(),
            'is_defaulted': self._is_defaulted,
            'origination_date': self.origination_date.isoformat(),
            'maturity_date': self.get_maturity_date().isoformat()
        }
    
    def __repr__(self) -> str:
        return (f"InterbankLoan(id={self.loan_id}, "
                f"{self.lender_id}->{self.borrower_id}, "
                f"principal={self.principal:.2f}, rate={self.interest_rate:.2%})")
