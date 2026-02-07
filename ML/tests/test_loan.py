"""
Unit tests for InterbankLoan class.
"""

import pytest
from datetime import datetime, timedelta
from loan import InterbankLoan


class TestInterbankLoan:
    """Test suite for InterbankLoan class."""
    
    def test_loan_initialization(self):
        """Test basic loan initialization."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=5000
        )
        
        assert loan.loan_id == "LOAN-001"
        assert loan.lender_id == "BANK-001"
        assert loan.borrower_id == "BANK-002"
        assert loan.principal == 10000
        assert loan.interest_rate == 0.05
        assert loan.maturity == 365
        assert loan.collateral == 5000
    
    def test_loan_negative_principal_raises_error(self):
        """Test that negative principal raises ValueError."""
        with pytest.raises(ValueError):
            InterbankLoan(
                loan_id="LOAN-001",
                lender_id="BANK-001",
                borrower_id="BANK-002",
                principal=-10000,
                interest_rate=0.05,
                maturity=365
            )
    
    def test_calculate_interest(self):
        """Test interest calculation."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365
        )
        
        interest = loan.calculate_interest()
        assert abs(interest - 500) < 0.01  # 10000 * 0.05 * 1 year
    
    def test_calculate_total_repayment(self):
        """Test total repayment calculation."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365
        )
        
        total = loan.calculate_total_repayment()
        assert abs(total - 10500) < 0.01  # 10000 + 500
    
    def test_get_loan_to_value_ratio(self):
        """Test LTV ratio calculation."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=12500
        )
        
        ltv = loan.get_loan_to_value_ratio()
        assert abs(ltv - 0.8) < 0.01  # 10000 / 12500
    
    def test_is_secured(self):
        """Test secured loan detection."""
        secured_loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=5000
        )
        
        unsecured_loan = InterbankLoan(
            loan_id="LOAN-002",
            lender_id="BANK-001",
            borrower_id="BANK-003",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=0
        )
        
        assert secured_loan.is_secured()
        assert not unsecured_loan.is_secured()
    
    def test_calculate_recovery_value_secured(self):
        """Test recovery value for secured loan."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=5000
        )
        
        recovery = loan.calculate_recovery_value(recovery_rate=0.4)
        # min(5000, 10000 * 0.4) = min(5000, 4000) = 4000
        assert recovery == 4000
    
    def test_calculate_recovery_value_unsecured(self):
        """Test recovery value for unsecured loan."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=0
        )
        
        recovery = loan.calculate_recovery_value(recovery_rate=0.4)
        assert recovery == 4000  # 10000 * 0.4
    
    def test_calculate_loss_given_default(self):
        """Test LGD calculation."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=0
        )
        
        lgd = loan.calculate_loss_given_default(recovery_rate=0.4)
        # Total exposure: 10500, Recovery: 4000, LGD: 6500
        assert abs(lgd - 6500) < 1
    
    def test_mark_default(self):
        """Test default marking."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365
        )
        
        assert not loan.is_defaulted()
        loan.mark_default()
        assert loan.is_defaulted()
    
    def test_get_maturity_date(self):
        """Test maturity date calculation."""
        origination = datetime(2024, 1, 1)
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            origination_date=origination
        )
        
        maturity_date = loan.get_maturity_date()
        expected = origination + timedelta(days=365)
        assert maturity_date == expected
    
    def test_get_loan_details(self):
        """Test loan details snapshot."""
        loan = InterbankLoan(
            loan_id="LOAN-001",
            lender_id="BANK-001",
            borrower_id="BANK-002",
            principal=10000,
            interest_rate=0.05,
            maturity=365,
            collateral=5000
        )
        
        details = loan.get_loan_details()
        
        assert details['loan_id'] == "LOAN-001"
        assert details['principal'] == 10000
        assert details['is_secured'] is True
        assert details['is_defaulted'] is False
        assert 'total_repayment' in details


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
