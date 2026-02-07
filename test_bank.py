"""
Unit tests for Bank and CCP classes.
"""

import pytest
from bank import Bank, CCP


class TestBank:
    """Test suite for Bank class."""
    
    def test_bank_initialization(self):
        """Test basic bank initialization."""
        bank = Bank(
            bank_id="BANK-001",
            equity=100,
            assets=1000,
            liabilities=900,
            liquidity=200
        )
        
        assert bank.bank_id == "BANK-001"
        assert bank.equity == 100
        assert bank.assets == 1000
        assert bank.liabilities == 900
        assert bank.liquidity == 200
        assert bank.is_solvent()
    
    def test_bank_negative_values_raise_error(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError):
            Bank(
                bank_id="BANK-001",
                equity=-100,
                assets=1000,
                liabilities=900,
                liquidity=200
            )
    
    def test_calculate_leverage(self):
        """Test leverage calculation."""
        bank = Bank(
            bank_id="BANK-001",
            equity=100,
            assets=1000,
            liabilities=900,
            liquidity=200
        )
        
        assert bank.calculate_leverage() == 10.0  # 1000 / 100
    
    def test_calculate_liquidity_ratio(self):
        """Test liquidity ratio calculation."""
        bank = Bank(
            bank_id="BANK-001",
            equity=100,
            assets=1000,
            liabilities=900,
            liquidity=450
        )
        
        assert bank.calculate_liquidity_ratio() == 0.5  # 450 / 900
    
    def test_update_balance_sheet(self):
        """Test balance sheet updates."""
        bank = Bank(
            bank_id="BANK-001",
            equity=100,
            assets=1000,
            liabilities=900,
            liquidity=200
        )
        
        # Simulate a loss
        bank.update_balance_sheet(delta_assets=-50, delta_liabilities=0)
        
        assert bank.assets == 950
        assert bank.equity == 50  # 950 - 900
        assert bank.is_solvent()
    
    def test_insolvency_detection(self):
        """Test that insolvency is correctly detected."""
        bank = Bank(
            bank_id="BANK-001",
            equity=100,
            assets=1000,
            liabilities=900,
            liquidity=200
        )
        
        # Create a large loss that wipes out equity
        bank.update_balance_sheet(delta_assets=-200, delta_liabilities=0)
        
        assert bank.equity <= 0
        assert not bank.is_solvent()
        assert bank.is_defaulted()
    
    def test_get_balance_sheet(self):
        """Test balance sheet snapshot."""
        bank = Bank(
            bank_id="BANK-001",
            equity=100,
            assets=1000,
            liabilities=900,
            liquidity=200
        )
        
        snapshot = bank.get_balance_sheet()
        
        assert snapshot['bank_id'] == "BANK-001"
        assert snapshot['equity'] == 100
        assert snapshot['leverage'] == 10.0
        assert snapshot['is_solvent'] is True


class TestCCP:
    """Test suite for CCP class."""
    
    def test_ccp_initialization(self):
        """Test CCP initialization."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000,
            initial_margin_rate=0.05,
            default_fund_size=500
        )
        
        assert ccp.bank_id == "CCP-001"
        assert ccp.equity == 1000
        assert ccp.default_fund == 500
        assert ccp.margin_requirements['initial_margin_rate'] == 0.05
        assert ccp.risk_weight == 1.0  # CCPs are systemically critical
    
    def test_calculate_initial_margin_fixed(self):
        """Test initial margin calculation with fixed percentage."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000,
            initial_margin_rate=0.05
        )
        
        margin = ccp.calculate_initial_margin(position_value=10000)
        assert margin == 500  # 10000 * 0.05
    
    def test_calculate_initial_margin_var(self):
        """Test initial margin calculation with VaR model."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000,
            initial_margin_rate=0.05
        )
        
        ccp.margin_requirements['margin_model'] = 'VaR'
        margin = ccp.calculate_initial_margin(position_value=10000, volatility=0.02)
        
        # VaR: 10000 * 0.02 * 2.33 = 466
        assert abs(margin - 466) < 1
    
    def test_calculate_variation_margin(self):
        """Test variation margin calculation."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000
        )
        
        vm = ccp.calculate_variation_margin(position_value=10000, price_change=-0.05)
        assert vm == -500  # 10000 * -0.05
    
    def test_add_netting_position(self):
        """Test adding netting positions."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000
        )
        
        ccp.add_netting_position("BANK-001", 5000)
        ccp.add_netting_position("BANK-002", -3000)
        
        assert ccp.netting_positions["BANK-001"] == 5000
        assert ccp.netting_positions["BANK-002"] == -3000
        assert ccp.get_total_exposure() == 8000  # |5000| + |-3000|
    
    def test_default_waterfall_default_fund_sufficient(self):
        """Test default waterfall when default fund covers loss."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000,
            default_fund_size=500
        )
        
        allocation = ccp.apply_default_waterfall(loss_amount=300)
        
        assert allocation['default_fund'] == 300
        assert allocation['ccp_equity'] == 0
        assert allocation['mutualized_loss'] == 0
        assert allocation['remaining_loss'] == 0
        assert ccp.default_fund == 200  # 500 - 300
        assert ccp.is_solvent()
    
    def test_default_waterfall_uses_equity(self):
        """Test default waterfall when loss exceeds default fund."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000,
            default_fund_size=500
        )
        
        allocation = ccp.apply_default_waterfall(loss_amount=800)
        
        assert allocation['default_fund'] == 500
        assert allocation['ccp_equity'] == 300
        assert allocation['mutualized_loss'] == 0
        assert allocation['remaining_loss'] == 0
        assert ccp.default_fund == 0
        assert ccp.equity == 700  # 1000 - 300
        assert ccp.is_solvent()
    
    def test_default_waterfall_ccp_fails(self):
        """Test default waterfall when CCP itself fails."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000,
            default_fund_size=500
        )
        
        allocation = ccp.apply_default_waterfall(loss_amount=2000)
        
        assert allocation['default_fund'] == 500
        assert allocation['ccp_equity'] == 1000
        assert allocation['mutualized_loss'] == 500  # Remaining loss
        assert allocation['remaining_loss'] == 500
        assert ccp.default_fund == 0
        assert ccp.equity == 0
        assert not ccp.is_solvent()
        assert ccp.is_defaulted()
    
    def test_get_ccp_status(self):
        """Test CCP status report."""
        ccp = CCP(
            ccp_id="CCP-001",
            equity=1000,
            default_fund_size=500
        )
        
        ccp.add_netting_position("BANK-001", 5000)
        ccp.add_netting_position("BANK-002", -3000)
        
        status = ccp.get_ccp_status()
        
        assert status['default_fund'] == 500
        assert status['total_exposure'] == 8000
        assert status['num_members'] == 2
        assert 'margin_requirements' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
