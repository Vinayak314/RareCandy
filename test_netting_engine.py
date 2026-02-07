"""
Unit tests for NettingEngine class.
"""

import pytest
from netting_engine import NettingEngine
from bank import CCP
from loan import InterbankLoan


class TestNettingEngine:
    """Test suite for NettingEngine."""
    
    @pytest.fixture
    def ccp(self):
        """Create a test CCP."""
        return CCP(ccp_id="CCP-TEST", equity=10000, initial_margin_rate=0.05)
    
    @pytest.fixture
    def sample_loans(self):
        """Create sample bilateral loans for testing."""
        return [
            # A lends to B: $1000
            InterbankLoan("L1", "BANK-A", "BANK-B", 1000, 0.05, 30),
            # B lends to A: $600
            InterbankLoan("L2", "BANK-B", "BANK-A", 600, 0.05, 30),
            # B lends to C: $500
            InterbankLoan("L3", "BANK-B", "BANK-C", 500, 0.05, 30),
            # C lends to A: $400
            InterbankLoan("L4", "BANK-C", "BANK-A", 400, 0.05, 30),
        ]
    
    def test_calculate_bilateral_exposures(self, ccp, sample_loans):
        """Test bilateral exposure calculation."""
        engine = NettingEngine(ccp)
        exposures = engine.calculate_bilateral_exposures(sample_loans)
        
        assert exposures[("BANK-A", "BANK-B")] == 1000
        assert exposures[("BANK-B", "BANK-A")] == 600
        assert exposures[("BANK-B", "BANK-C")] == 500
        assert exposures[("BANK-C", "BANK-A")] == 400
    
    def test_calculate_net_positions(self, ccp, sample_loans):
        """Test net position calculation."""
        engine = NettingEngine(ccp)
        net = engine.calculate_net_positions(sample_loans)
        
        # BANK-A: receives 1000, pays 600+400=1000 -> net 0
        # BANK-B: receives 600+500=1100, pays 1000 -> net +100
        # BANK-C: receives 400, pays 500 -> net -100
        assert net["BANK-A"] == 0
        assert net["BANK-B"] == 100
        assert net["BANK-C"] == -100
        
        # Net positions should sum to zero
        assert sum(net.values()) == 0
    
    def test_apply_netting(self, ccp, sample_loans):
        """Test applying netting to CCP."""
        engine = NettingEngine(ccp)
        net = engine.apply_netting(sample_loans)
        
        # Check CCP has recorded positions
        assert ccp.netting_positions["BANK-A"] == 0
        assert ccp.netting_positions["BANK-B"] == 100
        assert ccp.netting_positions["BANK-C"] == -100
    
    def test_compression_ratio(self, ccp, sample_loans):
        """Test compression ratio calculation."""
        engine = NettingEngine(ccp)
        compression = engine.calculate_compression_ratio(sample_loans)
        
        # Gross = 1000 + 600 + 500 + 400 = 2500
        # Net = (|0| + |100| + |-100|) / 2 = 100
        # Compression = 1 - 100/2500 = 0.96
        assert abs(compression - 0.96) < 0.01
    
    def test_margin_requirements(self, ccp, sample_loans):
        """Test margin requirement calculation."""
        engine = NettingEngine(ccp)
        margins = engine.calculate_margin_requirements(sample_loans)
        
        # Check savings exist
        total_before = sum(margins['before'].values())
        total_after = sum(margins['after'].values())
        
        assert total_before > total_after
        assert sum(margins['savings'].values()) > 0
    
    def test_netting_report(self, ccp, sample_loans):
        """Test comprehensive netting report."""
        engine = NettingEngine(ccp)
        report = engine.get_netting_report(sample_loans)
        
        assert report['num_loans'] == 4
        assert report['gross_exposure'] == 2500
        assert report['net_exposure'] == 100
        assert report['compression_ratio'] > 0.9
        assert report['num_net_payers'] == 1  # C
        assert report['num_net_receivers'] == 1  # B
        assert report['total_margin_saved'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
