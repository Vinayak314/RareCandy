"""
Unit tests for ShockGenerator and CascadeSimulator.
"""

import pytest
from cascade_simulator import ShockGenerator, CascadeSimulator, ShockType
from bank import Bank, CCP
from loan import InterbankLoan


class TestShockGenerator:
    """Test suite for ShockGenerator."""
    
    def test_idiosyncratic_shock(self):
        """Test single bank shock generation."""
        gen = ShockGenerator(seed=42)
        shock = gen.idiosyncratic_shock("BANK-A", magnitude=0.5)
        
        assert shock.shock_type == ShockType.IDIOSYNCRATIC
        assert shock.target_bank_ids == ["BANK-A"]
        assert shock.magnitude == 0.5
    
    def test_systematic_shock(self):
        """Test multi-bank shock generation."""
        gen = ShockGenerator(seed=42)
        shock = gen.systematic_shock(["BANK-A", "BANK-B", "BANK-C"], magnitude=0.3)
        
        assert shock.shock_type == ShockType.SYSTEMATIC
        assert len(shock.target_bank_ids) == 3
        assert shock.magnitude == 0.3
    
    def test_random_shock(self):
        """Test random shock generation."""
        gen = ShockGenerator(seed=42)
        banks = {
            "BANK-A": Bank("BANK-A", 100, 1000, 900, 200),
            "BANK-B": Bank("BANK-B", 100, 1000, 900, 200),
        }
        shock = gen.random_shock(banks)
        
        assert shock.shock_type == ShockType.IDIOSYNCRATIC
        assert shock.target_bank_ids[0] in banks


class TestCascadeSimulator:
    """Test suite for CascadeSimulator."""
    
    @pytest.fixture
    def simple_network(self):
        """Create a simple 3-bank network for testing."""
        banks = {
            "BANK-A": Bank("BANK-A", equity=100, assets=1000, liabilities=900, liquidity=200),
            "BANK-B": Bank("BANK-B", equity=100, assets=1000, liabilities=900, liquidity=200),
            "BANK-C": Bank("BANK-C", equity=100, assets=1000, liabilities=900, liquidity=200),
        }
        
        # A lends to B: $150 (large exposure)
        # B lends to C: $150
        loans = [
            InterbankLoan("L1", "BANK-A", "BANK-B", 150, 0.05, 30),
            InterbankLoan("L2", "BANK-B", "BANK-C", 150, 0.05, 30),
        ]
        
        return banks, loans
    
    def test_apply_shock(self, simple_network):
        """Test applying a shock to a bank."""
        banks, loans = simple_network
        sim = CascadeSimulator(banks, loans)
        
        gen = ShockGenerator()
        shock = gen.idiosyncratic_shock("BANK-C", magnitude=1.0)  # Wipe out C
        
        defaults = sim.apply_shock(shock)
        
        assert "BANK-C" in defaults
        assert "BANK-C" in sim.defaulted_banks
    
    def test_no_cascade_small_shock(self, simple_network):
        """Test that small shocks don't cause cascades."""
        banks, loans = simple_network
        sim = CascadeSimulator(banks, loans)
        
        gen = ShockGenerator()
        shock = gen.idiosyncratic_shock("BANK-C", magnitude=0.3)  # Small shock
        
        result = sim.run_simulation(shock)
        
        # C might survive a 30% hit
        assert result['cascade_size'] == 0 or result['cascade_size'] == 1
    
    def test_cascade_propagation(self, simple_network):
        """Test that defaults cascade through the network."""
        banks, loans = simple_network
        # Reduce equity to make cascade more likely
        for b in banks.values():
            b.equity = 50
            b.assets = 950
        
        sim = CascadeSimulator(banks, loans, recovery_rate=0.0)  # No recovery
        
        gen = ShockGenerator()
        shock = gen.idiosyncratic_shock("BANK-C", magnitude=1.0)  # Kill C
        
        result = sim.run_simulation(shock)
        
        # C defaults -> B loses 150 -> B defaults -> A loses 150 -> A might default
        assert result['cascade_size'] >= 1
        assert "BANK-C" in result['defaulted_banks']
    
    def test_run_simulation_returns_metrics(self, simple_network):
        """Test that simulation returns proper metrics."""
        banks, loans = simple_network
        sim = CascadeSimulator(banks, loans)
        
        gen = ShockGenerator()
        shock = gen.idiosyncratic_shock("BANK-A", magnitude=0.5)
        
        result = sim.run_simulation(shock)
        
        assert 'shock' in result
        assert 'cascade_size' in result
        assert 'cascade_depth' in result
        assert 'survival_rate' in result
        assert 'total_equity_lost' in result
        assert 0 <= result['survival_rate'] <= 1
    
    def test_systemic_fragility(self, simple_network):
        """Test systemic fragility calculation."""
        banks, loans = simple_network
        sim = CascadeSimulator(banks, loans)
        
        # No defaults yet
        assert sim.get_systemic_fragility() == 0.0
        
        # Apply a shock that causes default
        gen = ShockGenerator()
        shock = gen.idiosyncratic_shock("BANK-A", magnitude=1.0)
        sim.run_simulation(shock)
        
        # Now there should be fragility
        fragility = sim.get_systemic_fragility()
        assert fragility > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
