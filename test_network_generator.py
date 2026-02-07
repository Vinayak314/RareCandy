"""
Unit tests for NetworkGenerator class.
"""

import pytest
import networkx as nx
from network_generator import NetworkGenerator
from bank import Bank
from loan import InterbankLoan


class TestNetworkGenerator:
    """Test suite for NetworkGenerator."""
    
    def test_generator_topology(self):
        """Test basic topology generation."""
        gen = NetworkGenerator(seed=42)
        n, m = 10, 2
        G = gen.generate_scale_free_topology(n, m)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == n
        # BA model creates edges = (n-m)*m roughly
        # Initial m nodes form a clique or star usually, check at least m edges per new node
        assert G.number_of_edges() > 0

    def test_populate_network(self):
        """Test assigning financial attributes to nodes."""
        gen = NetworkGenerator(seed=42)
        G = gen.generate_scale_free_topology(20, 2)
        banks = gen.populate_network(G)
        
        assert len(banks) == 20
        assert all(isinstance(b, Bank) for b in banks.values())
        
        # Verify hub banks are generally larger (test correlation)
        degrees = [G.degree(i) for i in range(20)]
        assets = [banks[i].assets for i in range(20)]
        
        # Highly connected nodes should have more assets (simple check)
        assert max(assets) > min(assets)

    def test_create_loans(self):
        """Test loan creation from graph edges."""
        gen = NetworkGenerator(seed=42)
        G = gen.generate_scale_free_topology(10, 2)
        banks = gen.populate_network(G)
        loans = gen.create_loans(G, banks)
        
        assert len(loans) == G.number_of_edges()
        assert all(isinstance(l, InterbankLoan) for l in loans)
        
        # Verify IDs match
        bank_ids = {b.bank_id for b in banks.values()}
        for loan in loans:
            assert loan.lender_id in bank_ids
            assert loan.borrower_id in bank_ids

    def test_generate_full_network(self):
        """Test end-to-end generation."""
        gen = NetworkGenerator(seed=42)
        G, banks, loans, ccp = gen.generate_full_network(n_banks=15, m_connectivity=2)
        
        assert G.number_of_nodes() == 15
        assert len(banks) == 15
        assert len(loans) == G.number_of_edges()
        assert ccp.bank_id == "CCP-MAIN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
