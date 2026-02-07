"""
Scale-Free Network Generator for financial networks using the Barabási-Albert model.
"""

from typing import List, Dict, Tuple, Optional
import networkx as nx
import numpy as np
from bank import Bank, CCP
from loan import InterbankLoan


class NetworkGenerator:
    """
    Generates realistic financial networks with scale-free properties.
    
    Scale-free networks are characterized by a power-law degree distribution,
    meaning a few 'hub' banks have many connections while most have few.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def generate_scale_free_topology(self, n: int, m: int) -> nx.Graph:
        """
        Generate a scale-free graph using the Barabási-Albert model.
        
        Args:
            n: Total number of nodes (banks)
            m: Number of edges to attach from a new node to existing nodes
            
        Returns:
            NetworkX Graph object
        """
        if m < 1 or m >= n:
            raise ValueError("m must be between 1 and n-1")
            
        # Use NetworkX BA model implementation
        return nx.barabasi_albert_graph(n, m, seed=self.seed)

    def populate_network(
        self, 
        G: nx.Graph, 
        total_assets_range: Tuple[float, float] = (1000, 10000),
        leverage_range: Tuple[float, float] = (10, 20),
        liquidity_ratio_range: Tuple[float, float] = (0.1, 0.3)
    ) -> Dict[int, Bank]:
        """
        Assign financial attributes to nodes based on their degree (centrality).
        
        Larger banks (hubs) naturally have more assets and lower leverage.
        
        Args:
            G: The topology graph
            total_assets_range: (min, max) for bank assets
            leverage_range: (min, max) for bank leverage
            liquidity_ratio_range: (min, max) for liquidity/liabilities
            
        Returns:
            Dictionary mapping node IDs to Bank objects
        """
        banks = {}
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        min_deg = min(degrees.values()) if degrees else 1
        
        for node in G.nodes():
            # Linear scaling of assets based on degree (hubs are bigger)
            # Normalize degree between 0 and 1
            norm_deg = (degrees[node] - min_deg) / (max_deg - min_deg) if max_deg > min_deg else 0.5
            
            # Hubs get more assets
            assets = total_assets_range[0] + norm_deg * (total_assets_range[1] - total_assets_range[0])
            # Hubs tend to have slightly lower leverage (more capital buffer)
            leverage = leverage_range[1] - norm_deg * (leverage_range[1] - leverage_range[0])
            
            equity = assets / leverage
            liabilities = assets - equity
            
            # Liquidity is a fraction of liabilities
            liq_ratio = liquidity_ratio_range[0] + np.random.random() * (liquidity_ratio_range[1] - liquidity_ratio_range[0])
            liquidity = liabilities * liq_ratio
            
            # Risk weight: hubs are more systemically important
            risk_weight = 0.3 + 0.7 * norm_deg
            
            banks[node] = Bank(
                bank_id=f"BANK-{node:03d}",
                equity=equity,
                assets=assets,
                liabilities=liabilities,
                liquidity=liquidity,
                risk_weight=risk_weight
            )
            
        return banks

    def create_loans(
        self, 
        G: nx.Graph, 
        banks: Dict[int, Bank],
        interest_rate_range: Tuple[float, float] = (0.01, 0.05),
        maturity_range: Tuple[int, int] = (30, 365)
    ) -> List[InterbankLoan]:
        """
        Create bilateral loans based on the graph edges.
        
        Args:
            G: The topology graph
            banks: Dictionary of Bank objects
            interest_rate_range: (min, max) interest rates
            maturity_range: (min, max) maturity in days
            
        Returns:
            List of InterbankLoan objects
        """
        loans = []
        for i, (u, v) in enumerate(G.edges()):
            # Determine lender and borrower (random for now)
            if np.random.random() > 0.5:
                lender_id, borrower_id = banks[u].bank_id, banks[v].bank_id
            else:
                lender_id, borrower_id = banks[v].bank_id, banks[u].bank_id
            
            # Principal size based on borrower liabilities (e.g., 5-10% of total liabilities)
            borrower_node = v if banks[v].bank_id == borrower_id else u
            principal = banks[borrower_node].liabilities * (0.05 + np.random.random() * 0.05)
            
            # Terms
            rate = interest_rate_range[0] + np.random.random() * (interest_rate_range[1] - interest_rate_range[0])
            maturity = np.random.randint(maturity_range[0], maturity_range[1] + 1)
            
            # Collateral (50-100% of principal for some loans)
            collateral = 0
            if np.random.random() > 0.3:  # 70% of loans are secured
                collateral = principal * (0.5 + np.random.random() * 0.5)
            
            loans.append(InterbankLoan(
                loan_id=f"LOAN-{i:04d}",
                lender_id=lender_id,
                borrower_id=borrower_id,
                principal=principal,
                interest_rate=rate,
                maturity=maturity,
                collateral=collateral
            ))
            
        return loans

    def generate_full_network(
        self, 
        n_banks: int, 
        m_connectivity: int,
        ccp_equity: float = 5000.0
    ) -> Tuple[nx.Graph, Dict[int, Bank], List[InterbankLoan], CCP]:
        """
        Generate a complete financial network with Banks, Loans, and a CCP.
        
        Args:
            n_banks: Number of banks in the network
            m_connectivity: BA model parameter
            ccp_equity: Starting equity for the CCP
            
        Returns:
            Tuple of (Graph, Banks dict, Loans list, CCP object)
        """
        # 1. Topology
        G = self.generate_scale_free_topology(n_banks, m_connectivity)
        
        # 2. Bank nodes
        banks = self.populate_network(G)
        
        # 3. Interbank edges (Loans)
        loans = self.create_loans(G, banks)
        
        # 4. Central Counterparty
        ccp = CCP(
            ccp_id="CCP-MAIN",
            equity=ccp_equity,
            default_fund_size=ccp_equity * 0.5
        )
        
        return G, banks, loans, ccp
