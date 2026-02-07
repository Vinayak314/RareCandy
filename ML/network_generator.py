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
        liquidity_ratio_range: Tuple[float, float] = (0.1, 0.3),
        cds_spread_range: Tuple[float, float] = (30, 150),
        volatility_range: Tuple[float, float] = (0.01, 0.05),
        npl_ratio_range: Tuple[float, float] = (0.01, 0.05),
        n_asset_classes: int = 5
    ) -> Dict[int, Bank]:
        """
        Assign financial attributes to nodes based on their degree (centrality).
        
        Larger banks (hubs) naturally have more assets and lower leverage.
        
        Args:
            G: The topology graph
            total_assets_range: (min, max) for bank assets
            leverage_range: (min, max) for bank leverage
            liquidity_ratio_range: (min, max) for liquidity/liabilities
            cds_spread_range: (min, max) for CDS spread in basis points
            volatility_range: (min, max) for stock volatility
            npl_ratio_range: (min, max) for non-performing loan ratio
            n_asset_classes: Number of asset classes for portfolio (default: 5)
            
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
            
            # New attributes from task.txt
            # HQLA (High Quality Liquid Assets) - typically 80-120% of liquidity
            hqla = liquidity * (0.8 + np.random.random() * 0.4)
            
            # Net cash outflows (30 days) - typically 5-15% of liabilities
            net_cash_outflows_30d = liabilities * (0.05 + np.random.random() * 0.1)
            
            # Total loans - typically 50-70% of assets
            total_loans = assets * (0.5 + np.random.random() * 0.2)
            
            # Bad loans (NPL) - inverse correlation with hub status (bigger banks more stable)
            npl_ratio = npl_ratio_range[1] - norm_deg * (npl_ratio_range[1] - npl_ratio_range[0])
            npl_ratio += np.random.random() * 0.01  # Small random variance
            bad_loans = total_loans * npl_ratio
            
            # CDS spread - inverse correlation with size (bigger = safer = lower spread)
            cds_spread = cds_spread_range[1] - norm_deg * (cds_spread_range[1] - cds_spread_range[0])
            cds_spread += np.random.random() * 20  # Random variance
            
            # Stock volatility - inverse correlation with size
            stock_volatility = volatility_range[1] - norm_deg * (volatility_range[1] - volatility_range[0])
            stock_volatility += np.random.random() * 0.01
            
            # Asset portfolio - Dirichlet distribution for realistic allocation
            asset_portfolio = np.random.dirichlet(np.ones(n_asset_classes))
            
            banks[node] = Bank(
                bank_id=f"BANK-{node:03d}",
                equity=equity,
                assets=assets,
                liabilities=liabilities,
                liquidity=liquidity,
                risk_weight=risk_weight,
                hqla=hqla,
                net_cash_outflows_30d=net_cash_outflows_30d,
                total_loans=total_loans,
                bad_loans=bad_loans,
                cds_spread=cds_spread,
                stock_volatility=stock_volatility,
                asset_portfolio=asset_portfolio,
                interbank_borrowing=0.0  # Will be calculated from loans
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

    def calculate_interbank_borrowing(
        self,
        banks: Dict[int, Bank],
        loans: List[InterbankLoan]
    ) -> Dict[str, float]:
        """
        Calculate total interbank borrowing for each bank from loans.
        
        Updates the interbank_borrowing attribute of each bank.
        
        Args:
            banks: Dictionary of Bank objects
            loans: List of interbank loans
            
        Returns:
            Dictionary mapping bank_id -> total interbank borrowing
        """
        borrowing = {}
        
        # Sum up all borrowing by bank
        for loan in loans:
            borrowing[loan.borrower_id] = borrowing.get(loan.borrower_id, 0) + loan.principal
        
        # Update bank objects
        for bank in banks.values():
            if bank.bank_id in borrowing:
                bank.update_interbank_borrowing(borrowing[bank.bank_id])
        
        return borrowing

    def calculate_asset_correlation_matrix(
        self,
        banks: Dict[int, Bank]
    ) -> np.ndarray:
        """
        Calculate the asset correlation matrix between all banks.
        
        High correlation means a crash in one asset class hurts multiple banks.
        
        Args:
            banks: Dictionary of Bank objects
            
        Returns:
            Correlation matrix (n_banks x n_banks)
        """
        bank_list = list(banks.values())
        n = len(bank_list)
        correlation_matrix = np.zeros((n, n))
        
        for i, bank_a in enumerate(bank_list):
            for j, bank_b in enumerate(bank_list):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                elif i < j:
                    corr = bank_a.calculate_asset_correlation(bank_b)
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr  # Symmetric
        
        return correlation_matrix

    def get_risk_assessment(
        self,
        banks: Dict[int, Bank],
        loans: List[InterbankLoan]
    ) -> Dict:
        """
        Generate comprehensive risk assessment for the network.
        
        Includes all metrics from task.txt:
        A. Balance Sheet Fundamentals
        B. Market Signals
        C. Network Position
        
        Args:
            banks: Dictionary of Bank objects
            loans: List of interbank loans
            
        Returns:
            Dictionary with network-wide risk metrics
        """
        bank_metrics = []
        for bank in banks.values():
            metrics = bank.get_risk_metrics()
            metrics['bank_id'] = bank.bank_id
            metrics['counterparty_risk_score'] = bank.calculate_counterparty_risk_score()
            bank_metrics.append(metrics)
        
        # Network-wide statistics
        leverage_ratios = [m['leverage_ratio'] for m in bank_metrics if m['leverage_ratio'] != float('inf')]
        lcr_values = [m['lcr'] for m in bank_metrics if m['lcr'] != float('inf')]
        npl_ratios = [m['npl_ratio'] for m in bank_metrics]
        cds_spreads = [m['cds_spread'] for m in bank_metrics]
        volatilities = [m['stock_volatility'] for m in bank_metrics]
        risk_scores = [m['counterparty_risk_score'] for m in bank_metrics]
        
        # Calculate correlation matrix
        correlation_matrix = self.calculate_asset_correlation_matrix(banks)
        avg_correlation = np.mean(correlation_matrix[np.triu_indices(len(banks), k=1)])
        
        return {
            'summary': {
                'avg_leverage': np.mean(leverage_ratios) if leverage_ratios else 0,
                'max_leverage': max(leverage_ratios) if leverage_ratios else 0,
                'high_leverage_banks': sum(1 for l in leverage_ratios if l > 20),
                'avg_lcr': np.mean(lcr_values) if lcr_values else 0,
                'min_lcr': min(lcr_values) if lcr_values else 0,
                'desperate_banks': sum(1 for l in lcr_values if l < 1.0),
                'avg_npl_ratio': np.mean(npl_ratios),
                'max_npl_ratio': max(npl_ratios),
                'avg_cds_spread': np.mean(cds_spreads),
                'max_cds_spread': max(cds_spreads),
                'warning_signal_banks': sum(1 for s in cds_spreads if s >= 200),
                'avg_volatility': np.mean(volatilities),
                'avg_risk_score': np.mean(risk_scores),
                'high_risk_banks': sum(1 for s in risk_scores if s > 0.6),
                'avg_asset_correlation': avg_correlation,
                'total_interbank_exposure': sum(loan.principal for loan in loans),
            },
            'bank_metrics': bank_metrics,
            'correlation_matrix': correlation_matrix.tolist()
        }

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
        
        # 4. Calculate and update interbank borrowing
        self.calculate_interbank_borrowing(banks, loans)
        
        # 5. Central Counterparty
        ccp = CCP(
            ccp_id="CCP-MAIN",
            equity=ccp_equity,
            default_fund_size=ccp_equity * 0.5
        )
        
        return G, banks, loans, ccp
