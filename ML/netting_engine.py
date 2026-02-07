"""
Multilateral Netting Engine for CCP clearing.

Collapses bilateral loans into net positions per bank against the CCP,
reducing gross exposure and margin requirements.
"""

from typing import List, Dict, Tuple
from loan import InterbankLoan
from bank import Bank, CCP


class NettingEngine:
    """
    Implements multilateral netting to reduce counterparty exposure.
    
    In bilateral markets, Bank A might owe Bank B $100, while B owes A $80.
    Netting collapses this to: A owes CCP $20, B receives $20 from CCP.
    """
    
    def __init__(self, ccp: CCP):
        """
        Initialize the netting engine with a CCP.
        
        Args:
            ccp: The Central Counterparty that will hold net positions
        """
        self.ccp = ccp
    
    def calculate_bilateral_exposures(
        self, 
        loans: List[InterbankLoan]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate gross bilateral exposures from loan list.
        
        Args:
            loans: List of bilateral loans
            
        Returns:
            Dict mapping (lender_id, borrower_id) -> total principal
        """
        exposures = {}
        for loan in loans:
            key = (loan.lender_id, loan.borrower_id)
            exposures[key] = exposures.get(key, 0) + loan.principal
        return exposures
    
    def calculate_net_positions(
        self, 
        loans: List[InterbankLoan]
    ) -> Dict[str, float]:
        """
        Calculate net position for each bank after multilateral netting.
        
        Positive = bank is owed money (net receiver)
        Negative = bank owes money (net payer)
        
        Args:
            loans: List of bilateral loans
            
        Returns:
            Dict mapping bank_id -> net_position
        """
        # Track what each bank is owed vs. owes
        receivables = {}  # What banks are owed (as lenders)
        payables = {}     # What banks owe (as borrowers)
        
        for loan in loans:
            # Lender is owed the principal
            receivables[loan.lender_id] = receivables.get(loan.lender_id, 0) + loan.principal
            # Borrower owes the principal
            payables[loan.borrower_id] = payables.get(loan.borrower_id, 0) + loan.principal
        
        # Net position = receivables - payables
        all_banks = set(receivables.keys()) | set(payables.keys())
        net_positions = {}
        
        for bank_id in all_banks:
            net = receivables.get(bank_id, 0) - payables.get(bank_id, 0)
            net_positions[bank_id] = net
        
        return net_positions
    
    def apply_netting(self, loans: List[InterbankLoan]) -> Dict[str, float]:
        """
        Apply multilateral netting and register positions with CCP.
        
        Args:
            loans: List of bilateral loans to net
            
        Returns:
            Dict of net positions registered with CCP
        """
        net_positions = self.calculate_net_positions(loans)
        
        # Register with CCP
        for bank_id, position in net_positions.items():
            self.ccp.add_netting_position(bank_id, position)
        
        return net_positions
    
    def calculate_compression_ratio(self, loans: List[InterbankLoan]) -> float:
        """
        Calculate the compression ratio achieved by netting.
        
        Compression = 1 - (net_exposure / gross_exposure)
        Higher is better (more exposure eliminated).
        
        Args:
            loans: List of bilateral loans
            
        Returns:
            Compression ratio between 0 and 1
        """
        # Gross exposure = sum of all loan principals
        gross_exposure = sum(loan.principal for loan in loans)
        
        # Net exposure = sum of absolute net positions / 2
        # (divide by 2 because net positions sum to zero)
        net_positions = self.calculate_net_positions(loans)
        net_exposure = sum(abs(pos) for pos in net_positions.values()) / 2
        
        if gross_exposure == 0:
            return 0.0
        
        return 1 - (net_exposure / gross_exposure)
    
    def calculate_margin_requirements(
        self, 
        loans: List[InterbankLoan]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate margin requirements before and after netting.
        
        Args:
            loans: List of bilateral loans
            
        Returns:
            Dict with 'before' and 'after' margin requirements per bank
        """
        net_positions = self.calculate_net_positions(loans)
        
        # Group loans by bank
        bank_gross = {}
        for loan in loans:
            # Lender's exposure
            bank_gross[loan.lender_id] = bank_gross.get(loan.lender_id, 0) + loan.principal
            # Borrower's exposure (they owe)
            bank_gross[loan.borrower_id] = bank_gross.get(loan.borrower_id, 0) + loan.principal
        
        results = {
            'before': {},  # Gross margin requirements
            'after': {},   # Net margin requirements
            'savings': {}  # Margin saved per bank
        }
        
        for bank_id in net_positions.keys():
            gross = bank_gross.get(bank_id, 0)
            net = abs(net_positions[bank_id])
            
            margin_before = self.ccp.calculate_initial_margin(gross)
            margin_after = self.ccp.calculate_initial_margin(net)
            
            results['before'][bank_id] = margin_before
            results['after'][bank_id] = margin_after
            results['savings'][bank_id] = margin_before - margin_after
        
        return results
    
    def get_netting_report(self, loans: List[InterbankLoan]) -> Dict:
        """
        Generate a comprehensive netting report.
        
        Args:
            loans: List of bilateral loans
            
        Returns:
            Dict with full netting statistics
        """
        net_positions = self.calculate_net_positions(loans)
        compression = self.calculate_compression_ratio(loans)
        margins = self.calculate_margin_requirements(loans)
        
        gross_exposure = sum(loan.principal for loan in loans)
        net_exposure = sum(abs(pos) for pos in net_positions.values()) / 2
        
        return {
            'num_loans': len(loans),
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'compression_ratio': compression,
            'net_positions': net_positions,
            'num_net_payers': sum(1 for p in net_positions.values() if p < 0),
            'num_net_receivers': sum(1 for p in net_positions.values() if p > 0),
            'total_margin_before': sum(margins['before'].values()),
            'total_margin_after': sum(margins['after'].values()),
            'total_margin_saved': sum(margins['savings'].values())
        }
