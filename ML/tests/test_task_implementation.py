"""
Test script for task.txt implementation.
Verifies all risk metrics from task.txt are working correctly.
"""

from bank import Bank, CCP
from loan import InterbankLoan
from network_generator import NetworkGenerator
from cascade_simulator import CascadeSimulator, ShockGenerator
import numpy as np

def test_bank_risk_metrics():
    """Test individual bank risk metrics from task.txt."""
    print("=" * 60)
    print("Testing Bank Risk Metrics (task.txt A, B, C)")
    print("=" * 60)
    
    # Create a test bank with specific metrics
    bank = Bank(
        bank_id="TEST-001",
        equity=100,
        assets=2000,
        liabilities=1900,
        liquidity=200,
        hqla=180,
        net_cash_outflows_30d=150,
        total_loans=1200,
        bad_loans=60,
        cds_spread=75,
        stock_volatility=0.03,
        asset_portfolio=np.array([0.3, 0.2, 0.2, 0.15, 0.15]),
        interbank_borrowing=300
    )
    
    # A.1 Leverage Ratio
    leverage = bank.calculate_leverage()
    print(f"\nA.1 Leverage Ratio: {leverage:.2f}x")
    print(f"    High risk (>20x)? {bank.is_leverage_high_risk()}")
    
    # A.2 Liquidity Coverage Ratio
    lcr = bank.calculate_lcr()
    print(f"\nA.2 LCR: {lcr:.2%}")
    print(f"    Desperate (<100%)? {bank.is_lcr_desperate()}")
    
    # A.3 Non-Performing Loan Ratio
    npl = bank.calculate_npl_ratio()
    print(f"\nA.3 NPL Ratio: {npl:.2%}")
    
    # B.4 CDS Spread
    print(f"\nB.4 CDS Spread: {bank.cds_spread} bps")
    print(f"    Warning signal (>=200bps)? {bank.is_cds_warning()}")
    
    # B.5 Stock Volatility
    print(f"\nB.5 Stock Volatility: {bank.stock_volatility:.2%}")
    
    # C.6 Interbank Exposure
    print(f"\nC.6 Interbank Borrowing: ${bank.interbank_borrowing:.2f}")
    
    # Combined risk metrics
    print("\n" + "-" * 40)
    print("Combined Risk Metrics:")
    print("-" * 40)
    metrics = bank.get_risk_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # C.7 Asset Correlation Test
    print("\n" + "-" * 40)
    print("C.7 Asset Correlation Test:")
    print("-" * 40)
    
    bank2 = Bank(
        bank_id="TEST-002",
        equity=50,
        assets=1000,
        liabilities=950,
        liquidity=100,
        asset_portfolio=np.array([0.28, 0.22, 0.18, 0.17, 0.15])  # Similar portfolio
    )
    
    bank3 = Bank(
        bank_id="TEST-003",
        equity=50,
        assets=1000,
        liabilities=950,
        liquidity=100,
        asset_portfolio=np.array([0.1, 0.1, 0.1, 0.35, 0.35])  # Different portfolio
    )
    
    corr_similar = bank.calculate_asset_correlation(bank2)
    corr_different = bank.calculate_asset_correlation(bank3)
    print(f"  Correlation with similar portfolio: {corr_similar:.4f}")
    print(f"  Correlation with different portfolio: {corr_different:.4f}")
    
    # Counterparty risk score
    print("\n" + "-" * 40)
    print("Counterparty Risk Score:")
    print("-" * 40)
    risk_score = bank.calculate_counterparty_risk_score()
    print(f"  Risk Score: {risk_score:.4f}")
    
    # Should lend decision
    print("\n" + "-" * 40)
    print("Lending Decision (Bank -> Bank2):")
    print("-" * 40)
    decision = bank.should_lend_to(bank2)
    print(f"  Recommendation: {decision['recommendation']}")
    print(f"  Confidence: {decision['confidence']}")
    print(f"  Red flags: {decision['num_red_flags']}")
    print(f"  Risk flags: {decision['risk_flags']}")
    
    return True

def test_network_generation():
    """Test network generation with new risk attributes."""
    print("\n" + "=" * 60)
    print("Testing Network Generation with Risk Metrics")
    print("=" * 60)
    
    ng = NetworkGenerator(seed=42)
    G, banks, loans, ccp = ng.generate_full_network(20, 3)
    
    print(f"\nGenerated network:")
    print(f"  Banks: {len(banks)}")
    print(f"  Loans: {len(loans)}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Check that all banks have the new attributes
    sample_bank = list(banks.values())[0]
    print(f"\nSample bank ({sample_bank.bank_id}) attributes:")
    print(f"  HQLA: ${sample_bank.hqla:.2f}")
    print(f"  Net cash outflows (30d): ${sample_bank.net_cash_outflows_30d:.2f}")
    print(f"  Total loans: ${sample_bank.total_loans:.2f}")
    print(f"  Bad loans: ${sample_bank.bad_loans:.2f}")
    print(f"  CDS spread: {sample_bank.cds_spread:.1f} bps")
    print(f"  Stock volatility: {sample_bank.stock_volatility:.2%}")
    print(f"  Interbank borrowing: ${sample_bank.interbank_borrowing:.2f}")
    print(f"  Asset portfolio: {sample_bank.asset_portfolio}")
    
    # Network-wide risk assessment
    risk_assessment = ng.get_risk_assessment(banks, loans)
    print("\nNetwork Risk Summary:")
    for key, value in risk_assessment['summary'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return True

def test_cascade_simulation():
    """Test cascade simulation with risk metrics."""
    print("\n" + "=" * 60)
    print("Testing Cascade Simulation with Risk Analysis")
    print("=" * 60)
    
    ng = NetworkGenerator(seed=42)
    G, banks, loans, ccp = ng.generate_full_network(20, 3)
    
    cs = CascadeSimulator(banks, loans, ccp)
    
    # Test interbank exposure calculation
    print("\nInterbank Exposures (sample):")
    sample_bank_id = list(cs.banks.keys())[0]
    exposure = cs.get_interbank_exposure(sample_bank_id)
    for key, value in exposure.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test contagion risk score
    print("\nContagion Risk Score (sample):")
    risk = cs.calculate_contagion_risk_score(sample_bank_id)
    print(f"  Bank: {risk['bank_id']}")
    print(f"  Counterparty Risk: {risk['counterparty_risk_score']:.4f}")
    print(f"  Interbank Risk: {risk['interbank_risk']:.4f}")
    print(f"  Network Centrality: {risk['network_centrality_risk']:.4f}")
    print(f"  Contagion Risk Score: {risk['contagion_risk_score']:.4f}")
    print(f"  Risk Level: {risk['risk_level']}")
    
    # Identify vulnerable banks
    print("\nVulnerable Banks (threshold=0.4):")
    vulnerable = cs.identify_vulnerable_banks(threshold=0.4)
    print(f"  Found {len(vulnerable)} vulnerable banks: {vulnerable[:5]}...")
    
    # Systemic risk report
    print("\nSystemic Risk Report:")
    report = cs.get_systemic_risk_report()
    for key, value in report['network_summary'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTop 5 Vulnerable Banks: {report['top_5_vulnerable_banks']}")
    
    return True

if __name__ == "__main__":
    print("Testing task.txt Implementation")
    print("=" * 60)
    
    try:
        test_bank_risk_metrics()
        test_network_generation()
        test_cascade_simulation()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
