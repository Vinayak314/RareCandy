"""Quick test script for model2.py contagion simulation"""
import random
random.seed(42)

from model2 import (
    load_bank_attributes, 
    generate_random_graph_with_sccs, 
    BankingNetworkContagion,
    load_stock_prices,
    distribute_shares
)

print("="*60)
print("CCP CONTAGION SIMULATION RESULTS")
print("="*60)

# Load data
bank_attrs = load_bank_attributes('us_banks_top50_nodes_final.csv')
G = generate_random_graph_with_sccs(bank_attrs, num_sccs=4, prob_intra=0.4, prob_inter=0.05)

# Initialize simulator
sim = BankingNetworkContagion(G)

# Show sample health scores
print("\nInitial Bank Health Scores:")
for b in ['JPM', 'BAC', 'GS', 'WFC', 'IBOC']:
    if b in sim.graph:
        print(f"  {b}: {sim.get_bank_health(b):.1f}/100")

# Run scenarios
scenarios = [
    ('IBOC', 30, "Small bank, moderate shock"),
    ('IBOC', 50, "Small bank, large shock"),
    ('JPM', 30, "Large bank, moderate shock"),
    ('JPM', 80, "Large bank, severe shock"),
    ('BAC', 80, "2nd largest, severe shock"),
]

print("\n" + "="*60)
print("CONTAGION SCENARIOS")
print("="*60)

for bank, shock, desc in scenarios:
    result = sim.propagate_devaluation(bank, shock)
    status = "SYSTEMIC COLLAPSE" if result['system_collapsed'] else "Contained"
    print(f"\n{desc}")
    print(f"  {bank} @ {shock}% shock -> {result['num_failed_banks']}/{result['total_banks']} failed")
    print(f"  Total Loss: ${result['total_asset_loss']:.2f}B | Rounds: {result['rounds_until_stability']} | {status}")
    if result['failed_banks']:
        banks_str = ', '.join(result['failed_banks'][:5])
        if len(result['failed_banks']) > 5:
            banks_str += f" +{len(result['failed_banks'])-5} more"
        print(f"  Failed: {banks_str}")

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)
