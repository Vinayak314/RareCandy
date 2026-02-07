from algo import CCPTrainingEnvironment, MarginDecision

env = CCPTrainingEnvironment()
env.ccp.load()

# Run single episode and see detailed results
result = env.run_episode(train=False)
print(f"Failed banks: {result['failed_banks']}")
print(f"Approval rate: {result['approval_rate']*100:.1f}%")
print(f"Avg health: {result['avg_health']:.1f}")

# See model's decision distribution
for d in MarginDecision:
    print(f"{d.name}: {env.ccp.decisions[d]}")