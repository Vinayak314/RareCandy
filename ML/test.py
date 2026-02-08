from algo import CCPTrainingEnvironment, MarginLevel, RL_CONFIG

env = CCPTrainingEnvironment()

# Load model
loaded = env.ccp.load(RL_CONFIG['MODEL_PATH'])

if not loaded:
    print("⚠️  No trained model found! Run 'python algo.py' first to train.")
else:
    print("\n" + "="*60)
    print("TESTING CCP MODEL")
    print("="*60)
    
    # Run multiple episodes
    num_test = 100
    results = []
    for i in range(num_test):
        result = env.run_episode(train=False)
        results.append(result)
        print(f"  Episode {i+1}: Failed={result['failed_banks']:2d}, "
              f"Survival={result['survival_rate']*100:.0f}%, "
              f"Margin={result['final_margin']*100:.0f}%")
    
    import numpy as np
    
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    print(f"Avg Failed Banks: {np.mean([r['failed_banks'] for r in results]):.1f} / {env.num_banks}")
    print(f"Avg Survival Rate: {np.mean([r['survival_rate'] for r in results])*100:.1f}%")
    print(f"Avg Reward: {np.mean([r['reward'] for r in results]):+.1f}")
    print(f"Avg Final Health: {np.mean([r['avg_health'] for r in results]):.1f}")
    
    # CCP decision distribution
    print("\n" + "-"*60)
    print("CCP MARGIN DECISIONS")
    print("-"*60)
    total = sum(env.ccp.decisions.values())
    for m in MarginLevel:
        count = env.ccp.decisions[m]
        pct = count / max(total, 1) * 100
        rate = env.ccp.MARGIN_RATES[m.value] * 100
        bar = "█" * int(pct / 2)
        print(f"  {m.name:10s} ({rate:2.0f}%): {pct:5.1f}% {bar}")