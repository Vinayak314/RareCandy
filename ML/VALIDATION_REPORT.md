# Model Validation Report: Ground Truth vs Predictions

## Executive Summary

Rigorous validation comparing GNN model predictions against actual contagion simulations for 18 test scenarios.

**Key Finding**: The model has a **systematic bias** - it predicts CCP capital requirements even when ground truth shows **no contagion occurs**.

---

## Validation Methodology

### Test Scenarios (18 total):
1. **Shock magnitude sweep** (JPM bank): 10%, 20%, 30%, 40%, 50%
2. **Bank size comparison** (30% shock): Large (JPM), Medium (CFG), Small (IBOC)
3. **Random test set samples**: 10 scenarios from held-out test data

### Ground Truth Generation:
- Ran **actual contagion simulations** for each scenario
- Applied same logic as training: equity shocks, LCR thresholds, network propagation
- Calculated true minimum CCP capital needed

---

## Critical Discovery: The "No Contagion" Problem

### What We Found:

**Ground Truth**: Many scenarios show **$0 CCP capital** needed because:
- Shocked bank doesn't default (equity/LCR still above thresholds)
- No contagion spreads through network
- System remains stable

**Model Predictions**: Model **always predicts positive capital** requirements:
- JPM 10% shock: Predicts $31.79B (Truth: $0)
- JPM 50% shock: Predicts $921.13B (Truth: $0)
- Small banks: Predicts $6-50B (Truth: $0)

### Why This Happens:

The model learned from **10,000 training scenarios** where:
- Many scenarios had contagion (positive CCP capital)
- Model learned to predict based on bank features + shock magnitude
- **BUT**: Model doesn't capture the binary nature of contagion (happens vs doesn't happen)

---

## Detailed Results

### Scenarios Where Ground Truth = $0 (No Contagion):

| Scenario | Shock | Ground Truth | Prediction | Error |
|----------|-------|--------------|------------|-------|
| JPM | 10% | $0 | $31.79B | +$31.79B |
| JPM | 20% | $0 | $221.22B | +$221.22B |
| JPM | 30% | $0 | $514.68B | +$514.68B |
| JPM | 40% | $0 | $758.39B | +$758.39B |
| JPM | 50% | $0 | $921.13B | +$921.13B |
| CFG (Medium) | 30% | $0 | $6.89B | +$6.89B |
| IBOC (Small) | 30% | $0 | $12.87B | +$12.87B |

**Pattern**: Model systematically **over-predicts** when no contagion occurs.

### Scenarios Where Ground Truth = $475.21B (Contagion Occurs):

| Scenario | Shock | Ground Truth | Prediction | Error | Error % |
|----------|-------|--------------|------------|-------|---------|
| WSFS | 10% | $475.21B | $440.84B | -$34.37B | -7.2% |
| KEY | 20% | $475.21B | $282.13B | -$193.09B | -40.6% |
| COF | 20% | $475.21B | $378.75B | -$96.46B | -20.3% |
| WAL | 20% | $475.21B | $138.84B | -$336.37B | -70.8% |
| CATY | 20% | $475.21B | $133.55B | -$341.66B | -71.9% |

**Pattern**: When contagion **does** occur, model **under-predicts** capital needs.

---

## Performance Metrics

### Overall:
- **MAE**: $232.96B
- **RMSE**: $356.86B  
- **MAPE**: 11.7%
- **R²**: -1.81 (negative = worse than mean baseline)

### By Shock Magnitude:
- **10% shock**: MAE = $33.08B, MAPE = 3.6%
- **20% shock**: MAE = $237.76B, MAPE = 40.7%
- **30% shock**: MAE = $185.84B
- **40% shock**: MAE = $291.50B
- **50% shock**: MAE = $474.41B

---

## Root Cause Analysis

### Why the Model Struggles:

1. **Binary Classification Hidden in Regression**:
   - Real problem: "Will contagion happen?" (Yes/No) → "How much capital?" (Amount)
   - Model treats as pure regression, missing the classification step

2. **Training Data Imbalance**:
   - Training data likely has mix of contagion/no-contagion scenarios
   - Model learned to predict average, not binary outcome

3. **Network Threshold Effects**:
   - Contagion has sharp thresholds (equity < 0, LCR < 1.0)
   - Small changes in shock can flip from no-contagion to full-contagion
   - Model's smooth predictions can't capture this discontinuity

---

## Recommendations

### Short-term (Use Current Model):
1. **Interpret predictions conservatively**:
   - Predictions < $100B → Likely no contagion (be cautious)
   - Predictions > $300B → Likely significant contagion

2. **Use for relative comparisons**:
   - Compare scenarios: "Which bank is riskier?"
   - Trend analysis: "How does risk change with shock magnitude?"

### Long-term (Improve Model):

1. **Two-Stage Model**:
   ```
   Stage 1: Binary classifier → Will contagion occur? (Yes/No)
   Stage 2: Regression model → If yes, how much capital?
   ```

2. **Add Threshold Features**:
   - Distance to default threshold
   - LCR margin above 1.0
   - Network vulnerability metrics

3. **Rebalance Training Data**:
   - Ensure equal representation of contagion/no-contagion scenarios
   - Use weighted loss function

4. **Alternative Architectures**:
   - Add LSTM layers for temporal dynamics
   - Use mixture of experts (separate models for different regimes)
   - Implement attention mechanisms to identify critical banks

---

## Conclusion

The GNN model shows **promise** but has a **critical limitation**: it cannot distinguish between scenarios where contagion occurs vs doesn't occur.

**Current Best Use**: 
- Stress testing and scenario comparison
- Identifying systemically important banks
- Understanding relative risk levels

**Not Recommended For**:
- Precise capital requirement calculations
- Binary contagion predictions
- Regulatory compliance (without further validation)

**Next Steps**: Implement two-stage classification + regression approach for production use.

---

## Visualizations

See [`validation_plots.png`](file:///c:/Users/vinayak/Desktop/RareCandy/ML/dataset/validation_plots.png) for:
- Predicted vs Ground Truth scatter
- Error distribution
- Error by shock magnitude
- Percentage error by scenario

## Data

Full validation results: [`validation_results.csv`](file:///c:/Users/vinayak/Desktop/RareCandy/ML/dataset/validation_results.csv)
