import pandas as pd

# Load the CSV file
file_path = 'us_banks_top50_nodes_final.csv'
df = pd.read_csv(file_path)

# 1. Calculate new metrics for each bank
# Leverage Ratio (LR) = Total Assets / Equity
# Liquidity Coverage Ratio (LCR) = HQLA / Net Outflows 30d
df['LR'] = df['Total_Assets'] / df['Equity']
df['LCR'] = df['HQLA'] / df['Net_Outflows_30d']

# Save the per-bank ratios to a separate CSV file
df[['Bank', 'LR', 'LCR']].to_csv('bank_ratios.csv', index=False)

# 2. Calculate min, max, and difference for all numeric columns
numeric_df = df.select_dtypes(include=['number'])
stats = {}
for col in numeric_df.columns:
    min_val = numeric_df[col].min()
    max_val = numeric_df[col].max()
    stats[col] = {
        'min': min_val,
        'max': max_val,
        'diff': max_val - min_val
    }

# 3. Write the results to a text file
output_file = 'column_stats.txt'
with open(output_file, 'w') as f:
    f.write("Updated Column Statistics (Including LR and LCR)\n")
    f.write("=" * 50 + "\n")
    for col, values in stats.items():
        f.write(f"Column: {col}\n")
        f.write(f"  Minimum:    {values['min']:.4f}\n")
        f.write(f"  Maximum:    {values['max']:.4f}\n")
        f.write(f"  Difference: {values['diff']:.4f}\n")
        f.write("-" * 35 + "\n")

print(f"Ratios saved to 'bank_ratios.csv' and stats saved to '{output_file}'")