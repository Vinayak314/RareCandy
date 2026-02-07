import pandas as pd

# Load the CSV file
file_path = 'us_banks_top50_nodes_final.csv'
df = pd.read_csv(file_path)

# Filter for numeric columns only to calculate differences
numeric_df = df.select_dtypes(include=['number'])

# Calculate stats
stats = {}
for col in numeric_df.columns:
    min_val = numeric_df[col].min()
    max_val = numeric_df[col].max()
    stats[col] = {
        'min': min_val,
        'max': max_val,
        'diff': max_val - min_val
    }

# Write the results to a text file
output_file = 'column_stats.txt'
with open(output_file, 'w') as f:
    f.write("Column Statistics (Min, Max, and Difference)\n")
    f.write("=" * 45 + "\n")
    for col, values in stats.items():
        f.write(f"Column: {col}\n")
        f.write(f"  Minimum:    {values['min']}\n")
        f.write(f"  Maximum:    {values['max']}\n")
        f.write(f"  Difference: {values['diff']}\n")
        f.write("-" * 30 + "\n")

print(f"Statistics successfully saved to {output_file}")