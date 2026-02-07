import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- STEP 1: DEFINE THE TOP 50 US BANKS (TICKERS) ---
# Sorted roughly by asset size
tickers = [
    'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
    'SCHW', 'BK', 'STT', 'AXP', 'FITB', 'HBAN', 'RF', 'KEY', 'CFG', 'MTB',
    'SYF', 'ALLY', 'SOFI', 'ZION', 'CMA', 'WAL', 'FHN', 'WBS', 'EWBC',
    'PNFP', 'CBSH', 'BOKF', 'FNB', 'ONB', 'SNV', 'VLY', 'WTFC', 'ASB',
    'CFR', 'UMBF', 'PB', 'IBOC', 'HWC', 'UCBI', 'SF', 'BOH', 'WSFS',
    'TCBI', 'COLB', 'CATY'
]

print(f"🚀 Starting data extraction for {len(tickers)} banks...")
print("This may take 1-2 minutes. Please wait.\n")

# Storage lists
bank_data = []
close_prices = pd.DataFrame()

# --- STEP 2: LOOP AND FETCH DATA ---
# --- STEP 2: LOOP AND FETCH DATA ---
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        
        # A. Balance Sheet (Most recent annual)
        bs = stock.balance_sheet
        if bs.empty:
            print(f"⚠️ Skipping {ticker}: No balance sheet found.")
            continue
            
        latest = bs.iloc[:, 0] # Get latest year column
        
        # Get Market Cap (Fast Info) - for Merton Model
        try:
            mkt_cap = stock.fast_info['market_cap'] / 1e9
        except:
            mkt_cap = np.nan # Will fill later
        
        # B. Core Metrics (Billions)
        assets = latest.get('Total Assets', np.nan) / 1e9
        equity = latest.get('Stockholders Equity', np.nan) / 1e9
        liabs = latest.get('Total Liabilities Net Minority Interest', assets - equity) / 1e9
        
        # Fill missing mkt_cap with Book Equity (Conservative proxy)
        if pd.isna(mkt_cap):
            mkt_cap = equity 
        
        # C. Loan Metrics
        loans = latest.get('Net Loans', 
                  latest.get('Loans Receivable', 
                  latest.get('Gross Loans', assets * 0.50))) / 1e9
        
        # Bad Loans (Credit Cycle Sensitive)
        # Base 1.5% + Stress buffer if market view is weak (Market Cap < Book Value)
        base_npl = 0.015
        stress_factor = 0.0
        if mkt_cap < equity: # Market signaling distress
            stress_factor = 0.01 * (1 - (mkt_cap / equity)) # Up to +1% extra bad loans
            
        bad_loans = latest.get('Allowance For Loan And Lease Losses', loans * (base_npl + stress_factor)) / 1e9
        
        # D. Liquidity Metrics (HQLA)
        # Refinement: Include 50% of Long Term Investments (Agency MBS / Treasuries HTM)
        cash = latest.get('Cash And Cash Equivalents', 0) / 1e9
        st_invest = latest.get('Other Short Term Investments', 0) / 1e9
        lt_invest = latest.get('Investment Properties', latest.get('Long Term Investments', 0)) / 1e9
        
        hqla = cash + st_invest + (lt_invest * 0.50)
        
        # Net Cash Outflows (NCO) Proxy: Size-Dependent Run-off
        # Large banks (> $250B) rely more on wholesale funding -> Higher Run-off rate
        # Range: 5% (stickiest) to 20% (flighty)
        run_off_rate = 0.05 + 0.15 * (np.log(assets) / np.log(4000)) # Simple scaling
        run_off_rate = min(max(run_off_rate, 0.05), 0.25)
        nco = liabs * run_off_rate
        
        # E. Market Metrics (Volatility & Correlation)
        hist = stock.history(period="1y")['Close']
        if len(hist) > 0:
            close_prices[ticker] = hist
            
            # Annualized Volatility (Log Returns)
            log_returns = np.log(hist / hist.shift(1)).dropna()
            volatility = log_returns.std() * np.sqrt(252)
            
            # Implied CDS Spread (Merton Distance-to-Default Proxy)
            # Refinement using Market Cap and Volatility
            # DtD approx = ln(V/D) / (vol * sqrt(T))
            # We use a simplified spread proxy: (Vol_Equity * Leverage_Mkt) * Scale
            
            mkt_leverage = (liabs + mkt_cap) / mkt_cap
            # Higher Vol + Higher Leverage = Higher Spread
            cds_proxy = (volatility * mkt_leverage) * 100 # Basis Points
            
        else:
            volatility = 0.30 
            cds_proxy = 150
            
        # F. Append to List
        bank_data.append([
            ticker, assets, equity, liabs, hqla, nco, 
            bad_loans, loans, cds_proxy, volatility
        ])
        
        print(f"✔ Processed: {ticker:<5} | Assets: ${assets:>7.2f}B | CDS Score: {cds_proxy:>5.0f}")

    except Exception as e:
        print(f"❌ Error {ticker}: {e}")

# --- STEP 3: CREATE DATAFRAMES ---

cols = [
    'Bank', 'Total_Assets', 'Equity', 'Total_Liabilities', 
    'HQLA', 'Net_Outflows_30d', 'Bad_Loans', 'Total_Loans', 
    'Est_CDS_Spread', 'Stock_Volatility'
]

df_nodes = pd.DataFrame(bank_data, columns=cols).round(2)

# Sort by Assets
df_nodes = df_nodes.sort_values(by='Total_Assets', ascending=False).reset_index(drop=True)

# Generate Correlation Matrix (Log Returns)
# Refinement: Use Log Returns for cleaner correlation properties
price_log_returns = np.log(close_prices / close_prices.shift(1))
corr_matrix = price_log_returns.corr()

print("\n" + "="*50)
print("🏦 TOP 50 US BANKS DATASET GENERATED (REFINED)")
print("="*50)
print(df_nodes.head(10)) 

# --- STEP 4: VISUALIZE SYSTEMIC RISK (CORRELATION) ---
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.title("Systemic Risk Map: Log-Return Correlations")
plt.show()

# --- STEP 5: EXPORT TO CSV ---
df_nodes.to_csv("us_banks_top50_nodes.csv", index=False)
corr_matrix.to_csv("us_banks_asset_correlation.csv")

# --- STEP 6: ESTIMATE INTERBANK POSITIONS (REFINED) ---

# Global stats for scaling
max_log_asset = np.log(df_nodes['Total_Assets'].max())
min_log_asset = np.log(df_nodes['Total_Assets'].min())

def estimate_interbank_lending(row):
    # Log-Linear Scaling
    # Smallest bank -> ~2% of assets
    # Largest bank  -> ~12% of assets (JPM scale)
    log_a = np.log(row['Total_Assets'])
    scale_pos = (log_a - min_log_asset) / (max_log_asset - min_log_asset)
    
    ratio = 0.02 + (0.10 * scale_pos) # 2% to 12%
    return row['Total_Assets'] * ratio

# 1. Apply the estimation
df_nodes['Interbank_Assets'] = df_nodes.apply(estimate_interbank_lending, axis=1)

# 2. Estimate Borrowing (Liabilities)
# We assume borrowing follows a similar pattern to lending size
# But we must SCALE it later to make the system balance.
df_nodes['Interbank_Liabilities'] = df_nodes['Interbank_Assets'] * 0.9 

# 3. Balance the System (Crucial for Max Entropy)
# Total Money Lent MUST EQUAL Total Money Borrowed.
total_lent = df_nodes['Interbank_Assets'].sum()
total_borrowed = df_nodes['Interbank_Liabilities'].sum()
scaling_factor = total_lent / total_borrowed

df_nodes['Interbank_Liabilities'] = df_nodes['Interbank_Liabilities'] * scaling_factor

print(f"System Balanced: Total Interbank Market Size = ${total_lent:.2f} Billion")

# --- STEP 7: MAXIMUM ENTROPY ALGORITHM (RAS) ---
# This generates the 50x50 Matrix

print("\nRunning Maximum Entropy Optimization (RAS Algorithm)...")

a = df_nodes['Interbank_Assets'].values      # Row Targets
l = df_nodes['Interbank_Liabilities'].values # Col Targets
names = df_nodes['Bank'].values
n = len(names)

# A. Initial Guess (Outer Product)
# "Spread the money proportionally based on size"
matrix = np.outer(a, l) / total_lent

# B. Iterative Fitting (The Math Loop)
# Adjust rows and columns 100 times until they match the targets perfectly.
for i in range(100):
    # Adjust Rows (Lending)
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums==0] = 1 # Safety to avoid divide by zero
    matrix = matrix * (a / row_sums)[:, np.newaxis]
    
    # Adjust Columns (Borrowing)
    col_sums = matrix.sum(axis=0)
    col_sums[col_sums==0] = 1
    matrix = matrix * (l / col_sums)

# C. Zero out the Diagonal (Banks don't lend to themselves)
np.fill_diagonal(matrix, 0)

# D. Final Cleanup (One last balance check)
# Because setting diagonal to 0 messes up the sums slightly, we run a quick polish
for i in range(20):
    row_sums = matrix.sum(axis=1)
    matrix = matrix * (a / row_sums)[:, np.newaxis]
    col_sums = matrix.sum(axis=0)
    matrix = matrix * (l / col_sums)

# --- STEP 8: SAVE AND VISUALIZE ---

# Create the Matrix DataFrame
df_matrix = pd.DataFrame(matrix, index=names, columns=names).round(2)

print("\n" + "="*50)
print("🕸️ INTERBANK NETWORK GENERATED")
print("="*50)

# Show the new columns in the Node List
print("\nUpdated Node List (First 5 rows):")
print(df_nodes[['Bank', 'Total_Assets', 'Interbank_Assets', 'Interbank_Liabilities']].head())

# Show the Matrix (Who owes JPM?)
print("\nExposure Matrix (Top 5x5 slice):")
print(df_matrix.iloc[:5, :5])

# 1. Update the original CSV with the new columns
df_nodes.to_csv("us_banks_top50_nodes_final.csv", index=False)
print("✅ Saved node data to: us_banks_top50_nodes_final.csv")

# 2. Save the 50x50 Interaction Matrix
df_matrix.to_csv("us_banks_interbank_matrix.csv")
print("✅ Saved network matrix to: us_banks_interbank_matrix.csv")

# 3. Visualize the Network Density
plt.figure(figsize=(10, 8))
sns.heatmap(df_matrix, cmap="viridis", vmax=10) # Cap max color at $10B for visibility
plt.title("Interbank Lending Network (Estimated Flows in Billions)")
plt.xlabel("Borrower")
plt.ylabel("Lender")
plt.show()