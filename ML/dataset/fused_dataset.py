import pandas as pd
import numpy as np

# 1. Load Data
stocks = pd.read_csv('./stocks_data_long.csv')
banks = pd.read_csv('./phase1_engineered_data.csv')

# Convert Date
stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks = stocks.sort_values(['Ticker', 'Date'])

# 2. Market Factor Construction (The "Systemic" Signal)
# We use the 1000 stocks to create a "Market Index"
market_stats = stocks.groupby('Date')['Close'].agg(['mean', 'std'])
market_stats.columns = ['Market_Index', 'Market_Dispersion']
market_stats['Market_Return'] = market_stats['Market_Index'].pct_change()
# Calculate 30-day Rolling Volatility of the Market
market_stats['Market_Vol_30d'] = market_stats['Market_Return'].rolling(30).std() * np.sqrt(252)

# 3. Stock Feature Engineering (The "Fast" Data)
# Calculate returns and volatility for each stock
stocks['Log_Return'] = np.log(stocks['Close'] / stocks['Close'].shift(1))
# Rolling 30-day volatility per ticker
stocks['Volatility_30d'] = stocks.groupby('Ticker')['Log_Return'].transform(lambda x: x.rolling(30).std() * np.sqrt(252))

# RSI Calculation (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

stocks['RSI'] = stocks.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))

# 4. Filter for the Banks
# We focus on the intersection of available stock data and bank balance sheets
common_tickers = np.intersect1d(stocks['Ticker'].unique(), banks['Bank'].unique())
bank_stocks = stocks[stocks['Ticker'].isin(common_tickers)].copy()

# 5. Balance Sheet Feature Engineering (The "Slow" Data)
# Convert absolute numbers into comparable Ratios
banks['Leverage_Ratio'] = banks['Total_Assets'] / banks['Equity'].replace(0, np.nan)
banks['Liquidity_Ratio'] = banks['HQLA'] / banks['Total_Assets'].replace(0, np.nan)
banks['NPL_Ratio'] = banks['Bad_Loans'] / banks['Total_Loans'].replace(0, 1) # Handle potential div by zero
banks['Interbank_Ratio'] = (banks['Interbank_Assets'] + banks['Interbank_Liabilities']) / banks['Total_Assets']

# 6. The Fusion (Merge)
# Broadcast static bank data to the daily stock data
merged_df = pd.merge(bank_stocks, banks, left_on='Ticker', right_on='Bank', how='left')
# Merge Systemic Market Factors
merged_df = pd.merge(merged_df, market_stats, on='Date', how='left')

# 7. Clean Up
# Avoid inplace to prevent SettingWithCopy warnings and ensure assignment
# Use `bfill()` because some pandas versions don't accept `method` in `fillna`
merged_df = merged_df.bfill()  # Fill initial rolling NAs
merged_df = merged_df.fillna(0)  # Safety fallback

# Final Feature Selection
final_df = merged_df[[
    'Date', 'Ticker', 'Close', 'Log_Return', 'Volatility_30d', 'RSI', 
    'Market_Return', 'Market_Vol_30d', 'Market_Dispersion', 
    'Leverage_Ratio', 'Liquidity_Ratio', 'NPL_Ratio', 'Interbank_Ratio',
    'Distance_to_Default', 'Implied_Asset_Volatility', 'Est_CDS_Spread'
]]

final_df.to_csv('phase2_merged_data.csv', index=False)
print("Data Fusion Complete: 'phase2_merged_data.csv' created.")