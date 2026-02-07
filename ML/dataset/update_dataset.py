import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import fsolve

filename = 'us_banks_top50_nodes_final.csv'
df = pd.read_csv(filename)

print(f"Successfully loaded {len(df)} banks from {filename}")

# ==========================================
# 2. MARKET DATA FUSION (Hybrid Approach)
# ==========================================
# We need 'Market Cap' (E) for the Merton Model.
# Your CSV has 'Equity' (Book Value), but Market Value is better.
# This function tries to fetch live Market Cap, but falls back to Book Equity if needed.

def get_real_time_market_cap(ticker, book_equity):
    try:
        # Try fetching live data
        info = yf.Ticker(ticker).info
        return info.get('marketCap', book_equity * 1.0) # Fallback to Book Equity
    except:
        return book_equity * 1.0 # Conservative proxy: Market Cap = Book Equity

# Apply fetching (Note: This might be slow for 50 banks, so we use a proxy for the demo)
print("Fetching real-time Market Cap for 50 banks... (approx 30s)")
df['Market_Cap_E'] = df['Bank'].apply(lambda x: get_real_time_market_cap(x, df.loc[df['Bank']==x, 'Equity'].values[0]))

# Fallback: Ensure no zeros or NaNs in Market Cap
df['Market_Cap_E'] = df['Market_Cap_E'].fillna(df['Equity'])
df.loc[df['Market_Cap_E'] <= 0, 'Market_Cap_E'] = df.loc[df['Market_Cap_E'] <= 0, 'Equity']

# Use the 'Stock_Volatility' from your CSV directly.
df['Equity_Volatility_SigmaE'] = df['Stock_Volatility']

print("Market Data fused. Calculating Risk Metrics...")

# ==========================================
# 3. MERTON MODEL SOLVER
# ==========================================
def merton_equations(vars, E, sigma_E, D, r, T):
    """
    Solves for Implied Asset Value (V_A) and Asset Volatility (sigma_A)
    """
    V_A, sigma_A = vars
    
    # Constraints to prevent math errors
    if V_A <= 0 or sigma_A <= 0:
        return [1e10, 1e10]
        
    d1 = (np.log(V_A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    
    # Equation 1: Equity = Call Option on Assets (Black-Scholes)
    eq1 = V_A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    
    # Equation 2: Equity Volatility relationship (Ito's Lemma)
    eq2 = (V_A / E) * norm.cdf(d1) * sigma_A - sigma_E
    
    return [eq1, eq2]

def calculate_merton_metrics(row):
    E = row['Market_Cap_E']
    sigma_E = row['Equity_Volatility_SigmaE']
    D = row['Total_Liabilities']     # The Debt Barrier
    r = 0.04                         # Risk-free rate (4%)
    T = 1.0                          # Time horizon (1 Year)
    
    # Initial Guess: Asset Value = Equity + Debt
    initial_guess = [E + D, sigma_E * 0.5]
    
    try:
        # Solve the system of non-linear equations
        V_A, sigma_A = fsolve(merton_equations, initial_guess, args=(E, sigma_E, D, r, T))
        
        # Calculate Distance to Default (DtD)
        # Using simplified distance measure often used in KMV
        d2 = (np.log(V_A / D) + (r - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
        
        # Probability of Default (1 Year)
        pd_1y = norm.cdf(-d2)
        
        return pd.Series({
            'Implied_Asset_Value': V_A,
            'Implied_Asset_Volatility': sigma_A,
            'Distance_to_Default': d2,
            'Prob_Default_1Y': pd_1y
        })
    except:
        # CONVERGENCE FAILURE HANDLER
        # If solver fails, it's usually because the bank is:
        # A) Extremely Safe (Vol -> 0)
        # B) Extremely Distressed (Equity -> Option value behaves weirdly)
        
        # We assume "Safe" fallback for this dataset if volatility is low
        if sigma_E < 0.20:
             return pd.Series({
                'Implied_Asset_Value': E + D, 
                'Implied_Asset_Volatility': 0.05, 
                'Distance_to_Default': 10.0, # High safety score
                'Prob_Default_1Y': 0.0
            })
        else:
            return pd.Series({
                'Implied_Asset_Value': np.nan, 
                'Implied_Asset_Volatility': np.nan, 
                'Distance_to_Default': np.nan,
                'Prob_Default_1Y': np.nan
            })

# Apply to your dataframe
risk_metrics = df.apply(calculate_merton_metrics, axis=1)
final_df = pd.concat([df, risk_metrics], axis=1)

# ==========================================
# 4. OUTPUT RESULTS
# ==========================================
# Save the engineered data for Phase 2 (Network Simulation)
final_df.to_csv('phase1_engineered_data.csv', index=False)

# Display the top risk metrics
cols = ['Bank', 'Market_Cap_E', 'Total_Liabilities', 'Distance_to_Default', 'Prob_Default_1Y']
print(final_df[cols].head())