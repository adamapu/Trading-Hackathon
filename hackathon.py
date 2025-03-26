import os
import subprocess
import sys
import requests
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'ensurepip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        __import__(package)

# Install missing libraries
install_and_import('requests')
install_and_import('numpy')
install_and_import('pandas')
install_and_import('scipy')

# CoinMarketCap API Setup
cmc_url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
cmc_parameters = {
    'start': '1',
    'limit': '10',
    'convert': 'USD'
}
cmc_headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': '8730b816-0710-4ed2-958b-514e1befe79e',
}

# Fetch Data from CoinMarketCap API
try:
    cmc_session = requests.Session()
    cmc_session.headers.update(cmc_headers)
    cmc_response = cmc_session.get(cmc_url, params=cmc_parameters)
    cmc_response.raise_for_status()
    cmc_data = json.loads(cmc_response.text)
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from CoinMarketCap API: {e}")
    sys.exit(1)

# Extract Relevant Data from CoinMarketCap
cmc_coins = cmc_data.get('data', [])
cmc_crypto_data = []
for coin in cmc_coins:
    cmc_crypto_data.append([coin['symbol'], coin['quote']['USD']['price'], coin['quote']['USD']['percent_change_24h']])

# Convert to DataFrame
cmc_df = pd.DataFrame(cmc_crypto_data, columns=['Symbol', 'Price', '24h Change'])

# Compute Financial Metrics
def compute_metrics(df):
    returns = df['24h Change'].dropna() / 100
    mean_return = returns.mean()
    std_dev = returns.std()
    risk_free_rate = 0.02 / 365  # Daily risk-free rate assumption
    
    # Sharpe Ratio
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Trade Frequency
    trade_frequency = (df['24h Change'].count() / len(df)) * 100
    
    return sharpe_ratio, max_drawdown, trade_frequency

sharpe_ratio_cmc, max_drawdown_cmc, trade_frequency_cmc = compute_metrics(cmc_df)

# Print Results
print("CoinMarketCap Data:")
print(cmc_df)

print("\nFinancial Metrics (CoinMarketCap):")
print(f"Sharpe Ratio: {sharpe_ratio_cmc:.2f}")
print(f"Maximum Drawdown: {max_drawdown_cmc:.2%}")
print(f"Trade Frequency: {trade_frequency_cmc:.2f}%")