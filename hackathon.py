#1) just data import from CoinMarketCap and detect anomaly
# import os
# import subprocess
# import sys

# # Ensure required libraries are installed
# def install_and_import(package):
#     try:
#         __import__(package)
#     except ImportError:
#         subprocess.check_call([sys.executable, '-m', 'ensurepip'])
#         subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
#         __import__(package)

# # Install missing libraries
# install_and_import('requests')
# install_and_import('numpy')
# install_and_import('pandas')
# install_and_import('sklearn')  # Note the change here from 'scikit-learn' to 'sklearn'

# # Imports
# import requests
# import json
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest

# # CoinMarketCap API Setup
# url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
# parameters = {
#     'start': '1',
#     'limit': '10',  # Fetching more data for better analysis
#     'convert': 'USD'
# }
# headers = {
#     'Accepts': 'application/json',
#     'X-CMC_PRO_API_KEY': '8730b816-0710-4ed2-958b-514e1befe79e',
# }

# # Fetch Data from API
# try:
#     session = requests.Session()
#     session.headers.update(headers)
#     response = session.get(url, params=parameters)
#     response.raise_for_status()  # Raise an error for bad HTTP responses
#     data = json.loads(response.text)
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching data from CoinMarketCap API: {e}")
#     sys.exit(1)

# # Extract Relevant Data
# coins = data.get('data', [])
# crypto_data = []
# for coin in coins:
#     crypto_data.append([coin['symbol'], coin['quote']['USD']['price'], coin['quote']['USD']['percent_change_24h']])

# # Convert to DataFrame
# df = pd.DataFrame(crypto_data, columns=['Symbol', 'Price', '24h Change'])

# # Machine Learning Model (Isolation Forest for Anomaly Detection)
# X = df[['24h Change']]
# isolation_forest = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
# df['Anomaly'] = isolation_forest.fit_predict(X)

# # Print Results
# print(df)
# print("\nPotential Market Inefficiencies Detected:")
# print(df[df['Anomaly'] == -1])  # Anomalies detected



#2) Show the data that does not meet the criteria
# import os
# import subprocess
# import sys
# import requests
# import json
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest

# # CoinMarketCap API Setup
# url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
# parameters = {
#     'start': '1',
#     'limit': '50',  # Fetch more data for better statistical analysis
#     'convert': 'USD'
# }
# headers = {
#     'Accepts': 'application/json',
#     'X-CMC_PRO_API_KEY': '8730b816-0710-4ed2-958b-514e1befe79e',
# }

# # Fetch Data from API
# try:
#     session = requests.Session()
#     session.headers.update(headers)
#     response = session.get(url, params=parameters)
#     response.raise_for_status()
#     data = json.loads(response.text)
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching data from CoinMarketCap API: {e}")
#     sys.exit(1)

# # Extract Relevant Data
# coins = data.get('data', [])
# crypto_data = []
# for coin in coins:
#     crypto_data.append([
#         coin['symbol'],
#         coin['quote']['USD']['price'],
#         coin['quote']['USD']['percent_change_24h']
#     ])

# # Convert to DataFrame
# df = pd.DataFrame(crypto_data, columns=['Symbol', 'Price', '24h Change'])

# # Compute Log Returns
# df['Log Return'] = np.log(df['Price'] / df['Price'].shift(1))

# # Calculate Sharpe Ratio
# risk_free_rate = 0  # Assuming zero risk-free rate
# df['SR'] = df['Log Return'].mean() / df['Log Return'].std()

# # Maximum Drawdown Calculation
# df['Cumulative Return'] = (1 + df['Log Return']).cumprod()
# df['Peak'] = df['Cumulative Return'].cummax()
# df['Drawdown'] = df['Cumulative Return'] / df['Peak'] - 1
# df['MDD'] = df['Drawdown'].min()

# # Machine Learning Model (Isolation Forest for Anomaly Detection)
# X = df[['24h Change']]
# isolation_forest = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
# df['Anomaly'] = isolation_forest.fit_predict(X)

# # Trade Frequency Criteria
# trade_count = (df['Anomaly'] == -1).sum()
# trade_frequency = trade_count / len(df)

# # Filter based on success criteria
# df['Meets Criteria'] = (df['SR'] >= 1.8) & (df['MDD'] >= -0.4) & (trade_frequency >= 0.03)

# # Print Results
# print(df)
# print("\n✅ Potential Market Inefficiencies That Meet Success Criteria:")
# print(df[df['Meets Criteria']])




#3) 
# import os
# import subprocess
# import sys
# import requests
# import json
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest

# # CoinMarketCap API Setup
# CMC_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
# CMC_HEADERS = {
#     'Accepts': 'application/json',
#     'X-CMC_PRO_API_KEY': '8730b816-0710-4ed2-958b-514e1befe79e',
# }
# CMC_PARAMETERS = {
#     'start': '1',
#     'limit': '50',  # Fetching top 50 cryptos
#     'convert': 'USD'
# }

# # CoinAPI.io Setup
# COINAPI_URL = 'https://rest.coinapi.io/v1/assets'
# COINAPI_HEADERS = {
#     'Accepts': 'application/json',
#     'X-CoinAPI-Key': 'd13fcbf4-28a8-4a0c-a9f5-6695af970832',
# }

# # Fetch Data from APIs
# def fetch_data(url, headers, parameters=None, source='Unknown'):
#     try:
#         session = requests.Session()
#         session.headers.update(headers)
#         response = session.get(url, params=parameters) if parameters else session.get(url)
#         response.raise_for_status()
#         data = json.loads(response.text)
#         return data, source
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data from {source}: {e}")
#         return None, source

# # Fetch and process CoinMarketCap data
# cmc_data, cmc_source = fetch_data(CMC_URL, CMC_HEADERS, CMC_PARAMETERS, 'CoinMarketCap')
# cmc_crypto_data = []
# if cmc_data:
#     for coin in cmc_data.get('data', []):
#         cmc_crypto_data.append([
#             coin['symbol'], coin['quote']['USD']['price'], coin['quote']['USD']['percent_change_24h'], cmc_source
#         ])

# # Fetch and process CoinAPI.io data
# coinapi_data, coinapi_source = fetch_data(COINAPI_URL, COINAPI_HEADERS, source='CoinAPI.io')
# coinapi_crypto_data = []
# if coinapi_data:
#     for asset in coinapi_data:
#         if 'price_usd' in asset and 'asset_id' in asset:
#             coinapi_crypto_data.append([
#                 asset['asset_id'], asset['price_usd'], None, coinapi_source  # No percent change available
#             ])

# # Convert to DataFrame and Merge
# cmc_df = pd.DataFrame(cmc_crypto_data, columns=['Symbol', 'Price', '24h Change', 'Source'])
# coinapi_df = pd.DataFrame(coinapi_crypto_data, columns=['Symbol', 'Price', '24h Change', 'Source'])

# # Combine both data sources
# crypto_df = pd.concat([cmc_df, coinapi_df], ignore_index=True)

# # Machine Learning Model (Isolation Forest for Anomaly Detection)
# X = crypto_df[['Price']].dropna()
# isolation_forest = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
# crypto_df.loc[X.index, 'Anomaly'] = isolation_forest.fit_predict(X)

# # Calculate Sharpe Ratio, Maximum Drawdown, and Trade Frequency
# def calculate_metrics(df):
#     returns = df['Price'].pct_change().dropna()
#     sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
#     max_drawdown = (df['Price'].cummax() - df['Price']).max() / df['Price'].cummax().max()
#     trade_frequency = len(returns) / len(df) * 100
#     return sharpe_ratio, max_drawdown, trade_frequency

# sharpe_ratio, max_drawdown, trade_frequency = calculate_metrics(crypto_df)

# # Determine if criteria are met
# criteria = {
#     'Sharpe Ratio': sharpe_ratio >= 1.8,
#     'Maximum Drawdown': max_drawdown >= -0.40,
#     'Trade Frequency': trade_frequency >= 3.0
# }

# # Print Results
# print("Top 50 Cryptos from Both Sources:")
# print(crypto_df.head(50))
# print(f"\nSharpe Ratio: {sharpe_ratio:.2f} - {'Pass' if criteria['Sharpe Ratio'] else 'Fail'}")
# print(f"Maximum Drawdown: {max_drawdown:.2%} - {'Pass' if criteria['Maximum Drawdown'] else 'Fail'}")
# print(f"Trade Frequency: {trade_frequency:.2f}% - {'Pass' if criteria['Trade Frequency'] else 'Fail'}")



#4) Using open ai
# import os
# import subprocess
# import sys
# import requests
# import json
# import numpy as np
# import pandas as pd
# from scipy.stats import norm

# # OpenAI API Setup
# openai_api_key = "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"
# openai_endpoint = "https://api.openai.com/v1/chat/completions"

# def install_and_import(package):
#     try:
#         __import__(package)
#     except ImportError:
#         subprocess.check_call([sys.executable, '-m', 'ensurepip'])
#         subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
#         __import__(package)

# # Install missing libraries
# install_and_import('requests')
# install_and_import('numpy')
# install_and_import('pandas')
# install_and_import('scipy')

# # CoinMarketCap API Setup
# cmc_url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
# cmc_parameters = {
#     'start': '1',
#     'limit': '10',
#     'convert': 'USD'
# }
# cmc_headers = {
#     'Accepts': 'application/json',
#     'X-CMC_PRO_API_KEY': '8730b816-0710-4ed2-958b-514e1befe79e',
# }

# # CoinAPI Setup
# coinapi_url = 'https://rest.coinapi.io/v1/assets'
# coinapi_headers = {
#     'Accepts': 'application/json',
#     'X-CoinAPI-Key': 'd13fcbf4-28a8-4a0c-a9f5-6695af970832',
# }

# # Fetch Data from CoinMarketCap API
# try:
#     cmc_session = requests.Session()
#     cmc_session.headers.update(cmc_headers)
#     cmc_response = cmc_session.get(cmc_url, params=cmc_parameters)
#     cmc_response.raise_for_status()
#     cmc_data = json.loads(cmc_response.text)
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching data from CoinMarketCap API: {e}")
#     sys.exit(1)

# # Fetch Data from CoinAPI
# try:
#     coinapi_session = requests.Session()
#     coinapi_session.headers.update(coinapi_headers)
#     coinapi_response = coinapi_session.get(coinapi_url)
#     coinapi_response.raise_for_status()
#     coinapi_data = json.loads(coinapi_response.text)
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching data from CoinAPI: {e}")
#     sys.exit(1)

# # Extract Relevant Data from CoinMarketCap
# cmc_coins = cmc_data.get('data', [])
# cmc_crypto_data = []
# for coin in cmc_coins:
#     cmc_crypto_data.append([coin['symbol'], coin['quote']['USD']['price'], coin['quote']['USD']['percent_change_24h']])

# # Extract Relevant Data from CoinAPI
# coinapi_crypto_data = []
# for asset in coinapi_data:
#     if asset['type_is_crypto'] == 1:
#         coinapi_crypto_data.append([asset['asset_id'], asset.get('price_usd', None), asset.get('percent_change_24h', None)])

# # Convert to DataFrames
# cmc_df = pd.DataFrame(cmc_crypto_data, columns=['Symbol', 'Price', '24h Change'])
# coinapi_df = pd.DataFrame(coinapi_crypto_data, columns=['Symbol', 'Price', '24h Change'])

# # Compute Financial Metrics
# def compute_metrics(df):
#     returns = df['24h Change'].dropna() / 100
#     mean_return = returns.mean()
#     std_dev = returns.std()
#     risk_free_rate = 0.02 / 365  # Daily risk-free rate assumption
    
#     # Sharpe Ratio
#     sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan
    
#     # Maximum Drawdown
#     cumulative_returns = (1 + returns).cumprod()
#     rolling_max = cumulative_returns.cummax()
#     drawdown = (cumulative_returns - rolling_max) / rolling_max
#     max_drawdown = drawdown.min()
    
#     # Trade Frequency
#     trade_frequency = (df['24h Change'].count() / len(df)) * 100
    
#     return sharpe_ratio, max_drawdown, trade_frequency

# sharpe_ratio_cmc, max_drawdown_cmc, trade_frequency_cmc = compute_metrics(cmc_df)
# sharpe_ratio_coinapi, max_drawdown_coinapi, trade_frequency_coinapi = compute_metrics(coinapi_df)

# # Send Data to OpenAI for Analysis
# def analyze_with_openai(data):
#     headers = {
#         "Authorization": f"Bearer {openai_api_key}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "gpt-4o-mini",
#         "messages": [
#             {"role": "system", "content": "You are a financial analyst AI."},
#             {"role": "user", "content": f"Analyze the following financial metrics: {data}"}
#         ]
#     }
#     response = requests.post(openai_endpoint, headers=headers, json=payload)
    
#     print("\nOpenAI Response Status Code:", response.status_code)
#     if response.status_code != 200 or not response.text.strip():
#         print("Error: Invalid response from OpenAI API")
#         return {"error": "Invalid response from OpenAI API"}
    
#     try:
#         response_json = response.json()
#         # Extract the summary from the response
#         summary = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No summary provided.")
#         print("\nOpenAI Analysis Summary:")
#         print(summary)
#         return summary
#     except json.JSONDecodeError:
#         print("Error: Failed to decode JSON response")
#         return {"error": "Failed to decode JSON response"}

# # Call the function and print the summary
# openai_response = analyze_with_openai({
#     "CoinMarketCap": {
#         "Sharpe Ratio": sharpe_ratio_cmc,
#         "Maximum Drawdown": max_drawdown_cmc,
#         "Trade Frequency": trade_frequency_cmc
#     },
#     "CoinAPI": {
#         "Sharpe Ratio": sharpe_ratio_coinapi,
#         "Maximum Drawdown": max_drawdown_coinapi,
#         "Trade Frequency": trade_frequency_coinapi
#     }
# })

# # Print Results
# print("CoinMarketCap Data:")
# print(cmc_df)
# print("\nCoinAPI Data:")
# print(coinapi_df)

# print("\nFinancial Metrics (CoinMarketCap):")
# print(f"Sharpe Ratio: {sharpe_ratio_cmc:.2f}")
# print(f"Maximum Drawdown: {max_drawdown_cmc:.2%}")
# print(f"Trade Frequency: {trade_frequency_cmc:.2f}%")

# print("\nFinancial Metrics (CoinAPI):")
# print(f"Sharpe Ratio: {sharpe_ratio_coinapi:.2f}")
# print(f"Maximum Drawdown: {max_drawdown_coinapi:.2%}")
# print(f"Trade Frequency: {trade_frequency_coinapi:.2f}%")

# print("\nOpenAI Analysis:")
# print(openai_response)



#5)
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

 