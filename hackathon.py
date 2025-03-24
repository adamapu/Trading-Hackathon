import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler

# import requests
# import json
# import pandas as pd
# from datetime import datetime, timedelta

# # CoinAPI Setup
# coinapi_url = "https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history"
# coinapi_headers = {
#     "Accepts": "application/json",
#     "X-CoinAPI-Key": "e16c2a13-94a8-4fcb-b4bc-67e6e8f72b33",  # Replace with your actual API key
# }

# # Define time range (last 24 hours)
# end_time = datetime.utcnow()
# start_time = end_time - timedelta(days=1)

# # API Parameters
# parameters = {
#     "period_id": "1MIN",  # 1-minute interval data
#     "time_start": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
#     "time_end": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
#     "limit": 1440,  # 1440 minutes in 24 hours
# }

# # Fetch Data
# try:
#     response = requests.get(coinapi_url, headers=coinapi_headers, params=parameters)
#     response.raise_for_status()
#     btc_data = response.json()
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching BTC data from CoinAPI: {e}")
#     exit(1)

# # Process Data
# btc_prices = []
# for entry in btc_data:
#     btc_prices.append([
#         entry["time_period_start"],
#         entry["price_open"],
#         entry["price_high"],
#         entry["price_low"],
#         entry["price_close"],
#         entry["volume_traded"]
#     ])

# # Convert to DataFrame
# df = pd.DataFrame(btc_prices, columns=["Time", "Open", "High", "Low", "Close", "Volume"])

# # Save to CSV
# csv_file = "BTC_1Min_24H.csv"
# df.to_csv(csv_file, index=False, encoding="utf-8")

# print(f"✅ Bitcoin 1-minute data for the last 24 hours saved to {csv_file}")

# Load Data
df = pd.read_csv("BTC_1Min_24H.csv")

# Feature Selection
features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Apply HMM for Market Regime Detection
hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)
hmm_model.fit(features_scaled)
df['State'] = hmm_model.predict(features_scaled)

# Prepare Data for CNN Training
sequence_length = 60  # Using past 60 minutes to predict next close price
X, y = [], []
for i in range(len(df) - sequence_length):
    X.append(features_scaled[i:i+sequence_length])
    y.append(features_scaled[i+sequence_length, 3])  # Predict Close price
X, y = np.array(X), np.array(y)

# Build CNN Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train CNN Model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Generate Predictions
predictions = model.predict(X)
df = df.iloc[sequence_length:]
df['Predicted_Close'] = scaler.inverse_transform(features_scaled)[sequence_length:, 3]

# Generate Trading Signals
df['Signal'] = np.where(df['Predicted_Close'] > df['Close'], 1, -1)

# Calculate Trade Frequency
trade_count = df[df['Signal'] != 0].shape[0]
total_rows = df.shape[0]
trade_frequency = (trade_count / total_rows) * 100

# Ensure Minimum Trade Frequency of 3%
if trade_frequency < 3:
    print(f"⚠️ Trade Frequency: {trade_frequency:.2f}% (Below 3%, Adjusting Threshold)")
    df['Signal'] = np.where((df['Predicted_Close'] - df['Close']).abs() > 0.001, df['Signal'], 0)
    trade_count = df[df['Signal'] != 0].shape[0]
    trade_frequency = (trade_count / total_rows) * 100

print(f"✅ Trade Frequency: {trade_frequency:.2f}%")

# Calculate Sharpe Ratio & Max Drawdown
returns = df['Close'].pct_change()
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
cumulative_returns = (1 + returns).cumprod()
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Save Results
df.to_csv("BTC_Trading_Signals.csv", index=False)
print("✅ Trading signals saved to BTC_Trading_Signals.csv")