import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestRegressor
import matplotlib.dates as mdates
import time
import threading
import joblib
import os

# Global variables for data
historical_data = pd.DataFrame(columns=['Close'])
lock = threading.Lock()
rf_model = None
model_filename = 'rf_model.pkl'
data_filename = 'historical_data.csv'
max_candles = 100
fetch_interval = 60

# Telegram bot configuration
TELEGRAM_TOKEN = '7588980852:AAF1iW_4tcU4K-MsbCMklXXFVLYucK0DAho'
CHAT_ID = '839653783'

# Function to send notification to Telegram
def send_notification(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Error sending notification: {e}")

# Function to fetch the last 10 candles from the Binance API
def fetch_candles(symbol="BTCUSDT", interval="5m", limit=10):
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 
            'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
        ])
        df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
        df.set_index('OpenTime', inplace=True)
        df = df.astype(float)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching candles: {e}")
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# Function to save historical data to a CSV file
def save_historical_data(data, filename=data_filename):
    try:
        data.to_csv(filename)
        print(f"Historical data saved to {filename}.")
    except Exception as e:
        print(f"Error saving historical data: {e}")

# Function to load historical data from a CSV file
def load_historical_data(filename=data_filename):
    try:
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Historical data loaded from {filename}.")
            return data
        else:
            print(f"No historical data found at {filename}, starting fresh.")
            return pd.DataFrame(columns=['Close'])
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return pd.DataFrame(columns=['Close'])

# Function to continuously fetch data
def data_fetcher():
    global historical_data
    while True:
        candles = fetch_candles()
        if not candles.empty and not candles.isnull().all().any():
            with lock:
                historical_data = pd.concat([historical_data, candles[['Close']]], ignore_index=False)
                historical_data = historical_data[~historical_data.index.duplicated(keep='last')]
                if len(historical_data) > max_candles:
                    historical_data = historical_data.tail(max_candles)
                save_historical_data(historical_data)
        print("Latest fetched data:")
        print(historical_data.tail(1))
        time.sleep(fetch_interval)

# Function to save the trained model
def save_model(model, filename=model_filename):
    try:
        joblib.dump(model, filename)
        #print(f"Model saved to {filename}.")
    except Exception as e:
        print(f"Error saving model: {e}")

# Function to load the trained model
def load_model(filename=model_filename):
    try:
        if os.path.exists(filename):
            model = joblib.load(filename)
            print(f"Model loaded from {filename}.")
            return model
        else:
            print(f"No model found at {filename}, starting fresh.")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load historical data at the start of the script
historical_data = load_historical_data()
rf_model = load_model()

# Initialize the plot
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

# Start the data fetching thread
fetch_thread = threading.Thread(target=data_fetcher)
fetch_thread.daemon = True
fetch_thread.start()

# Initialize variables for valid node notifications
last_n_node_price = None
last_s_node_price = None
last_notification = None  # Track the last notification sent

# Main loop for visualization and training
while True:
    with lock:
        if not historical_data.empty:
            if len(historical_data) >= 10:
                # Node Detection
                historical_data['Min'] = historical_data['Close'].iloc[argrelextrema(historical_data['Close'].values, np.less_equal, order=5)[0]]
                historical_data['Max'] = historical_data['Close'].iloc[argrelextrema(historical_data['Close'].values, np.greater_equal, order=5)[0]]

                nodes = historical_data[['Min', 'Max']].dropna(how='all')

                # Prepare data for Random Forest
                historical_data['Time'] = np.arange(len(historical_data))
                X = historical_data[['Time']]
                y = historical_data['Close'].values

                if rf_model is None or len(historical_data) > 10:
                    #print("Training the model...")
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X, y.ravel())
                    #print(f"Model trained with {len(historical_data)} data points.")
                    save_model(rf_model)

                ax.clear()
                ax.plot(historical_data.index, historical_data['Close'], label='Real Price (5min)', color='black')

                if rf_model is not None:
                    historical_data['RF_Prediction'] = rf_model.predict(X)
                    ax.plot(historical_data.index, historical_data['RF_Prediction'], label='Random Forest Prediction', color='blue')

                # Detecting valid nodes for notifications
                if not nodes.empty:
                    if 'Max' in nodes.columns and not nodes['Max'].isnull().all():
                        last_n_node_price = nodes['Max'].iloc[-1]

                    if 'Min' in nodes.columns and not nodes['Min'].isnull().all():
                        last_s_node_price = nodes['Min'].iloc[-1]

                    # Notify for S node after an N node
                    if last_n_node_price is not None and last_s_node_price is not None:
                        if last_n_node_price < last_s_node_price and last_notification != 'S':
                            send_notification(f"S-node-Sell: price: {last_s_node_price:.2f}")
                            last_notification = 'S'  # Update the last notification sent
                            
                    # Notify for N node after an S node
                    if last_s_node_price is not None and last_n_node_price is not None:
                        if last_s_node_price < last_n_node_price and last_notification != 'N':
                            send_notification(f"N-node-Buy: price: {last_n_node_price:.2f}")
                            last_notification = 'N'  # Update the last notification sent

                # Plotting nodes
                ax.scatter(nodes.index, nodes['Min'], color='green', marker='^', label='S Nodes')
                ax.scatter(nodes.index, nodes['Max'], color='red', marker='v', label='N Nodes')

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax.set_title('BTC/USDT Price Prediction and Node Detection')
                ax.set_xlabel('Time (Tehran Time)')
                ax.set_ylabel('Price (USDT)')
                ax.legend()
                plt.xticks(rotation=45)
                plt.grid()

                plt.pause(0.1)

        else:
            print("Historical data is empty, unable to plot.")
    
    time.sleep(1)
