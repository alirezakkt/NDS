import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Function to fetch data for a given symbol and interval
def fetch_data(symbol, interval, period='1d'):
    df = yf.download(symbol, interval=interval, period=period)
    df.index = pd.to_datetime(df.index).tz_convert('Europe/Warsaw')  # Convert to 'Europe/Warsaw' timezone
    return df

# Fetch data for XAU/USD (Gold) in 5-minute intervals
xau_usd_5min = fetch_data("GC=F", interval="5m")

# Rename columns for clarity
xau_usd_5min.rename(columns={'Close': 'Close_5min', 'Open': 'Open_5min', 'High': 'High_5min', 'Low': 'Low_5min', 'Volume': 'Volume_5min'}, inplace=True)

# Node Detection (Using local maxima and minima)
xau_usd_5min['Min'] = xau_usd_5min['Close_5min'].iloc[argrelextrema(xau_usd_5min['Close_5min'].values, np.less_equal, order=5)[0]]
xau_usd_5min['Max'] = xau_usd_5min['Close_5min'].iloc[argrelextrema(xau_usd_5min['Close_5min'].values, np.greater_equal, order=5)[0]]

# Drop NaNs for easier processing of nodes
nodes = xau_usd_5min[['Min', 'Max']].dropna(how='all')

# Define Trend and Pullback Functions
def fit_polynomial(series, degree=2):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    return model, poly

# Fit trend (upward) and pullback (downward) functions between identified nodes
trends, pullbacks = [], []
node_indices = nodes.index

# Ensure that node_indices are sorted
node_indices = sorted(node_indices)

for i in range(1, len(node_indices)):
    start, end = node_indices[i-1], node_indices[i]
    segment = xau_usd_5min.loc[start:end, 'Close_5min']
    
    # Check if both start and end indices are valid
    if start in xau_usd_5min.index and end in xau_usd_5min.index:
        # Extract the Close prices as scalars
        start_price = xau_usd_5min.loc[start, 'Close_5min'].values[0]
        end_price = xau_usd_5min.loc[end, 'Close_5min'].values[0]
        
        # Determine if it's a trend or pullback
        if end_price < start_price:  # Downward (Pullback)
            model, poly = fit_polynomial(segment, degree=2)
            pullbacks.append((model, poly, segment.index))
        else:  # Upward (Trend)
            model, poly = fit_polynomial(segment, degree=2)
            trends.append((model, poly, segment.index))

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(xau_usd_5min.index, xau_usd_5min['Close_5min'], label='Price (5min)', color='black')

# Plotting the trends
for model, poly, indices in trends:
    X = np.arange(len(indices)).reshape(-1, 1)
    trend_values = model.predict(poly.transform(X))
    plt.plot(indices, trend_values, label='Trend', color='blue')

# Plotting the pullbacks
for model, poly, indices in pullbacks:
    X = np.arange(len(indices)).reshape(-1, 1)
    pullback_values = model.predict(poly.transform(X))
    plt.plot(indices, pullback_values, label='Pullback', color='red')

# Plotting the nodes
for idx in node_indices:
    plt.scatter(idx, xau_usd_5min.loc[idx, 'Close_5min'], color='green', zorder=5)


plt.title('Price with Trends, Pullbacks, and Nodes for XAU/USD (5min)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
