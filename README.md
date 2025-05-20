# Stock-Market-Forecasting-using-Time-Series-Models
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# 1. Data Collection
def fetch_data(ticker="AAPL", start="2015-01-01", end="2023-12-31"):
    data = yf.download(ticker, start=start, end=end)
    data = data[["Close"]]
    data.dropna(inplace=True)
    data = data.asfreq('B')  # Business days
    data.fillna(method='ffill', inplace=True)
    return data

# 2. Visualization
def plot_data(data):
    plt.figure(figsize=(14, 5))
    plt.plot(data, label="Close Price")
    plt.title("Stock Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

#3. ARIMA Model
def arima_model(data):
    train = data[:-100]
    test = data[-100:]
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=100)
    rmse = sqrt(mean_squared_error(test, forecast))
    print(f"ARIMA RMSE: {rmse:.2f}")
    plt.figure(figsize=(14, 5))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(test.index, forecast, label="Forecast")
    plt.title("ARIMA Forecast")
    plt.legend()
    plt.show()

# 4. Prophet Model
def prophet_model(data):
    df = data.reset_index()
    df.columns = ['ds', 'y']
    train = df[:-100]
    test = df[-100:]
    model = Prophet(daily_seasonality=True)
    model.fit(train)
    future = model.make_future_dataframe(periods=100)
    forecast = model.predict(future)
    pred = forecast[['ds', 'yhat']].iloc[-100:]
    rmse = sqrt(mean_squared_error(test['y'].values, pred['yhat'].values))
    print(f"Prophet RMSE: {rmse:.2f}")
    model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.show()

# 5. LSTM Model
def lstm_model(data):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict last 100 points
    test_data = scaled_data[-160:]
    X_test = []
    for i in range(60, len(test_data)):
        X_test.append(test_data[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = data[-100:].values

    rmse = sqrt(mean_squared_error(actual, predictions))
    print(f"LSTM RMSE: {rmse:.2f}")
    plt.figure(figsize=(14, 5))
    plt.plot(actual, label="Actual")
    plt.plot(predictions, label="LSTM Forecast")
    plt.legend()
    plt.title("LSTM Forecast")
    plt.show()

# Main Pipeline
def main():
    data = fetch_data("AAPL")
    plot_data(data)
    print("\n--- ARIMA ---")
    arima_model(data)
    print("\n--- Prophet ---")
    prophet_model(data)
    print("\n--- LSTM ---")
    lstm_model(data)

if __name__ == "__main__":
    main()
