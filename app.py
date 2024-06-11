import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Function to get cryptocurrency data from Yahoo Finance
def get_crypto_data(coin_id, start_date, end_date):
    ticker_symbol = f"{coin_id.upper()}-USD"
    crypto_data = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)
    if crypto_data.empty:
        raise ValueError(f"No data found for {ticker_symbol} in the given date range.")
    return crypto_data

# Function to perform ARIMA forecasting
def arima_forecast(data, forecast_periods):
    close_prices = data['Close'].values
    model = auto_arima(close_prices, seasonal=True, m=12, trace=True, suppress_warnings=True)
    model.fit(close_prices)
    forecast = model.predict(n_periods=forecast_periods)
    aic = model.aic()
    return forecast, aic

# Function to perform SARIMA forecasting
def sarima_forecast(data, forecast_periods):
    close_prices = data['Close']
    model = SARIMAX(close_prices, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    forecast = results.predict(start=len(data), end=len(data) + forecast_periods - 1, dynamic=False)
    aic = results.aic
    return forecast, aic

# Function to perform LSTM forecasting
def lstm_forecast(data, forecast_periods):
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare the data
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:training_data_len, :]
    test_data = scaled_data[training_data_len - 60:, :]

    X_train = []
    y_train = []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dense(units=50))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Prepare the test data
    X_test = []
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predictions
    predictions = model.predict(X_test)
    predictions_2d = np.zeros((predictions.shape[0], scaled_data.shape[1]))
    predictions_2d[:, 0] = predictions.flatten()
    predictions = scaler.inverse_transform(predictions_2d)[:, 0]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(close_prices[training_data_len:], predictions[:len(close_prices) - training_data_len]))

    return predictions[-forecast_periods:], rmse

# Function to evaluate and select the best model
def evaluate_models(data, forecast_periods):
    arima_forecast_values, arima_aic = arima_forecast(data, forecast_periods)
    sarima_forecast_values, sarima_aic = sarima_forecast(data, forecast_periods)
    lstm_forecast_values, lstm_rmse = lstm_forecast(data, forecast_periods)
    
    arima_rmse = np.sqrt(mean_squared_error(data['Close'][-forecast_periods:], arima_forecast_values))
    sarima_rmse = np.sqrt(mean_squared_error(data['Close'][-forecast_periods:], sarima_forecast_values))

    models = {
        "ARIMA": {"aic": arima_aic, "rmse": arima_rmse, "forecast": arima_forecast_values},
        "SARIMA": {"aic": sarima_aic, "rmse": sarima_rmse, "forecast": sarima_forecast_values},
        "LSTM": {"aic": np.nan, "rmse": lstm_rmse, "forecast": lstm_forecast_values}
    }

    # Select the best model based on AIC for ARIMA and SARIMA, and RMSE for all models
    best_model = min(models.items(), key=lambda x: (x[1]['aic'], x[1]['rmse']) if not np.isnan(x[1]['aic']) else (np.inf, x[1]['rmse']))
    
    return best_model

# Main function to run the Streamlit app
def main():
    # Set title and sidebar options
    st.title("Cryptocurrency Price Prediction")
    st.sidebar.title("Options")

    # Select cryptocurrency
    coin_id = st.sidebar.selectbox("Select cryptocurrency", ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "SHIB", "ADA", "MATIC", "AVAX", "DOT", "LINK",
                                                             "TRX", "BCH", "NEAR", "MATIC", "UNI", "LTC", "PEPE", "ICP", "LEO", "RNDR", "HBAR", "ATOM", 
                                                             "WIF", "ARB", "XLM", "CRO", "XMR", "FLOKI", "OKB", "VET", "MKR", "INJ", "FTM", "BONK", 
                                                             "ONDO", "ALGO", "ENA", "GALA"])

    # Select start and end dates
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    # Get cryptocurrency data
    crypto_data = get_crypto_data(coin_id, start_date, end_date)

    # Display cryptocurrency data
    st.write("Cryptocurrency Data:")
    st.write(crypto_data)

    # Choose forecasting method
    method = st.sidebar.selectbox("Choose forecasting method", ["ARIMA", "SARIMA", "LSTM", "Auto Select Best Model"])

    # Select forecast periods
    forecast_periods = st.sidebar.number_input("Forecast Periods", min_value=1, max_value=365, value=30, step=1)

    # Perform forecasting based on selected method
    if method == "ARIMA":
        forecast, _ = arima_forecast(crypto_data, forecast_periods)
    elif method == "SARIMA":
        forecast, _ = sarima_forecast(crypto_data, forecast_periods)
    elif method == "LSTM":
        forecast, _ = lstm_forecast(crypto_data, forecast_periods)
    elif method == "Auto Select Best Model":
        best_model = evaluate_models(crypto_data, forecast_periods)
        st.write(f"Best Model: {best_model[0]} (AIC: {best_model[1]['aic']:.3f}, RMSE: {best_model[1]['rmse']:.3f})")
        forecast = best_model[1]['forecast']

    # Display forecasted prices
    st.write("Forecasted Prices:")
    st.write(forecast)

    # Plot time series data
    st.write("Cryptocurrency Price Prediction Chart:")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crypto_data.index, y=crypto_data['Close'], mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(x=pd.date_range(start=end_date, periods=forecast_periods), y=forecast, mode='lines', name='Forecasted Data'))
    fig.update_layout(title='Cryptocurrency Price Prediction', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Run the app
if __name__ == "__main__":
    main()
