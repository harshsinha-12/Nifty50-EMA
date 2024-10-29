import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm

def get_stock_data(ticker, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Get data
nifty_50_data = get_stock_data("^NSEI", 299)
df = pd.DataFrame(nifty_50_data)
df.drop(['Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], axis=1, inplace=True)
df.index = pd.to_datetime(df.index)

# ARIMA Model
model = sm.tsa.arima.ARIMA(df['Close'], order=(1, 1, 1))
results = model.fit()

# Forecast future prices
n = int(input("Enter the number of days in future for which you want to predict the EMA: "))
forecast = results.get_forecast(steps=n)
forecast_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=n)  # Adjusted here
forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Close'])

# Append forecast to original data and calculate EMA
df = pd.concat([df, forecast_df])
df['200_day_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()

print("The EMA after the next", n, "days is: ", df['200_day_EMA'].iloc[-1])