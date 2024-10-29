import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def get_stock_data(ticker, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Fetch historical data
nifty_50_data = get_stock_data("^NSEI", 299)
df = pd.DataFrame(nifty_50_data)
df.drop(['Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], axis=1, inplace=True)
df.index = pd.to_datetime(df.index)
df['Days'] = (df.index - df.index.min()).days  # convert dates to integer for regression

# Linear Regression Model
X = df[['Days']]  # feature
y = df['Close']  # target
model = LinearRegression()
model.fit(X, y)

# Predict future values
n = int(input("Enter the number of days in future for which you want to predict the EMA: "))
future_dates = [df.index[-1] + timedelta(days=x) for x in range(1, n+1)]
future_days = [(date - df.index.min()).days for date in future_dates]  # convert future dates to integer
future_df = pd.DataFrame(data={'Days': future_days}, index=future_dates)
future_df['Close'] = model.predict(future_df[['Days']])

# Append future data to original and calculate EMA
df = pd.concat([df, future_df])
df['200_day_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()

# Print result
print("The EMA after the next", n, "days is: ", df['200_day_EMA'].iloc[-1])