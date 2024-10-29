import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(ticker, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Get the previous 200-day price of Nifty 50
nifty_50_data = get_stock_data("^NSEI", 299)
df = pd.DataFrame(nifty_50_data)

df.drop(['Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df['200_day_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
df.index = pd.to_datetime(df.index)
n = int(input("Enter the number of days in future for which you want to predict the EMA: "))
u = int(input("Upper limit range of Nifty 50: "))
l = int(input("Lower limit range of Nifty 50: "))
np.random.seed(0)
future_prices = np.linspace(l, u, n)
future_dates = [df.index[-1] + timedelta(days=x) for x in range(1, n+1)]  

future_df = pd.DataFrame(data=future_prices, index=future_dates, columns=['Close'])

df = pd.concat([df, future_df])

df.reset_index(drop=True, inplace=True) 
df = df.iloc[n:]
df['200_day_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
print("The EMA after the next", n, "days is: ", df['200_day_EMA'].iloc[-1], "for Nifty 50 between range of", l, "and", u)