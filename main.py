import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import plotly.express as px

# Define function to get stock data
def get_stock_data(ticker, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Linear Regression model function
def linear_regression_model(days_future):
    df = pd.DataFrame(get_stock_data("^NSEI", 299))
    df.drop(['Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], axis=1, inplace=True)
    df.index = pd.to_datetime(df.index)
    df['Days'] = (df.index - df.index.min()).days
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_dates = [df.index[-1] + timedelta(days=x) for x in range(1, days_future+1)]
    future_days = [(date - df.index.min()).days for date in future_dates]
    future_df = pd.DataFrame(data={'Days': future_days}, index=future_dates)
    future_df['Close'] = model.predict(future_df[['Days']])
    df = pd.concat([df, future_df])
    df['200_day_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

# ARIMA model function
def arima_model(days_future):
    df = pd.DataFrame(get_stock_data("^NSEI", 299))
    df.drop(['Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], axis=1, inplace=True)
    df['Close'] = df['Close'].astype(float)
    model = sm.tsa.statespace.SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=0)
    forecast = results.get_forecast(steps=days_future)
    forecast_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days_future)
    forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Close'])
    df = pd.concat([df, forecast_df])
    df['200_day_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

# Monte Carlo simulation function
def monte_carlo_simulation(days_future, upper_limit, lower_limit):
    df = pd.DataFrame(get_stock_data("^NSEI", 299))
    df.drop(['Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], axis=1, inplace=True)
    np.random.seed(0)
    future_prices = np.linspace(lower_limit, upper_limit, days_future)
    future_dates = [df.index[-1] + timedelta(days=x) for x in range(1, days_future+1)]
    future_df = pd.DataFrame(data=future_prices, index=future_dates, columns=['Close'])
    df = pd.concat([df, future_df])
    df['200_day_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

# Streamlit user interface
st.title('Nifty50 200-Day EMA Prediction📈')
model_choice = st.selectbox('Choose a prediction model:', ['Linear Regression', 'ARIMA', 'Monte Carlo Simulation'])
days_future = st.number_input('Enter the number of days for future prediction:', min_value=1, max_value=365, value=30)

if model_choice == 'Monte Carlo Simulation':
    upper_limit = st.number_input('Enter the upper limit of Nifty50:', min_value=20000, max_value=50000, value=25000)
    lower_limit = st.number_input('Enter the lower limit of Nifty50:', min_value=20000, max_value=50000, value=24000)
    if st.button('Predict'):
        result = monte_carlo_simulation(days_future, upper_limit, lower_limit)
        fig = px.line(result, x=result.index, y='200_day_EMA', title='Monte Carlo Simulation: 200-Day EMA Prediction')
        st.plotly_chart(fig)
        st.write(f"The predicted 200-day EMA on the last day is: {result['200_day_EMA'].iloc[-1]:.2f}")
elif model_choice == 'Linear Regression':
    if st.button('Predict'):
        result = linear_regression_model(days_future)
        fig = px.line(result, x=result.index, y='200_day_EMA', title='Linear Regression: 200-Day EMA Prediction')
        st.plotly_chart(fig)
        st.write(f"The predicted 200-day EMA on the last day is: {result['200_day_EMA'].iloc[-1]:.2f}")
elif model_choice == 'ARIMA':
    if st.button('Predict'):
        result = arima_model(days_future)
        fig = px.line(result, x=result.index, y='200_day_EMA', title='ARIMA Model: 200-Day EMA Prediction')
        st.plotly_chart(fig)
        st.write(f"The predicted 200-day EMA on the last day is: {result['200_day_EMA'].iloc[-1]:.2f}")