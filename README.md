# Nifty50 200-Day EMA Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlitapp.com)

## Overview

The 200-day Exponential Moving Average (EMA) is a key technical indicator used by traders and investors to analyze long-term price trends. It helps smooth out price data by creating a single flowing line, making it easier to identify the direction of the trend. The Nifty50 index, which represents the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange (NSE), is often analyzed using EMAs to make informed investment decisions. This project utilizes three different predictive models to forecast the future values of the Nifty50 200-day EMA.

## Repository Structure

- NiftyEMA.ipynb    - Jupyter notebook used for initial testing, data exploration, and model intuition.

- test.py           - Python script for the Monte Carlo simulation model.

- testLR.py         - Python script for the Linear Regression model.

- testarima.py      - Python script for the ARIMA model.

- main.py           - Streamlit application for interactive EMA predictions.

- requirements.txt  - List of packages required to run the Streamlit app.

## Models
- **Linear Regression**: Predicts future stock prices based on past data trends and calculates the 200-day EMA.
- **ARIMA (AutoRegressive Integrated Moving Average)**: Utilizes statistical analysis to forecast future price movements and compute the EMA.
- **Monte Carlo Simulation**: Generates a range of possible outcomes based on defined upper and lower limits to predict the EMA.

## Getting Started
1. **Clone the repository**

`git clone https://github.com/yourusername/Nifty50-EMA-Prediction.git`

2. **Navigate to the repository**

`cd Nifty50-EMA-Prediction`

3. **Install dependencies**

`pip install -r requirements.txt`

4. **Run the Streamlit application**

`streamlit run main.py`

## Usage
After launching the Streamlit application, you will see an interface where you can:
- Select the prediction model from a dropdown (Linear Regression, ARIMA, or Monte Carlo Simulation).
- Specify the number of future days for which you want to predict the 200-day EMA.
- (For Monte Carlo Simulation) Input the expected upper and lower limits of Nifty50.

The application will then display the predicted 200-day EMA on a Plotly graph, along with the predicted value on the last day of the forecast.


