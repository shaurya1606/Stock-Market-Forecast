import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import plotly.express as px 
import datetime 
from datetime import date, timedelta 
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
import matplotlib as plt
 
# Define model save/load path
model_dir = "models"
model_filename = "forecasting_model.pkl"
model_path = os.path.join(model_dir, model_filename)

# Function to train the model
def train_model(data, selected_columns, p, d, q, seasonal_order):
    # Train a SARIMAX model
    model = sm.tsa.statespace.SARIMAX(data[selected_columns], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
    model = model.fit()
    
    # Save the trained model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, model_path)
    
    return model

# Function to load the previous model
def load_model():
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        return None

# Title and Header Section
app_name = 'Stock Market Forecasting Application'
st.title(app_name)
app_header = 'This app is created to forecast the stock market price of the selected company.'
st.subheader(app_header)

# Add an image
image_address = 'https://imgs.search.brave.com/jJAUX895aD7fVSe1Fc7AGWI6mTIHK3X4GzEWLYJUdiA/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90NC5m/dGNkbi5uZXQvanBn/LzA1LzY4Lzg1LzI3/LzM2MF9GXzU2ODg1/Mjc3Ml9QNEVtYktQ/OFlQYXJnenNQek5R/QXhVUjFKSUhGaE1s/dC5qcGc'
st.image(image_address)

# Sidebar Configuration
st.sidebar.header('Select the parameters from below')

# User inputs for start and end dates
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))

# Dropdown for selecting a stock ticker
ticker_list = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'META', 'TSLA', 'NVDA', 'ADBE', 'PYPL', 'INTC', 'CMCSA', 'NFLX', 'PEP']
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Add data as a column to the dataframe
st.write('Data for', ticker, 'from', start_date, 'to', end_date)
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.dataframe(data, width=1000, height=450)
# Visualization Section
st.header('Interactive Multi-Column Data Visualization')
st.subheader('Select the column to visualize')
st.write('**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column')

# Column Selection for Visualization
columns_available = [item[0] for item in data if item[0] != 'Date']
selected_columns = st.sidebar.selectbox('Select columns to visualize', columns_available)

# Plot the selected column
fig = px.line(data, x='Date', y=data[selected_columns].values.ravel(), title=f'{selected_columns} of the stock over time', width=1000 , height=600)
st.plotly_chart(fig)

# Subsetting the data
data = data[['Date', selected_columns]]
st.subheader('Selected column data')
st.dataframe(data, width=350, height=450)

# Optional Functionality: Show Summary Statistics
if st.sidebar.checkbox('Show Summary Statistics'):
    st.subheader(f'Summary Statistics for {selected_columns}')
    st.write(data[selected_columns].describe())

# Calculate 50-day, 100-day, and 150-day moving averages
data['50_day_MA'] = data[selected_columns].rolling(window=50).mean()
data['100_day_MA'] = data[selected_columns].rolling(window=100).mean()
data['150_day_MA'] = data[selected_columns].rolling(window=150).mean()
data['200_day_MA'] = data[selected_columns].rolling(window=200).mean()

# Plotting the stock data with moving averages
fig = go.Figure()

# Add actual stock price plot
fig.add_trace(go.Scatter(x=data['Date'], y=data[selected_columns].values.ravel(), mode='lines', name='Actual', line=dict(color='blue')))

# Add 50-day moving average plot
fig.add_trace(go.Scatter(x=data['Date'], y=data['50_day_MA'].values.ravel(), mode='lines', name='50-Day MA', line=dict(color='red', dash='dash')))

# Add 100-day moving average plot
fig.add_trace(go.Scatter(x=data['Date'], y=data['100_day_MA'].values.ravel(), mode='lines', name='100-Day MA', line=dict(color='orange', dash='dash')))

# Add 150-day moving average plot
fig.add_trace(go.Scatter(x=data['Date'], y=data['150_day_MA'].values.ravel(), mode='lines', name='150-Day MA', line=dict(color='purple', dash='dash')))

# Add 200-day moving average plot
fig.add_trace(go.Scatter(x=data['Date'], y=data['200_day_MA'].values.ravel(), mode='lines', name='200-Day MA', line=dict(color='green', dash='dash')))

# Customize the layout of the figure
fig.update_layout(
    title=f'{selected_columns} with 50-Day, 100-Day, 150-Day, and 200-Day Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price',
    width=1000,
    height=600,
    template="plotly_dark"  # Optional: Use dark theme for better visibility
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True, key="moving_average_plot")


# ADF test check stationarity
st.header('TS Data Stationary')
st.write('**Note:** If p-value is less than 0.05 then data is stationary')
st.write(adfuller(data[selected_columns])[1] < 0.5)

# Decomposition Section
st.header('Decomposition of the Data')
decomposition = seasonal_decompose(data[selected_columns], model='additive', period=12)
st.write(decomposition.plot())

# Plotting the decomposition in Plotly
st.write('## Plotting the decomposition in plotly')
st.plotly_chart(px.line(x=data['Date'], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'), key="trend_plot")
st.plotly_chart(px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonality', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'), key="seasonality_plot")
st.plotly_chart(px.line(x=data['Date'], y=decomposition.resid, title='Residuals', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'), key="residual_plot")

# Model Section: Forecasting

# Check if a model exists
model = load_model()
# Model Section: SARIMAX or ARIMA Forecasting

# User input for model selection
model_option = st.sidebar.selectbox('Select the forecasting model', ['SARIMAX', 'ARIMA'])

if model_option == 'SARIMAX':
    st.subheader('SARIMAX Model Forecasting')
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)
    seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 4)

    # Create and fit SARIMAX model
    model = sm.tsa.statespace.SARIMAX(data[selected_columns], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
    model = model.fit()

    # Forecasting with SARIMAX
    forecast_period = st.number_input("## Enter forecast period in days for SARIMAX", value=10, min_value=1, key="forecast_period_sarimax")

    # Check if the forecast period exceeds the available data range
    available_data_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    if forecast_period > available_data_days:
        st.warning(f"The forecast period ({forecast_period} days) exceeds the available data range ({available_data_days} days).")

    # Predict all values for the forecast period
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period - 1)
    predictions = predictions.predicted_mean

    # Add index to results dataframe as dates
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, 'Date', predictions.index)
    predictions.reset_index(drop=True, inplace=True)
    st.write('## Predictions', predictions)

    # Plot Actual vs Predicted Data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[selected_columns].values.ravel(), mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'].values.ravel(), mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig, key="sarimax_forecast_plot")

elif model_option == 'ARIMA':
    st.subheader('ARIMA Model Forecasting')
    p_arima = st.slider('Select the value of p for ARIMA', 0, 5, 2)
    d_arima = st.slider('Select the value of d for ARIMA', 0, 5, 1)
    q_arima = st.slider('Select the value of q for ARIMA', 0, 5, 2)

    # ARIMA model
    arima_model = ARIMA(data[selected_columns], order=(p_arima, d_arima, q_arima))
    arima_model_fit = arima_model.fit()

    # Forecasting with ARIMA
    forecast_period = st.number_input("## Enter forecast period in days for ARIMA", value=10, min_value=1, key="forecast_period_arima")

    # Forecasting with ARIMA
    arima_predictions = arima_model_fit.forecast(steps=forecast_period)
    forecast_mean = arima_predictions

    # Create a date range for the forecasted data
    forecast_dates = pd.date_range(start=end_date, periods=forecast_period, freq='D')

    # Create a DataFrame for the forecasted data
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_mean})

    # Display forecasted data
    st.write(f"ARIMA model forecast for the next {forecast_period} days:")
    st.write(forecast_df)

    # Plot the forecasted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[selected_columns].values.ravel(), mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig, key="arima_forecast_plot")

# Option to use existing model or train a new one
model_option = st.sidebar.selectbox('Select an option', ['Use Previous Model', 'Train New Model'])

if model_option == 'Train New Model':
    # User input for SARIMAX parameters
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)
    seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 4)

    # Train a new model
    model = train_model(data, selected_columns, p, d, q, seasonal_order)
    st.write("New model trained and saved!")

elif model_option == 'Use Previous Model' and model is not None:
    st.write("## Using the previously trained model.")

else:
    st.write("No trained model found. Please train a new model.")

# Forecasting with SARIMAX
if model:
    forecast_period = st.number_input("## Enter forecast period in days", value=10, min_value=1)

    # Forecasting with SARIMAX
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period - 1)
    predictions = predictions.predicted_mean

    # Add index to results dataframe as dates
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, 'Date', predictions.index)
    predictions.reset_index(drop=True, inplace=True)
    st.write('## Predictions', predictions)

    # Plot Actual vs Predicted Data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[selected_columns].values.ravel(), mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'].values.ravel(), mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig)

# About the Author and Social Media Links Section
st.write('---')
st.write('# About the Author:')

# Author name
st.write('<p style="color:white; font-size: 30px;">Shaurya Srivastava</br></p>', unsafe_allow_html=True)

st.write('# About the Contributer:')

st.write('<p style="color:white; font-size: 30px;">Vedansh Gupta, Sheetal, Nitya Umrao, Suhani Sharma</p>', unsafe_allow_html=True)

# Social media links
st.write('## Connect with me on social media')

# URLs for social media icons
linkedin_url = 'https://img.icons8.com/color/48/000000/linkedin.png'
github_url = 'https://img.icons8.com/fluent/48/000000/github.png'
instagram_url = 'https://img.icons8.com/fluent/48/000000/instagram-new.png'

# Redirect URLs for social media
linkedin_redirect_url = "/blank"
github_redirect_url = "/blank"
instagram_redirect_url = "/blank"

# Display social media links
st.markdown(f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width ="60" height = "60"></a>', unsafe_allow_html=True)
st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}" width ="60" height = "60"></a>', unsafe_allow_html=True)
st.markdown(f'<a href="{instagram_redirect_url}"><img src="{instagram_url}" width ="60" height = "60"></a>', unsafe_allow_html=True)
