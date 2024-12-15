# Import libraries
import streamlit as st 
import yfinance as yf 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go 
import plotly.express as px 
import datetime 
from datetime import date, timedelta 
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller

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

# Optional Functionality: Correlation Heatmap
if st.sidebar.checkbox('Show Correlation Heatmap'):
    st.subheader('Correlation Heatmap of Stock Data')
    corr = data[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']].corr()
    fig_heatmap = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Correlation Heatmap')
    st.plotly_chart(fig_heatmap)

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
st.plotly_chart(px.line(x=data['Date'], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonality', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data['Date'], y=decomposition.resid, title='Residuals', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))

# Model Section: SARIMAX Forecasting
st.header('Model Forecasting')

# User input for three parameters of the model and seasonal order
p = st.slider('Select the value of p', 0, 5, 1)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 1)
seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 2)

# Create and fit SARIMAX model
model = sm.tsa.statespace.SARIMAX(data[selected_columns], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

# Model Summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")

# Forecasting Section: Predict future values
st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the Data</p>", unsafe_allow_html=True)
forecast_period = st.number_input("## Enter forecast period in days", value=10)

# Predict all values for the forecast period and the current dataset
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions = predictions.predicted_mean

# Add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write('## Predictions', predictions)
st.write('## Actual Data', data)
st.write('---')

# Plotting Actual vs Predicted Data
fig = go.Figure()

# Add actual data to the plot
fig.add_trace(
    go.Scatter(
        x=data['Date'],  # X-axis: Dates from the dataset
        y=data[selected_columns].values.ravel(),  # Flatten Y-axis values
        mode='lines',
        name='Actual',
        line=dict(color='blue'),
    )
)

# Add predicted data to the plot
fig.add_trace(
    go.Scatter(
        x=predictions['Date'],  # X-axis: Prediction dates
        y=predictions['predicted_mean'].values.ravel(),  # Flatten predicted values
        mode='lines',
        name='Predicted',
        line=dict(color='red'),
    )
)

# Set the title and axis labels
fig.update_layout(
    title='Actual vs Predicted',
    xaxis_title='Date',
    yaxis_title='Price',
    width=1000,
    height=600,
)

# Display the plot
st.plotly_chart(fig)

# About the Author and Social Media Links Section
st.write('---')
st.write('### About the Author:')

# Author name
st.write('<p style="color:white; font-weight: bold; font-size: 50px;">Mr. Shaurya Srivastava</p>', unsafe_allow_html=True)

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
st.markdown(f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width ="60" height = "60"></a>'
            f'<a href="{instagram_redirect_url}"><img src="{instagram_url}" width ="60" height = "60"></a>'
            f'<a href="{github_redirect_url}"><img src="{github_url}" width ="60" height = "60"></a>', unsafe_allow_html=True)
