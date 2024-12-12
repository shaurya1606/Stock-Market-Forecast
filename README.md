# Stock Market Forecasting Application

This repository contains the source code for the **Stock Market Forecasting Application**, a Streamlit-based web app that allows users to analyze, visualize, and forecast stock market data. It is designed to provide interactive visualizations, statistical analysis, and predictive modeling for selected stock tickers.

---

## Features

### 1. **Interactive User Interface**

- **Start and End Date Selection:** Users can choose custom date ranges for stock data analysis.
- **Stock Ticker Selection:** Dropdown to select from popular stock tickers like `AAPL`, `MSFT`, `GOOG`, etc.
- **Interactive Visualizations:** Line plots, summary statistics, and correlation heatmaps are available.

### 2. **Data Visualization**

- Multi-column data visualization with Plotly.
- Heatmaps for exploring relationships between key metrics.
- Time-series decomposition to analyze trends, seasonality, and residuals.

### 3. **Statistical and Predictive Modeling**

- Augmented Dickey-Fuller (ADF) test for stationarity check.
- Seasonal decomposition of time series data (trend, seasonality, residuals).
- SARIMAX model for forecasting stock prices with customizable parameters (`p`, `d`, `q`, and seasonal order).
- Predictions for future stock prices with visual comparisons between actual and predicted values.

### 4. **Customizable Parameters**

- Allows users to set values for the SARIMAX model’s hyperparameters (p, d, q, and seasonal order).
- Flexible forecasting period input.

### 5. **Responsive Design**

- Built-in sidebar for parameter inputs and feature toggles.
- Dynamic resizing of visualizations.

### 6. **About Section**

- Information about the developer.
- Links to connect on social media platforms like LinkedIn, GitHub, and Instagram.

---

## Prerequisites

To run this project locally, you need to have the following installed:

- Python 3.8+
- A web browser (for running the Streamlit app)

---

## Installation and Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/shauryasrivastava-1612/Stock-Market-Forecast.git
cd Stock-Market-Forecast
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

Start the Streamlit app by running:

```bash
streamlit run app.py
```

The app will open in your default web browser. If not, navigate to `http://localhost:8501`.

---

## Project Structure

```
Stock-Market-Forecast/
├── app.py                # Main application file
├── requirements.txt      # List of dependencies
├── .gitignore           # Ignored files and folders
└── venv/                # Virtual environment folder (not included in the repository)
```

---

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **yFinance**: For fetching historical stock market data.
- **Pandas and NumPy**: For data manipulation and analysis.
- **Matplotlib, Seaborn, Plotly**: For data visualization.
- **Statsmodels**: For time series analysis and forecasting.
- **Python Libraries**: Various utilities and dependencies listed in `requirements.txt`.

---

## Key Functionalities and Workflow

1. **Data Retrieval**:

   - Users select a company ticker and date range.
   - Stock data is fetched using `yfinance` and displayed in an interactive table.

2. **Visualization**:

   - Users can select specific columns to visualize using line charts.
   - Optional features like correlation heatmaps can be enabled.

3. **Time Series Analysis**:

   - ADF test checks for stationarity.
   - Seasonal decomposition splits data into trend, seasonal, and residual components.

4. **Predictive Modeling**:

   - Users customize SARIMAX model parameters.
   - Predictions are generated and compared with actual data.
   - Forecasting for user-specified future periods.

5. **Interactive Forecasting**:

   - Graphs show both actual and predicted values side by side.

---

## Planned Enhancements

- Add support for more stock tickers or indices.
- Enhance model selection with automated hyperparameter tuning.
- Provide more user-friendly error handling for invalid inputs.
- Include additional forecasting models (e.g., LSTM, Prophet).

---

## About the Author

**Shaurya Srivastava**

A tech enthusiast and aspiring professional in the field of High-Frequency Trading (HFT) and consultancy. Connect with me on:

- [LinkedIn](https://linkedin.com/)
- [GitHub](https://github.com/)
- [Instagram](https://instagram.com/)

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
