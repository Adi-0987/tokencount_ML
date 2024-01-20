import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product

# Function to load data and perform forecasting
def load_data():
    # Load your dataset (replace 'your_data.csv' with your actual file)
    df = pd.read_csv('Facility_Token_Generation_Trend.csv')

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    return df

# Function to find optimal ARIMA parameters
def find_optimal_params(data):
    p = d = q = range(0, 3)  # You can adjust the range based on your data
    pdq = list(product(p, d, q))

    # Find the best parameters using AIC (Akaike Information Criterion)
    aic_values = []
    for param in pdq:
        try:
            model = ARIMA(data, order=param)
            results = model.fit()
            aic_values.append((param, results.aic))
        except:
            continue

    # Select the parameters with the minimum AIC
    best_params = min(aic_values, key=lambda x: x[1])[0]
    return best_params

# Function to predict token count
def predict_token_count(df, selected_month, selected_year, state_name):
    # Filter data for the selected month and year
    selected_data = df[(df['Date'].dt.month == selected_month) & (df['Date'].dt.year == selected_year)]

    # Group by state and sum token count
    state_token_count = selected_data[selected_data['State Name'] == state_name].groupby('Date')['Token Count'].sum()

    # Find optimal ARIMA parameters
    optimal_params = find_optimal_params(state_token_count)

    # ARIMA model
    model = ARIMA(state_token_count, order=optimal_params)
    results = model.fit()

    # Predict future values
    future_dates = pd.date_range(start=selected_data['Date'].max(), periods=12, freq='M')
    forecast = results.get_forecast(steps=len(future_dates))
    forecast_index = forecast.predicted_mean.index

    # Plotting the forecast
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Create a subplot with 1 row and 2 columns
    # Actual values
    axs[0].bar(state_token_count.index, state_token_count.values, label='Actual', alpha=0.7)
    axs[0].set_title(f'Token Count for {state_name} - Actual')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Token Count')
    axs[0].legend()
    axs[0].tick_params(rotation=45)

    # Predicted values
    axs[1].bar(forecast_index, forecast.predicted_mean.values, label='Predicted', alpha=0.7, color='red')
    axs[1].set_title(f'Token Count for {state_name} - Predicted')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Token Count')
    axs[1].legend()
    axs[1].tick_params(rotation=45)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plots using st.pyplot()
    st.pyplot(fig)

    # Calculate percentage increase or decrease and model confidence
    actual_values = state_token_count.values
    predicted_values = forecast.predicted_mean.values  # Use forecast_values instead of forecast.predicted_mean

    percentage_change = ((predicted_values[-1] - actual_values[-1]) / actual_values[-1]) * 100
    confidence_interval = forecast.conf_int(alpha=0.05)

    st.write(f"Percentage Change: {percentage_change:.2f}%")
    st.write(f"Model Confidence Interval: {confidence_interval.iloc[-1, 0]:.2f} to {confidence_interval.iloc[-1, 1]:.2f}")

# Main Streamlit app
def main():
    st.title('Predictive Analysis of Token Counts')

    # Load data
    df = load_data()

    # Display dataset
    st.header('Dataset')
    st.write(df)

    # Prediction model
    st.header('Prediction Model')

    # Date selection
    selected_month = st.slider('Select Month', 1, 12)
    selected_year = st.slider('Select Year', df['Date'].dt.year.min(), df['Date'].dt.year.max())

    # State selection
    states = df['State Name'].unique()
    selected_state = st.selectbox('Select State', states)

    # Perform forecasting and display results
    predict_token_count(df, selected_month, selected_year, selected_state)

if __name__ == "__main__":
    main()
