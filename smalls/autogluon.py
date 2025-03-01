import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import requests
import io

# GitHub repository raw URL
GITHUB_RAW_URL = "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data/"

# List of available datasets
DATASETS = {
    "Hospitality Employees": "HospitalityEmployees.csv",
    "Logan Housing": "Logan_housing.csv",
    "US GDP": "US_gdp.csv",
    "US Macro": "US_macro.csv",
    "US Macro Quarterly": "US_macro_Quarterly.csv",
    "US Macro Monthly": "US_macro_monthly.csv",
    "Airline Passengers": "airline_passengers.csv",
    "Employee Data": "employee.csv",
    "Stock Market (Yahoo Finance)": "yfinance.csv"
}

def load_default_data(filename):
    """Fetch data from GitHub and return a DataFrame."""
    url = GITHUB_RAW_URL + filename
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    else:
        st.error(f"Failed to load {filename}. Check the URL or internet connection.")
        return None

# Streamlit App Title
st.title("📈 AutoGluon Time Series Forecasting App")

# Sidebar: Choose Dataset or Upload File
st.sidebar.header("📂 Choose a Dataset")

# Dropdown for selecting a dataset
dataset_choice = st.sidebar.selectbox("Select a default dataset", list(DATASETS.keys()))

# File uploader for custom datasets
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type="csv")

# Load dataset
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🔄 **Custom Data Uploaded!**")
elif dataset_choice:
    df = load_default_data(DATASETS[dataset_choice])
    st.write(f"📊 **Loaded Default Dataset:** {dataset_choice}")

# Ensure data is loaded
if df is not None:
    st.write("### Dataset Preview", df.head())

    # User selects timestamp column
    timestamp_col = st.selectbox("🕒 Select the Timestamp Column", df.columns)

    # Auto-detect if dataset is wide (like stock data)
    if dataset_choice == "Stock Market (Yahoo Finance)" or df.iloc[:, 1:].applymap(lambda x: isinstance(x, (int, float))).all().all():
        st.write("🔄 **Wide dataset detected! Reshaping to long format...**")
        df = df.melt(id_vars=[timestamp_col], var_name="item_id", value_name="target")
        st.write("✅ Reshaped Data Preview", df.head())

    # User selects item_id (or create one if missing)
    id_options = ["Create New ID"] + list(df.columns)
    item_id_col = st.selectbox("🔢 Select Item ID Column", id_options)

    # Assign unique ID if the dataset doesn't have one
    if item_id_col == "Create New ID":
        df["item_id"] = "series_1"
        item_id_col = "item_id"

    # User selects target column
    target_col = st.selectbox("🎯 Select the Target Column", df.columns)

    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Convert to AutoGluon TimeSeriesDataFrame
    ts_data = TimeSeriesDataFrame.from_data_frame(df, id_column=item_id_col, timestamp_column=timestamp_col)

    # User Input: Prediction Length
    prediction_length = st.slider("📅 Select Forecast Horizon (steps)", min_value=1, max_value=60, value=30)

    # Train Model
    if st.button("🚀 Train Model"):
        predictor = TimeSeriesPredictor(target=target_col, prediction_length=prediction_length)
        predictor.fit(ts_data)
        predictor.save("ts_model")
        st.success("🎯 Model Training Complete!")

    # Make Predictions
    if st.button("📈 Make Predictions"):
        predictor = TimeSeriesPredictor.load("ts_model")
        forecasts = predictor.predict(ts_data)

        st.write("📊 **Forecast Results:**", forecasts.head())

        # Plot Forecasts
        fig, ax = plt.subplots(figsize=(10, 5))
        df.groupby(timestamp_col)[target_col].mean().plot(ax=ax, label="Actual")
        forecasts.groupby("timestamp")["mean"].mean().plot(ax=ax, label="Forecast", linestyle="--")
        ax.legend()
        st.pyplot(fig)
