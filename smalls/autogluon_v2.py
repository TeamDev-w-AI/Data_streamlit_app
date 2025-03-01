import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import requests
import io
import re
from typing import Tuple, List, Dict, Any

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
        return response.text
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
if uploaded_file:
    # Read directly with pandas but save the raw content for potential preprocessing
    try:
        df = pd.read_csv(uploaded_file)
        st.write("🔄 **Custom Data Uploaded!**")
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        st.stop()
elif dataset_choice:
    csv_content = load_default_data(DATASETS[dataset_choice])
    if csv_content:
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            st.write(f"📊 **Loaded Default Dataset:** {dataset_choice}")
        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")
            st.stop()
    else:
        st.error("Failed to load dataset")
        st.stop()
else:
    st.info("Please select a dataset or upload a file")
    st.stop()

# Ensure data is loaded
if df is not None:
    st.write("### Dataset Preview", df.head())
    
    # Data type inspection
    with st.expander("Data Types Information"):
        st.write(df.dtypes)
    
    # Column selection (with explicit type conversion)
    st.header("📊 Time Series Configuration")
    
    # Timestamp column selection
    st.subheader("Step 1: Select the Timestamp Column")
    
    # Try to identify potential date columns
    date_cols = []
    for col in df.columns:
        # Look for date-related column names
        if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day']):
            date_cols.append(col)
    
    # If no date columns found by name, try to detect by content
    if not date_cols:
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if contents look like dates
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', str(sample)):
                    date_cols.append(col)
    
    # Select timestamp column with date columns prioritized
    all_cols = date_cols + [col for col in df.columns if col not in date_cols]
    timestamp_col = st.selectbox(
        "Select the timestamp column:", 
        all_cols,
        index=0 if date_cols else 0,
        help="Column containing date/time information"
    )
    
    # Convert timestamp to datetime with error handling and display format
    try:
        # First try to infer format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        st.success(f"✅ Converted '{timestamp_col}' to datetime format")
        
        # Show the first few values after conversion
        st.write("Sample timestamp values after conversion:")
        st.write(df[timestamp_col].head())
    except Exception as e:
        st.warning(f"⚠️ Automatic datetime conversion failed: {str(e)}")
        
        # Offer manual format specification
        date_format = st.text_input(
            "Enter date format (e.g., '%Y-%m-%d', '%m/%d/%Y'):",
            value="%Y-%m-%d" if re.search(r'\d{4}-\d{2}-\d{2}', str(df[timestamp_col].iloc[0])) else "%m/%d/%Y",
            help="Specify the date format pattern"
        )
        
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_format)
            st.success(f"✅ Converted '{timestamp_col}' to datetime with format '{date_format}'")
            
            # Show the first few values after conversion
            st.write("Sample timestamp values after conversion:")
            st.write(df[timestamp_col].head())
        except Exception as e2:
            st.error(f"⚠️ Failed to convert timestamp column: {str(e2)}")
            st.info("Please try a different column or check your data format")
            st.stop()
    
    # Data format detection and handling
    st.subheader("Step 2: Data Format Detection")
    
    # Check if this is wide format data (multiple numeric columns)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != timestamp_col]
    
    if len(numeric_cols) > 1:
        st.info("📊 Multiple numeric columns detected. This may be wide-format data.")
        
        data_format = st.radio(
            "Select data format:",
            ["Wide format (multiple series as columns)", "Long format (already in correct format)"],
            index=0 if "Stock Market" in dataset_choice or len(numeric_cols) > 3 else 1,
            help="Wide format has multiple time series as separate columns. Long format has one row per timestamp and series."
        )
        
        if data_format == "Wide format (multiple series as columns)":
            # Let user select which columns to include in the conversion
            st.write("Select columns to include as separate time series:")
            
            # Default selection - limit to first 5 numeric columns if there are many
            default_selection = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
            
            selected_cols = st.multiselect(
                "Series columns:", 
                numeric_cols,
                default=default_selection
            )
            
            if not selected_cols:
                st.warning("Please select at least one column to use as a time series")
                st.stop()
            
            # Convert wide to long format
            df_long = df.melt(
                id_vars=[timestamp_col],
                value_vars=selected_cols,
                var_name="item_id",
                value_name="target"
            )
            
            # Preview the melted data
            st.write("### Preview of reshaped data (first few rows):")
            st.dataframe(df_long.head())
            
            # Replace the original dataframe with the long format version
            df = df_long
    
    # Step 3: Setup TimeSeriesDataFrame
    st.subheader("Step 3: Final Column Configuration")
    
    # Determine item_id column
    if "item_id" in df.columns:
        item_id_col = "item_id"
        st.success(f"✅ Using existing 'item_id' column")
    else:
        # User selects item_id column or creates a new one
        id_options = ["Create New ID"] + [col for col in df.columns if col != timestamp_col]
        item_id_col = st.selectbox(
            "Select or create Item ID column:", 
            id_options,
            index=0,
            help="Column that identifies different time series"
        )
        
        if item_id_col == "Create New ID":
            df["item_id"] = "series_1"
            item_id_col = "item_id"
            st.info("✅ Created a new 'item_id' column with default value 'series_1'")
    
    # Determine target column
    if "target" in df.columns:
        target_col = "target"
        st.success(f"✅ Using existing 'target' column")
    else:
        # Find potential target columns (numeric and not used for timestamp or item_id)
        potential_targets = [
            col for col in df.select_dtypes(include=['number']).columns 
            if col != timestamp_col and col != item_id_col
        ]
        
        if not potential_targets:
            potential_targets = [
                col for col in df.columns 
                if col != timestamp_col and col != item_id_col
            ]
        
        if not potential_targets:
            st.error("No suitable columns available for target")
            st.stop()
        
        target_col = st.selectbox(
            "Select Target column:", 
            potential_targets,
            index=0,
            help="Column containing the values to forecast"
        )
    
    # Create copies of selected columns with standardized names
    df_copy = df.copy()
    
    # Display column mapping
    st.write("### Column Mapping")
    st.write(f"- Timestamp: '{timestamp_col}'")
    st.write(f"- Item ID: '{item_id_col}'")
    st.write(f"- Target: '{target_col}'")
    
    # Ensure we have the right datatypes
    try:
        df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce')
        df_copy.dropna(subset=[target_col], inplace=True)
        st.success(f"✅ Converted '{target_col}' to numeric format")
    except Exception as e:
        st.error(f"⚠️ Error converting target column to numeric: {str(e)}")
        st.stop()
    
    # Debug information
    with st.expander("Debugging Information"):
        st.write("DataFrame info:")
        buffer = io.StringIO()
        df_copy.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("DataFrame columns:")
        st.write(df_copy.columns.tolist())
        
        st.write("DataFrame sample:")
        st.dataframe(df_copy.head())
    
    # Create TimeSeriesDataFrame
    try:
        # Create a new DataFrame with renamed columns to ensure consistency
        ts_df = df_copy.rename(columns={
            timestamp_col: 'timestamp',
            item_id_col: 'item_id',
            target_col: 'target'
        })
        
        # Ensure the columns exist
        for col in ['timestamp', 'item_id', 'target']:
            if col not in ts_df.columns:
                st.error(f"⚠️ Required column '{col}' is missing")
                st.stop()
        
        # Create the TimeSeriesDataFrame with the renamed columns
        ts_data = TimeSeriesDataFrame.from_data_frame(
            ts_df, 
            id_column='item_id', 
            timestamp_column='timestamp',
            target_column='target'
        )
        
        st.success("✅ Successfully created TimeSeriesDataFrame")
        
        # Show time series info
        st.write("### Time Series Information")
        num_series = len(ts_data.item_ids)
        st.write(f"**Number of time series:** {num_series}")
        st.write(f"**Time range:** {min(ts_data.index.levels[1])} to {max(ts_data.index.levels[1])}")
        
        # Forecast section
        st.header("🔮 Forecasting")
        
        # Calculate reasonable default for prediction length
        avg_series_len = sum(len(ts_data.loc[item_id]) for item_id in ts_data.item_ids) / num_series
        reasonable_pred_len = max(1, int(avg_series_len * 0.2))  # 20% of data length
        
        prediction_length = st.slider(
            "Forecast Horizon (number of time steps)", 
            min_value=1, 
            max_value=min(60, int(avg_series_len * 0.5)),  # Cap at 50% of data or 60 steps
            value=min(30, reasonable_pred_len)
        )
        
        # Advanced model options
        with st.expander("Advanced Options"):
            eval_metric = st.selectbox(
                "Evaluation Metric", 
                ["MASE", "MAPE", "RMSE", "MAE"],
                help="Metric used to evaluate model performance"
            )
            
            time_limit = st.slider(
                "Training Time Limit (seconds)", 
                min_value=30, 
                max_value=900,  # 15 minutes max
                value=300
            )
            
            models_to_use = st.multiselect(
                "Models to Include", 
                ["DeepAR", "SimpleFeedForward", "TemporalFusionTransformer", "PatchTST"],
                default=["DeepAR", "SimpleFeedForward"],
                help="Deep learning models to include in training"
            )
        
        # Train Model
        if st.button("🚀 Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    predictor = TimeSeriesPredictor(
                        target='target',  # Using standardized column name
                        prediction_length=prediction_length,
                        eval_metric=eval_metric,
                        path="autogluon_models"
                    )
                    
                    predictor.fit(
                        ts_data,
                        hyperparameters={model: {} for model in models_to_use} if models_to_use else None,
                        time_limit=time_limit,
                        verbosity=2
                    )
                    
                    predictor.save("ts_model")
                    st.success("🎯 Model Training Complete!")
                    
                    # Show model info
                    with st.expander("🧠 Model Information"):
                        st.write("**Models trained:**")
                        for model_name in predictor.get_model_names():
                            st.write(f"- {model_name}")
                        
                        leaderboard = predictor.leaderboard()
                        st.write("**Model Leaderboard:**")
                        st.dataframe(leaderboard)
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.info("Please check your data and try adjusting parameters.")
        
        # Make Predictions
        if st.button("📈 Generate Forecasts"):
            with st.spinner("Generating forecasts..."):
                try:
                    predictor = TimeSeriesPredictor.load("ts_model")
                    forecasts = predictor.predict(ts_data)
                    
                    st.success("✅ Forecast Generated!")
                    
                    # Show forecast summary
                    st.write("### Forecast Preview")
                    st.dataframe(forecasts.head(10))
                    
                    # Plot Forecasts
                    st.write("### Forecast Visualization")
                    
                    # For datasets with multiple series, allow choosing which to view
                    if num_series > 1:
                        plot_option = st.radio(
                            "What to visualize:",
                            ["Average of all series", "Individual series"]
                        )
                        
                        if plot_option == "Average of all series":
                            # Create plot
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Prepare actual data
                            actual_by_time = ts_df.groupby('timestamp')['target'].mean()
                            
                            # Prepare forecast data - average across all series
                            forecast_by_time = forecasts.groupby("timestamp")["mean"].mean()
                            
                            # Plot
                            actual_by_time.plot(ax=ax, label="Actual (Average)")
                            forecast_by_time.plot(ax=ax, label="Forecast (Average)", linestyle="--")
                            
                            # Add confidence intervals if available
                            if '0.1' in forecasts.columns and '0.9' in forecasts.columns:
                                lower_bound = forecasts.groupby("timestamp")["0.1"].mean()
                                upper_bound = forecasts.groupby("timestamp")["0.9"].mean()
                                ax.fill_between(
                                    lower_bound.index, 
                                    lower_bound.values, 
                                    upper_bound.values, 
                                    alpha=0.2, 
                                    color='blue', 
                                    label="80% Confidence Interval"
                                )
                            
                            ax.legend()
                            plt.title("Average Time Series Forecast")
                            plt.xlabel("Date")
                            plt.ylabel('Value')
                            st.pyplot(fig)
                        else:
                            # Let user select which series to plot
                            selected_item = st.selectbox(
                                "Select series to visualize:",
                                options=sorted(ts_data.item_ids)
                            )
                            
                            # Create plot
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Filter data for the selected item
                            item_data = ts_df[ts_df['item_id'] == selected_item]
                            item_forecast = forecasts[forecasts.index.get_level_values('item_id') == selected_item]
                            
                            # Plot
                            item_data.set_index('timestamp')['target'].plot(ax=ax, label=f"Actual")
                            item_forecast["mean"].reset_index().set_index("timestamp")["mean"].plot(
                                ax=ax, label=f"Forecast", linestyle="--"
                            )
                            
                            # Add confidence intervals if available
                            if '0.1' in item_forecast.columns and '0.9' in item_forecast.columns:
                                lower_bound = item_forecast["0.1"].reset_index().set_index("timestamp")["0.1"]
                                upper_bound = item_forecast["0.9"].reset_index().set_index("timestamp")["0.9"]
                                ax.fill_between(
                                    lower_bound.index, 
                                    lower_bound.values, 
                                    upper_bound.values, 
                                    alpha=0.2, 
                                    color='blue', 
                                    label="80% Confidence Interval"
                                )
                            
                            ax.legend()
                            plt.title(f"Forecast for {selected_item}")
                            plt.xlabel("Date")
                            plt.ylabel('Value')
                            st.pyplot(fig)
                    else:
                        # For single series
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot
                        ts_df.set_index('timestamp')['target'].plot(ax=ax, label="Actual")
                        forecasts.reset_index().set_index("timestamp")["mean"].plot(ax=ax, label="Forecast", linestyle="--")
                        
                        # Add confidence intervals if available
                        if '0.1' in forecasts.columns and '0.9' in forecasts.columns:
                            lower_bound = forecasts.reset_index().set_index("timestamp")["0.1"]
                            upper_bound = forecasts.reset_index().set_index("timestamp")["0.9"]
                            ax.fill_between(
                                lower_bound.index, 
                                lower_bound.values, 
                                upper_bound.values, 
                                alpha=0.2, 
                                color='blue', 
                                label="80% Confidence Interval"
                            )
                        
                        ax.legend()
                        plt.title("Time Series Forecast")
                        plt.xlabel("Date")
                        plt.ylabel('Value')
                        st.pyplot(fig)
                    
                    # Download forecast as CSV
                    forecast_csv = forecasts.reset_index().to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=forecast_csv,
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )
                except FileNotFoundError:
                    st.error("Model not found. Please train the model first.")
                except Exception as e:
                    st.error(f"Error generating forecasts: {str(e)}")
                    st.info("Please check if the model was trained successfully.")
    except Exception as e:
        st.error(f"⚠️ Error creating TimeSeriesDataFrame: {str(e)}")
        st.write("Error details:", str(e))
        st.info("Please check your column selections and data format")