
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st
import os
import sys
import traceback

# Import your module components
from models.deep_learning import SimpleRNNModel, LSTMModel, StackedModel, create_dl_model
from models.traditional import ARIMAModel, SARIMAModel, create_traditional_model
from models.machine_learning import RandomForestModel, XGBoostModel, create_ml_model
from data.preprocessor import TimeSeriesPreprocessor
from utils.visualization import DataVisualizer
from utils.helpers import validate_data, get_time_index, setup_environment
from models.trainer import ModelTrainer
from config import *  # Import constants from config.py

from config import *
from data.loader import load_data, get_github_files, get_dtypes_info
from data.preprocessor import TimeSeriesPreprocessor
from utils.visualization import DataVisualizer
from utils.helpers import setup_environment, validate_data, get_time_index
from utils.model_n_forecast import modeling_and_forecasting


# Check for optional dependencies
try:
    from models.prophet import ProphetModel, create_prophet_model
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from models.autogluon import AutoGluonModel, create_autogluon_model
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


def results_and_export():
    """Fourth stage: Results visualization and export."""
    st.markdown('<div class="section-header"><h2>📊 Results & Export</h2></div>', unsafe_allow_html=True)
    
    if 'model_results' not in st.session_state or not st.session_state.model_results:
        st.warning("No model results available. Please complete the Modeling & Forecasting stage first.")
        if st.button("Go to Modeling & Forecasting"):
            st.session_state.current_stage = "Modeling & Forecasting"
            st.rerun()
        return
    
    st.info("This section is under development. Please check back later!")
    st.markdown("""
    ### Coming Soon:
    - Detailed result visualizations
    - Model comparison
    - Export options for forecasts
    - Report generation
    """)

def main():
    """Main application function with improved UI flow."""
    setup_environment()
    
    st.set_page_config(
        layout="wide",
        page_title="Time Series Forecasting App",
        page_icon="📈"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Time Series Forecasting Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    An advanced platform for time series data analysis and forecasting with multiple models,
    including traditional approaches, machine learning, deep learning, Facebook Prophet, and AutoGluon.
    """)
    
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio(
            "Select Stage",
            ["Data Upload & Preview", "Data Preprocessing", "Modeling & Forecasting", "Results & Export"]
        )
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'time_column' not in st.session_state:
        st.session_state.time_column = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    
    if app_mode == "Data Upload & Preview":
        data_upload_and_preview()
    elif app_mode == "Data Preprocessing":
        data_preprocessing()
    elif app_mode == "Modeling & Forecasting":
        modeling_and_forecasting()
    elif app_mode == "Results & Export":
        results_and_export()

def data_upload_and_preview():
    """First stage: Data upload, preview, and initial analysis."""
    st.markdown('<div class="section-header"><h2>📤 Data Upload & Preview</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Select Data Source")
    data_source = st.radio(
        "How would you like to provide data?",
        ["Upload CSV File", "GitHub Repository URL", "Example Datasets"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file:
            with st.spinner("Loading data..."):
                df = load_data("upload", uploaded_file=uploaded_file)
                if df is not None:
                    st.success("File uploaded and processed successfully!")
    
    elif data_source == "GitHub Repository URL":
        github_url = st.text_input(
            "Enter GitHub repository URL",
            "https://github.com/PJalgotrader/Deep_forecasting-USU/tree/main/data"
        )
        if github_url:
            with st.spinner("Fetching repository contents..."):
                csv_files = get_github_files(github_url)
                if csv_files:
                    selected_file = st.selectbox(
                        "Select a CSV file from repository",
                        [file[0] for file in csv_files]
                    )
                    if selected_file:
                        with st.spinner("Loading file..."):
                            file_url = next(file[1] for file in csv_files if file[0] == selected_file)
                            df = load_data("github", github_url=github_url, selected_file=file_url)
                            if df is not None:
                                st.success(f"File {selected_file} loaded successfully!")
                else:
                    st.warning("No CSV files found in the repository.")
    
    elif data_source == "Example Datasets":
        example_datasets = {
            "Stock Prices": "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data/yfinance.csv",
            "Energy Consumption": "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data/AEP_hourly.csv",
            "Air Quality": "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data/air_quality.csv"
        }
        selected_dataset = st.selectbox("Select an example dataset", list(example_datasets.keys()))
        
        if selected_dataset:
            with st.spinner(f"Loading {selected_dataset} dataset..."):
                try:
                    if selected_dataset == "Stock Prices":
                        # Use fallback method since parse_dates fails for this dataset
                        df = pd.read_csv(example_datasets[selected_dataset], index_col=0, header=[0, 1])['Close']
                        df = df.reset_index()
                        st.success("Loaded stock price data using fallback method")
                    else:
                        df = pd.read_csv(example_datasets[selected_dataset])
                        st.success(f"{selected_dataset} dataset loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading example dataset: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if df is not None:
        st.session_state.data = df
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        with st.expander("View Data Sample", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("View Data Types and Information"):
            st.dataframe(get_dtypes_info(df), use_container_width=True)
        
        with st.expander("View Statistics"):
            if df.select_dtypes(include=['number']).shape[1] > 0:
                st.dataframe(df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistics.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⏱️ Time Index & Target Selection")
        
        # Debug output to inspect index and columns
        # st.write("Debug: DataFrame index type:", type(df.index))
        # st.write("Debug: DataFrame columns:", df.columns.tolist())
        # st.write("Debug: Is index a DatetimeIndex?", isinstance(df.index, pd.DatetimeIndex))
        
        # Identify potential datetime columns (for debugging purposes)
        time_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_cols.append(col)
            elif col.lower() == 'date' or any(time_word in col.lower() for time_word in ['date', 'time', 'day', 'year', 'month']):
                time_cols.append(col)
            else:
                try:
                    pd.to_datetime(df[col].head(), errors='raise')
                    time_cols.append(col)
                except:
                    pass
        
        # Always include the DataFrame index as a selectable option
        has_date_index = isinstance(df.index, pd.DatetimeIndex)
        time_index_options = ["Use DataFrame Index"]
        
        if not has_date_index and (df.index.name not in df.columns):
            st.write(f"Debug: Adding index to selection options. Index type: {type(df.index)}")
            time_index_options.extend(df.columns.tolist())
            time_index_options.append("None/Other")
        else:
            time_index_options.extend(df.columns.tolist())
            time_index_options.append("None/Other")
        
        time_index_selection = st.selectbox(
            "Select Time Index Column",
            time_index_options,
            index=0
        )
        
        st.write("Debug: Selected time index:", time_index_selection)
        
        if time_index_selection == "Use DataFrame Index":
            if has_date_index:
                st.info("Using DataFrame index as time index (already datetime format).")
                time_column = None
            else:
                try:
                    df_copy = df.copy()
                    df_copy.index = pd.to_datetime(df_copy.index)
                    df = df_copy
                    st.success("Converted index to datetime format!")
                    time_column = None
                    has_date_index = True
                except Exception as e:
                    st.error(f"Could not convert index to datetime: {str(e)}")
                    if len(df.columns) > 0:
                        time_column = df.columns[0]
                    else:
                        time_column = None
        elif time_index_selection == "None/Other":
            time_column = st.selectbox("Select column to use as time index", df.columns)
            if time_column and not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                try:
                    df[time_column] = pd.to_datetime(df[time_column])
                    st.success(f"Automatically converted {time_column} to datetime format!")
                except Exception as e:
                    st.warning(f"Could not convert {time_column} to datetime format: {str(e)}")
        else:
            time_column = time_index_selection
        
        if time_column is not None and time_column != "None/Other":
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                    df[time_column] = pd.to_datetime(df[time_column])
                df.index = df[time_column]
                st.info(f"Set {time_column} as time index.")
                if st.checkbox("Remove time column from data", value=False):
                    df = df.drop(columns=[time_column])
            except Exception as e:
                st.warning(f"Could not set time index automatically: {str(e)}")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            target_column = st.selectbox(
                "Select Target Variable (the value you want to forecast)",
                numeric_cols
            )
            
            if target_column:
                st.subheader(f"Target Variable: {target_column}")
                
                fig = px.line(
                    df, y=target_column, x=df.index if isinstance(df.index, pd.DatetimeIndex) else None,
                    title=f"{target_column} Time Series"
                )
                fig.update_layout(
                    xaxis_title="Time" if isinstance(df.index, pd.DatetimeIndex) else "Data Points",
                    yaxis_title=target_column,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.target_column = target_column
                st.session_state.time_column = time_column
                
                if st.button("Proceed to Data Preprocessing"):
                    st.session_state.data = df
                    st.rerun()
        else:
            st.warning("No numeric columns found for target selection.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please select a data source and load data to continue.")

def data_preprocessing():
    """Second stage: Data preprocessing and feature engineering."""
    st.markdown('<div class="section-header"><h2>🧹 Data Preprocessing</h2></div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please go back to Data Upload & Preview.")
        if st.button("Go to Data Upload"):
            st.rerun()
        return
    
    df = st.session_state.data.copy()
    target_column = st.session_state.target_column
    
    if target_column is None or target_column not in df.columns:
        st.error("Target column not selected or not found in data.")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Current Dataset")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"Target Variable: {target_column}")
    
    with st.expander("View Current Data Sample"):
        st.dataframe(df.head())
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preprocessing Options")
    
    missing_col, outlier_col = st.columns(2)
    
    with missing_col:
        st.write("**Missing Value Handling**")
        missing_values = df.isnull().sum().sum()
        st.write(f"Total missing values: {missing_values}")
        
        missing_method = st.selectbox(
            "Method for Handling Missing Values",
            ["None", "Drop Rows", "Forward Fill", "Backward Fill", "Linear Interpolation", "Mean/Median/Mode"]
        )
        
        if missing_method == "Mean/Median/Mode":
            fill_method = st.radio(
                "Fill method for numeric columns:",
                ["Mean", "Median", "Mode"]
            )
    
    with outlier_col:
        st.write("**Outlier Detection and Handling**")
        
        outlier_method = st.selectbox(
            "Method for Detecting Outliers",
            ["None", "IQR Method", "Z-Score Method", "Isolation Forest", "Winsorization"]
        )
        
        if outlier_method != "None":
            outlier_action = st.radio(
                "Action for Outliers",
                ["Remove", "Cap", "Replace with Mean/Median"]
            )
            
            if outlier_method == "Z-Score Method":
                z_threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
            elif outlier_method == "IQR Method":
                iqr_factor = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1)
    
    transform_col, features_col = st.columns(2)
    
    with transform_col:
        st.write("**Data Transformation**")
        
        transform_method = st.selectbox(
            "Transformation Method",
            ["None", "Log Transform", "Square Root", "Box-Cox", "Standardization", "Min-Max Scaling"]
        )
        
        if transform_method == "Box-Cox":
            st.info("Box-Cox requires positive values.")
    
    with features_col:
        st.write("**Feature Engineering**")
        
        add_lag_features = st.checkbox("Add Lag Features")
        if add_lag_features:
            lag_values = st.multiselect(
                "Select Lag Values",
                [1, 2, 3, 5, 7, 14, 21, 28, 30],
                default=[1, 7]
            )
        
        add_rolling_features = st.checkbox("Add Rolling Window Features")
        if add_rolling_features:
            window_sizes = st.multiselect(
                "Select Window Sizes",
                [3, 5, 7, 14, 30],
                default=[7]
            )
            rolling_features = st.multiselect(
                "Select Rolling Features",
                ["Mean", "Std", "Min", "Max"],
                default=["Mean"]
            )
        
        add_date_features = st.checkbox("Add Date Features", value=True)
        if add_date_features and isinstance(df.index, pd.DatetimeIndex):
            date_features = st.multiselect(
                "Select Date Features",
                ["Year", "Month", "Day", "DayOfWeek", "Quarter", "WeekOfYear", "IsDayOff"],
                default=["Month", "DayOfWeek"]
            )
    
    if st.button("Apply Preprocessing"):
        try:
            with st.spinner("Applying preprocessing steps..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                progress_text.text("Handling missing values...")
                if missing_method != "None":
                    if missing_method == "Drop Rows":
                        df = df.dropna()
                    elif missing_method == "Forward Fill":
                        df = df.fillna(method='ffill')
                    elif missing_method == "Backward Fill":
                        df = df.fillna(method='bfill')
                    elif missing_method == "Linear Interpolation":
                        df = df.interpolate(method='linear')
                    elif missing_method == "Mean/Median/Mode":
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if fill_method == "Mean":
                            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                        elif fill_method == "Median":
                            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                        elif fill_method == "Mode":
                            for col in numeric_cols:
                                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].mean())
                progress_bar.progress(0.25)
                
                progress_text.text("Handling outliers...")
                if outlier_method != "None" and target_column in df.columns:
                    if outlier_method == "IQR Method":
                        Q1 = df[target_column].quantile(0.25)
                        Q3 = df[target_column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - iqr_factor * IQR
                        upper_bound = Q3 + iqr_factor * IQR
                        outliers = (df[target_column] < lower_bound) | (df[target_column] > upper_bound)
                    elif outlier_method == "Z-Score Method":
                        z_scores = abs(stats.zscore(df[target_column]))
                        outliers = z_scores > z_threshold
                    
                    if outlier_action == "Remove":
                        df = df[~outliers]
                    elif outlier_action == "Cap":
                        if outlier_method == "IQR Method":
                            df.loc[df[target_column] < lower_bound, target_column] = lower_bound
                            df.loc[df[target_column] > upper_bound, target_column] = upper_bound
                        elif outlier_method == "Z-Score Method":
                            df.loc[outliers, target_column] = df[target_column].mean()
                    elif outlier_action == "Replace with Mean/Median":
                        df.loc[outliers, target_column] = df[target_column].median()
                progress_bar.progress(0.5)
                
                progress_text.text("Applying transformations...")
                if transform_method != "None" and target_column in df.columns:
                    if transform_method == "Log Transform":
                        min_val = df[target_column].min()
                        if min_val <= 0:
                            offset = abs(min_val) + 1
                            df[target_column] = np.log1p(df[target_column] + offset)
                        else:
                            df[target_column] = np.log1p(df[target_column])
                    elif transform_method == "Square Root":
                        min_val = df[target_column].min()
                        if min_val < 0:
                            offset = abs(min_val) + 1
                            df[target_column] = np.sqrt(df[target_column] + offset)
                        else:
                            df[target_column] = np.sqrt(df[target_column])
                    elif transform_method == "Box-Cox":
                        min_val = df[target_column].min()
                        if min_val <= 0:
                            offset = abs(min_val) + 1
                            df[target_column], _ = stats.boxcox(df[target_column] + offset)
                        else:
                            df[target_column], _ = stats.boxcox(df[target_column])
                    elif transform_method == "Standardization":
                        df[target_column] = (df[target_column] - df[target_column].mean()) / df[target_column].std()
                    elif transform_method == "Min-Max Scaling":
                        df[target_column] = (df[target_column] - df[target_column].min()) / (df[target_column].max() - df[target_column].min())
                progress_bar.progress(0.75)
                
                progress_text.text("Engineering features...")
                preprocessor = TimeSeriesPreprocessor()
                
                if add_lag_features and lag_values:
                    for lag in lag_values:
                        df[f'lag_{lag}'] = df[target_column].shift(lag)
                
                if add_rolling_features and window_sizes:
                    for window in window_sizes:
                        if "Mean" in rolling_features:
                            df[f'rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
                        if "Std" in rolling_features:
                            df[f'rolling_std_{window}'] = df[target_column].rolling(window=window).std()
                        if "Min" in rolling_features:
                            df[f'rolling_min_{window}'] = df[target_column].rolling(window=window).min()
                        if "Max" in rolling_features:
                            df[f'rolling_max_{window}'] = df[target_column].rolling(window=window).max()
                
                if add_date_features and isinstance(df.index, pd.DatetimeIndex) and date_features:
                    if "Year" in date_features:
                        df['year'] = df.index.year
                    if "Month" in date_features:
                        df['month'] = df.index.month
                    if "Day" in date_features:
                        df['day'] = df.index.day
                    if "DayOfWeek" in date_features:
                        df['day_of_week'] = df.index.dayofweek
                    if "Quarter" in date_features:
                        df['quarter'] = df.index.quarter
                    if "WeekOfYear" in date_features:
                        df['week_of_year'] = df.index.isocalendar().week
                    if "IsDayOff" in date_features:
                        df['is_weekend'] = df.index.dayofweek >= 5
                
                df = df.dropna()
                
                progress_bar.progress(1.0)
                progress_text.text("Preprocessing complete!")
                
                st.session_state.preprocessed_data = df
                
                st.success("Preprocessing applied successfully!")
                
                progress_bar.empty()
                progress_text.empty()
                
                st.subheader("Preprocessed Data Preview")
                st.dataframe(df.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Shape", f"{st.session_state.data.shape[0]} × {st.session_state.data.shape[1]}")
                with col2:
                    st.metric("New Shape", f"{df.shape[0]} × {df.shape[1]}")
                
                st.subheader("Target Variable Before vs. After")
                
                before_after_fig = go.Figure()
                
                before_after_fig.add_trace(go.Scatter(
                    x=st.session_state.data.index,
                    y=st.session_state.data[target_column],
                    mode='lines',
                    name='Before Preprocessing'
                ))
                
                before_after_fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[target_column],
                    mode='lines',
                    name='After Preprocessing'
                ))
                
                before_after_fig.update_layout(
                    title=f"{target_column} - Before vs. After Preprocessing",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    legend_title="Legend",
                    template="plotly_white"
                )
                
                st.plotly_chart(before_after_fig, use_container_width=True)
                
                if st.button("Proceed to Modeling & Forecasting"):
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()
