"""Helper utilities for the time series forecasting app."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import tensorflow as tf


def validate_data(df, min_points=100):
    """
    Validate input data.

    Args:
        df: Input dataframe or series
        min_points: Minimum required data points

    Returns:
        bool: Whether data is valid
    """
    if df is None:
        return False

    if isinstance(df, pd.DataFrame):
        num_rows = df.shape[0]
    else:
        num_rows = len(df)

    if num_rows < min_points:
        st.error(f"Dataset too small for reliable training (minimum {min_points} points required)")
        return False

    return True


def get_time_index(df):
    """
    Get appropriate time index for the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Time index for plotting
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    else:
        try:
            first_col = df.iloc[:, 0]
            if pd.to_datetime(first_col, errors='coerce').notnull().all():
                return pd.to_datetime(first_col)
        except:
            pass
    return np.arange(len(df))


def setup_logging():
    """Configure logging and warning settings."""
    import warnings
    import os
    import tensorflow as tf

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')


def create_download_button(df, filename="results.csv"):
    """
    Create a download button for dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to download
        filename (str): Name for the downloaded file
    """
    st.download_button(
        label="Download results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv',
    )


def setup_environment():
    """
    Configure logging and environment variables.
    
    This function suppresses warnings and configures TensorFlow logging.
    """
    # Suppress warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Configure TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
    tf.get_logger().setLevel('ERROR')


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        dict: Dictionary with availability status of optional dependencies
    """
    dependencies = {
        "prophet": False,
        "autogluon.timeseries": False
    }
    
    # Check Prophet
    try:
        import prophet
        dependencies["prophet"] = True
    except ImportError:
        pass
    
    # Check AutoGluon
    try:
        import autogluon.timeseries
        dependencies["autogluon.timeseries"] = True
    except ImportError:
        pass
    
    return dependencies


def format_time_series(df, date_col=None, target_col=None, freq=None):
    """
    Format dataframe as time series with proper index.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_col (str): Date column name
        target_col (str): Target column name
        freq (str): Frequency string for resampling
        
    Returns:
        pd.DataFrame: Formatted time series dataframe
    """
    try:
        # If date_col is provided, set it as index
        if date_col is not None and date_col in df.columns:
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                st.warning("Could not convert index to datetime. Some functionality might be limited.")
        
        # Sort index
        df = df.sort_index()
        
        # Resample if frequency is provided
        if freq is not None and isinstance(df.index, pd.DatetimeIndex):
            if target_col is not None:
                # Resample only the target column
                resampled = df[target_col].resample(freq).mean()
                df = pd.DataFrame(resampled)
            else:
                # Resample all numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                df = df[numeric_cols].resample(freq).mean()
        
        return df
        
    except Exception as e:
        st.error(f"Error formatting time series: {str(e)}")
        return df  # Return original dataframe on error