"""Visualization utilities for the time series forecasting app."""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict


class DataVisualizer:
    """Class for data visualization functions."""

    def plot_time_series(self, df, selected_cols, time_index, title="Time Series Plot"):
        """
        Plot time series data with enhanced styling.
        
        Args:
            df (pd.DataFrame): Input dataframe
            selected_cols (list): List of columns to plot
            time_index: Time index for x-axis
            title (str): Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for col in selected_cols:
            ax.plot(time_index, df[col], label=col, alpha=0.7)

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def plot_predictions_with_intervals(self, time_index, actual, predictions,
                                        lower=None, upper=None, title="Model Predictions"):
        """
        Plot predictions with confidence intervals.
        
        Args:
            time_index: Time index for x-axis
            actual (array-like): Actual values
            predictions (array-like): Predicted values
            lower (array-like): Lower confidence bounds
            upper (array-like): Upper confidence bounds
            title (str): Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual values
        ax.plot(time_index, actual, label='Actual', color='black', alpha=0.7)

        # Plot predictions
        ax.plot(time_index, predictions, label='Predictions', color='blue', linestyle='--')

        # Plot confidence intervals if provided
        if lower is not None and upper is not None:
            ax.fill_between(time_index, lower, upper, color='blue', alpha=0.2,
                            label='Prediction Interval')

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def plot_forecast(self, history, forecast, intervals=None, title="Forecast"):
        """
        Plot historical data with forecast.
        
        Args:
            history (pd.Series): Historical data
            forecast (pd.Series): Forecast data
            intervals (dict): Confidence intervals
            title (str): Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        ax.plot(history.index, history, label='Historical', color='black', alpha=0.7)

        # Plot forecast
        ax.plot(forecast.index, forecast, label='Forecast', color='blue',
                linestyle='--', linewidth=2)

        # Plot prediction intervals if provided
        if intervals is not None:
            ax.fill_between(forecast.index,
                            intervals['lower'], intervals['upper'],
                            color='blue', alpha=0.2, label='Prediction Interval')

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def plot_model_comparison(self, results: List[Dict], metrics=['MAE', 'RMSE', 'MAPE']):
        """
        Plot model comparison charts.
        
        Args:
            results (list): List of dictionaries with model results
            metrics (list): List of metrics to compare
            
        Returns:
            pd.DataFrame: DataFrame with comparison results
        """
        # Convert results to DataFrame
        if not results:
            st.warning("No results to display")
            return None

        results_df = pd.DataFrame(results)

        # Create bar plots for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in results_df.columns:
                results_df.plot(kind='bar', y=metric, ax=ax)
                ax.set_title(f'Model Comparison - {metric}')
                ax.set_xlabel('Models')
                ax.set_ylabel(metric)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        return results_df

    def display_data_info(self, df):
        """
        Display enhanced dataset information.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        st.subheader("Dataset Overview")

        # Basic info in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024 ** 2:.2f} MB")

        # Data types and missing values
        st.subheader("Data Types and Missing Values")
        dtypes_df = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(dtypes_df)

        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        # Additional info for time series
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            st.subheader("Time Series Information")
            freq = pd.infer_freq(df.index)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Start Date", df.index[0])
            with col2:
                st.metric("End Date", df.index[-1])
            with col3:
                st.metric("Frequency", str(freq) if freq else "Irregular")