def train(self, train_data, val_data=None, **kwargs):
        """
        Train the AutoGluon model.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            **kwargs: Additional parameters
                - time_column: Column name for timestamps
                - target_column: Column name for target variable
                - item_id: Column name for item identifier
                - enable_ensemble: Whether to enable model ensembling
                
        Returns:
            self: Trained model
        """
        if not AUTOGLUON_AVAILABLE:
            st.error("AutoGluon not available. Install required packages.")
            return self
        
        if self.model is None:
            st.error("Model not initialized. Call build() first.")
            return self
        
        try:
            # Extract kwargs
            time_column = kwargs.get('time_column', self.time_column)
            target_column = kwargs.get('target_column', self.target_column)
            item_id = kwargs.get('item_id', self.item_id)
            enable_ensemble = kwargs.get('enable_ensemble', True)
            
            # Convert to TimeSeriesDataFrame if needed
            if not isinstance(train_data, TimeSeriesDataFrame):
                train_data = self._prepare_ts_dataframe(
                    train_data, 
                    time_column=time_column,
                    target_column=target_column,
                    item_id=item_id
                )
                
            if val_data is not None and not isinstance(val_data, TimeSeriesDataFrame):
                val_data = self._prepare_ts_dataframe(
                    val_data,
                    time_column=time_column,
                    target_column=target_column,
                    item_id=item_id
                )
            
            if train_data is None:
                st.error("Failed to prepare training data.")
                return None
                
            # Create a progress container
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Set up hyperparameters
            if self.hyperparameters is None:
                self.hyperparameters = {
                    'DeepAR': {},
                    'TemporalFusionTransformer': {},
                    'SimpleFeedforward': {},
                    'PatchTST': {},
                }
                
                # Basic models if ensemble is disabled
                if not enable_ensemble:
                    self.hyperparameters = {'DeepAR': {}}
            
            # Set up model configs
            config = {
                'hyperparameters': self.hyperparameters,
            }
            
            if self.time_limit is not None:
                config['time_limit'] = self.time_limit
            
            # Train the model
            status_text.text("Training AutoGluon model...")
            progress_bar.progress(0.1)
            
            try:
                self.model.fit(
                    train_data=train_data,
                    tuning_data=val_data,
                    **config
                )
                
                progress_bar.progress(1.0)
                status_text.text("AutoGluon model training complete!")
                
                # Cleanup progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success("AutoGluon model trained successfully!")
                
                # Show leaderboard
                try:
                    leaderboard = self.model.leaderboard()
                    st.subheader("Model Leaderboard")
                    st.dataframe(leaderboard)
                except Exception as e:
                    st.warning(f"Could not display leaderboard: {str(e)}")
                
                return self
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during AutoGluon model training: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return None
            
        except Exception as e:
            st.error(f"Error preparing for AutoGluon model training: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from .base import TimeSeriesModel

# Catch import errors to handle AutoGluon dependencies gracefully
try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class AutoGluonModel(TimeSeriesModel):
    """AutoGluon-based time series model implementation."""

    def __init__(self, name="AutoGluon"):
        super().__init__(name)
        self.model = None
        self.forecast = None
        self.prediction_length = 30
        self.freq = 'D'  # Default frequency (daily)
        self.time_column = None
        self.target_column = None
        self.item_id = None
        self.scaler = None

        # Check if AutoGluon is available
        if not AUTOGLUON_AVAILABLE:
            st.error("""
            AutoGluon is not installed. Install it with:
            ```
            pip install autogluon.timeseries
            ```
            """)

    def build(self, prediction_length=30, freq='D', hyperparameters=None, time_limit=None):
        """
        Build AutoGluon model with specified parameters.
        
        Args:
            prediction_length (int): Number of time steps to predict
            freq (str): Time series frequency (D=daily, H=hourly, etc.)
            hyperparameters (dict): Hyperparameters for AutoGluon
            time_limit (int): Time limit for training in seconds
            
        Returns:
            self: Model instance
        """
        if not AUTOGLUON_AVAILABLE:
            st.error("AutoGluon not available. Install required packages.")
            return self
            
        # Validate inputs
        if prediction_length <= 0:
            st.warning(f"Invalid prediction_length: {prediction_length}. Using 30 instead.")
            prediction_length = 30
            
        # Validate frequency
        valid_freqs = ['D', 'H', 'W', 'M', 'Q', 'Y', 'B', 'min', 'S']
        if freq not in valid_freqs and not any(freq.endswith(v) for v in valid_freqs):
            st.warning(f"Potentially invalid frequency: {freq}. This might cause issues.")
        
        self.prediction_length = prediction_length
        self.freq = freq
        
        # Set up default hyperparameters if not provided
        self.hyperparameters = hyperparameters if hyperparameters is not None else {
            'DeepAR': {},
            'TemporalFusionTransformer': {},
            'SimpleFeedforward': {},
            'PatchTST': {},
        }
        
        self.time_limit = time_limit
        
        # Initialize the predictor with optional hyperparameters
        try:
            model_path = os.path.join('models', f'autogluon_{self.name}')
            os.makedirs(model_path, exist_ok=True)
            
            self.model = TimeSeriesPredictor(
                prediction_length=prediction_length,
                path=model_path,
                freq=freq,
                eval_metric='MASE',  # Mean Absolute Scaled Error
                verbosity=2
            )
            
            st.success(f"AutoGluon TimeSeriesPredictor initialized with frequency {freq} and prediction length {prediction_length}")
        except Exception as e:
            st.error(f"Error initializing AutoGluon model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        
        return self

    def _prepare_ts_dataframe(self, df, time_column=None, target_column=None, item_id=None):
        """
        Convert a pandas DataFrame to AutoGluon TimeSeriesDataFrame format.
        
        Args:
            df (pd.DataFrame): Input dataframe
            time_column (str): Column name for timestamps
            target_column (str): Column name for target variable
            item_id (str): Column name for item identifier (for multiple time series)
            
        Returns:
            TimeSeriesDataFrame: AutoGluon formatted time series data
        """
        try:
            # Save column names for later use
            self.time_column = time_column
            self.target_column = target_column
            self.item_id = item_id
            
            # Handle different input formats
            if time_column is None:
                # If time_column is not specified, assume index is timestamp
                if isinstance(df.index, pd.DatetimeIndex):
                    time_index = df.index
                    if target_column is None:
                        # If target not specified, use first column
                        target_column = df.columns[0]
                    # Create a new DataFrame with timestamp as index and target as column
                    if item_id is None:
                        ts_df = pd.DataFrame({
                            'target': df[target_column]
                        }, index=time_index)
                    else:
                        # Multiple time series case
                        ts_df = pd.DataFrame({
                            'target': df[target_column],
                            'item_id': df[item_id]
                        }, index=time_index)
                else:
                    st.error("If time_column is not specified, DataFrame index must be a DatetimeIndex")
                    return None
            else:
                # If time_column is specified, use it as the index
                if target_column is None:
                    # If target not specified, use first numeric column that's not time_column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    target_candidates = [col for col in numeric_cols if col != time_column]
                    if len(target_candidates) > 0:
                        target_column = target_candidates[0]
                    else:
                        st.error("Could not identify a numeric target column")
                        return None
                
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_dtype(df[time_column]):
                    df[time_column] = pd.to_datetime(df[time_column])
                
                # Create TimeSeriesDataFrame
                if item_id is None:
                    # Single time series
                    ts_df = pd.DataFrame({
                        'target': df[target_column]
                    }, index=df[time_column])
                else:
                    # Multiple time series
                    ts_df = pd.DataFrame({
                        'target': df[target_column],
                        'item_id': df[item_id]
                    }, index=df[time_column])
            
            # Convert to AutoGluon TimeSeriesDataFrame
            ts_df = TimeSeriesDataFrame(ts_df)
            
            # Validate frequency
            inferred_freq = pd.infer_freq(ts_df.index)
            if inferred_freq is None:
                st.warning(f"Could not infer frequency from data. Using specified frequency: {self.freq}")
            else:
                # Update frequency if it's different from specified
                if inferred_freq != self.freq:
                    st.info(f"Inferred frequency '{inferred_freq}' differs from specified '{self.freq}'. Using inferred.")
                    self.freq = inferred_freq
            
            return ts_df
            
        except Exception as e:
            st.error(f"Error preparing TimeSeriesDataFrame: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def train(self, train_data, val_data=None, **kwargs):
        """
        Train the AutoGluon model.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            **kwargs: Additional parameters
                - time_column: Column name for timestamps
                - target_column: Column name for target variable
                - item_id: Column name for item identifier
                - enable_ensemble: Whether to enable model ensembling
                
        Returns:
            self: Trained model
        """
        if not AUTOGLUON_AVAILABLE:
            st.error("AutoGluon not available. Install required packages.")
            return self
        
        try:
            # Extract kwargs
            time_column = kwargs.get('time_column', self.time_column)
            target_column = kwargs.get('target_column', self.target_column)
            item_id = kwargs.get('item_id', self.item_id)
            enable_ensemble = kwargs.get('enable_ensemble', True)
            
            # Convert to TimeSeriesDataFrame if needed
            if not isinstance(train_data, TimeSeriesDataFrame):
                train_data = self._prepare_ts_dataframe(
                    train_data, 
                    time_column=time_column,
                    target_column=target_column,
                    item_id=item_id
                )
                
            if val_data is not None and not isinstance(val_data, TimeSeriesDataFrame):
                val_data = self._prepare_ts_dataframe(
                    val_data,
                    time_column=time_column,
                    target_column=target_column,
                    item_id=item_id
                )
            
            if train_data is None:
                return None
                
            # Create a progress container
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Set up hyperparameters
            if self.hyperparameters is None:
                self.hyperparameters = {
                    'DeepAR': {},
                    'TemporalFusionTransformer': {},
                    'SimpleFeedforward': {},
                    'PatchTST': {},
                }
                
                # Basic models if ensemble is disabled
                if not enable_ensemble:
                    self.hyperparameters = {'DeepAR': {}}
            
            # Set up model configs
            config = {
                'hyperparameters': self.hyperparameters,
                'time_limit': self.time_limit
            }
            
            # Train the model
            status_text.text("Training AutoGluon model...")
            progress_bar.progress(0.1)
            
            self.model.fit(
                train_data=train_data,
                tuning_data=val_data,
                **config
            )
            
            progress_bar.progress(1.0)
            status_text.text("AutoGluon model training complete!")
            
            # Cleanup progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("AutoGluon model trained successfully!")
            
            # Show leaderboard
            leaderboard = self.model.leaderboard()
            st.subheader("Model Leaderboard")
            st.dataframe(leaderboard)
            
            return self
            
        except Exception as e:
            st.error(f"Error training AutoGluon model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def predict(self, test_data=None, prediction_length=None, quantiles=None):
        """
        Generate forecasts with the trained model.
        
        Args:
            test_data: Test data for starting the forecast
            prediction_length (int): Length of forecast (defaults to self.prediction_length)
            quantiles (list): Quantiles to include in the forecast
            
        Returns:
            pd.DataFrame: Forecast results
        """
        if not AUTOGLUON_AVAILABLE or self.model is None:
            st.error("AutoGluon model not available or not trained.")
            return None
            
        try:
            # Use provided prediction length or default
            if prediction_length is None:
                prediction_length = self.prediction_length
                
            # Set default quantiles if not provided
            if quantiles is None:
                quantiles = [0.1, 0.5, 0.9]
                
            # Convert test data if provided
            if test_data is not None and not isinstance(test_data, TimeSeriesDataFrame):
                test_data = self._prepare_ts_dataframe(
                    test_data,
                    time_column=self.time_column,
                    target_column=self.target_column,
                    item_id=self.item_id
                )
                
            # Make predictions
            with st.spinner("Generating forecasts..."):
                if test_data is not None:
                    self.forecast = self.model.predict(
                        test_data,
                        quantiles=quantiles
                    )
                else:
                    # Use latest data from training
                    self.forecast = self.model.predict(
                        quantiles=quantiles
                    )
                    
            st.success("Forecast generated successfully!")
            return self.forecast
            
        except Exception as e:
            st.error(f"Error generating AutoGluon forecast: {str(e)}")
            return None

    def evaluate(self, test_data):
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test data with ground truth
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not AUTOGLUON_AVAILABLE or self.model is None:
            st.error("AutoGluon model not available or not trained.")
            return None
            
        try:
            # Convert test data if needed
            if not isinstance(test_data, TimeSeriesDataFrame):
                test_data = self._prepare_ts_dataframe(
                    test_data,
                    time_column=self.time_column,
                    target_column=self.target_column,
                    item_id=self.item_id
                )
                
            if test_data is None:
                return None
                
            # Evaluate the model
            with st.spinner("Evaluating model..."):
                scores = self.model.evaluate(test_data)
                
            # Create a nicer metrics dictionary
            metrics = {
                'MASE': scores['MASE'],
                'MAPE': scores['MAPE'] * 100 if 'MAPE' in scores else None,
                'RMSE': scores['RMSE'] if 'RMSE' in scores else None,
                'MAE': scores['MAE'] if 'MAE' in scores else None,
            }
            
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            return metrics
            
        except Exception as e:
            st.error(f"Error evaluating AutoGluon model: {str(e)}")
            return None

    def plot_forecast(self, past_data=None, item_id=None, **kwargs):
        """
        Plot the forecast.
        
        Args:
            past_data: Historical data to include in the plot
            item_id: ID of the item to plot for multi-time series
            **kwargs: Additional plotting parameters
            
        Returns:
            matplotlib.Figure: Plot figure
        """
        if not AUTOGLUON_AVAILABLE or self.forecast is None:
            st.error("No forecast available. Run predict() first.")
            return None
            
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data if provided
            if past_data is not None:
                if isinstance(past_data, TimeSeriesDataFrame):
                    ts_df = past_data
                else:
                    ts_df = self._prepare_ts_dataframe(
                        past_data,
                        time_column=self.time_column,
                        target_column=self.target_column,
                        item_id=self.item_id
                    )
                
                if ts_df is not None:
                    # If multi-time series, filter for the specified item
                    if item_id is not None:
                        if 'item_id' in ts_df.columns:
                            ts_df = ts_df[ts_df['item_id'] == item_id]
                        else:
                            st.warning(f"item_id column not found in historical data")
                    
                    # Plot historical data
                    ax.plot(ts_df.index, ts_df['target'], label='Historical', color='black')
            
            # Get the forecast
            forecast_df = self.forecast.copy()
            
            # Filter for a specific item if specified for multi-time series
            if item_id is not None and 'item_id' in forecast_df.index.names:
                forecast_df = forecast_df.xs(item_id, level='item_id')
            
            # Plot median forecast
            if 'mean' in forecast_df.columns:
                ax.plot(forecast_df.index, forecast_df['mean'], label='Forecast (Mean)', 
                        color='blue', linestyle='--')
            elif '0.5' in forecast_df.columns:
                ax.plot(forecast_df.index, forecast_df['0.5'], label='Forecast (Median)', 
                        color='blue', linestyle='--')
            
            # Plot prediction intervals if available
            lower_quantile = None
            upper_quantile = None
            
            # Find the lowest and highest quantiles
            for col in forecast_df.columns:
                try:
                    q = float(col)
                    if lower_quantile is None or q < lower_quantile:
                        lower_quantile = q
                    if upper_quantile is None or q > upper_quantile:
                        upper_quantile = q
                except:
                    pass
            
            # Plot prediction intervals
            if lower_quantile is not None and upper_quantile is not None:
                ax.fill_between(
                    forecast_df.index,
                    forecast_df[str(lower_quantile)],
                    forecast_df[str(upper_quantile)],
                    color='blue', alpha=0.2,
                    label=f'Prediction Interval ({lower_quantile}-{upper_quantile})'
                )
            
            # Add labels and legend
            ax.set_title("Time Series Forecast", fontsize=14, pad=20)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error plotting forecast: {str(e)}")
            return None
            if lower_quantile is not None and upper_quantile is not None:
                ax.fill_between(
                    forecast_df.index,
                    forecast_df[str(lower_quantile)],
                    forecast_df[str(upper_quantile)],
                    color='blue', alpha=0.2,
                    label=f'Prediction Interval ({lower_quantile}-{upper_quantile})'
                )
            
            # Add labels and legend
            ax.set_title("Time Series Forecast", fontsize=14, pad=20)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error plotting forecast: {str(e)}")
            return None

    def get_parameters(self):
        """Get the model's parameters."""
        return {
            'prediction_length': self.prediction_length,
            'freq': self.freq,
            'hyperparameters': self.hyperparameters,
            'time_limit': self.time_limit
        }

    def save(self, path):
        """Save model to file."""
        if not AUTOGLUON_AVAILABLE or self.model is None:
            st.error("AutoGluon model not available or not trained.")
            return
            
        try:
            # Save using AutoGluon's built-in save functionality
            # The model is already saved at the path specified during initialization
            model_path = os.path.join('models', f'autogluon_{self.name}')
            st.success(f"Model saved to {model_path}")
            
            # Save additional parameters
            import json
            params_path = os.path.join(path, 'model_params.json')
            with open(params_path, 'w') as f:
                json.dump({
                    'prediction_length': self.prediction_length,
                    'freq': self.freq,
                    'time_column': self.time_column,
                    'target_column': self.target_column,
                    'item_id': self.item_id
                }, f)
                
        except Exception as e:
            st.error(f"Error saving AutoGluon model: {str(e)}")

    def load(self, path):
        """Load model from file."""
        if not AUTOGLUON_AVAILABLE:
            st.error("AutoGluon not available. Install required packages.")
            return
            
        try:
            # Load the model
            self.model = TimeSeriesPredictor.load(path)
            
            # Load additional parameters
            import json
            params_path = os.path.join(path, 'model_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                    self.prediction_length = params.get('prediction_length', 30)
                    self.freq = params.get('freq', 'D')
                    self.time_column = params.get('time_column')
                    self.target_column = params.get('target_column')
                    self.item_id = params.get('item_id')
            
            st.success(f"Model loaded from {path}")
            
        except Exception as e:
            st.error(f"Error loading AutoGluon model: {str(e)}")


def create_autogluon_model(**kwargs):
    """Factory function to create an AutoGluon model."""
    return AutoGluonModel(**kwargs)