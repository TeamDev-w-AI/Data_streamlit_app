"""Traditional time series model implementations."""

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from .base import TimeSeriesModel


class ARIMAModel(TimeSeriesModel):
    """ARIMA model implementation."""

    def __init__(self, name="ARIMA", p=1, d=1, q=1):
        """
        Initialize ARIMA model.
        
        Args:
            name (str): Model name
            p (int): AR order
            d (int): Differencing order
            q (int): MA order
        """
        super().__init__(name)
        self.p = p
        self.d = d
        self.q = q
        self.fitted_model = None

    def build(self, p=None, d=None, q=None):
        """
        Build model with specified parameters.
        
        Args:
            p (int, optional): AR order
            d (int, optional): Differencing order
            q (int, optional): MA order
            
        Returns:
            self: Model instance
        """
        if p is not None:
            self.p = p
        if d is not None:
            self.d = d
        if q is not None:
            self.q = q
            
        return self

    def train(self, train_data, val_data=None, **kwargs):
        """
        Train ARIMA model.
        
        Args:
            train_data (pd.Series): Training data
            val_data: Not used for ARIMA
            
        Returns:
            self: Trained model
        """
        try:
            # Create progress indicators
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            status_text.text("Fitting ARIMA model...")
            progress_bar.progress(0.2)
            
            # Ensure data is a pandas Series
            if isinstance(train_data, pd.DataFrame):
                train_data = train_data.iloc[:, 0]
                
            # Fit ARIMA model
            self.model = ARIMA(train_data, order=(self.p, self.d, self.q))
            self.fitted_model = self.model.fit()
            
            progress_bar.progress(1.0)
            status_text.text("ARIMA model training complete!")
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return self
            
        except Exception as e:
            st.error(f"Error training ARIMA model: {str(e)}")
            return None

    def predict(self, steps=30, return_conf_int=True, alpha=0.05):
        """
        Generate forecasts.
        
        Args:
            steps (int): Number of steps to forecast
            return_conf_int (bool): Whether to return confidence intervals
            alpha (float): Significance level for confidence intervals
            
        Returns:
            np.array: Forecasted values
        """
        if self.fitted_model is None:
            st.error("Model must be trained before making predictions.")
            return None
            
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            
            if return_conf_int:
                # Get prediction intervals
                pred = self.fitted_model.get_forecast(steps=steps)
                pred_conf = pred.conf_int(alpha=alpha)
                
                return forecast, pred_conf
            
            return forecast
            
        except Exception as e:
            st.error(f"Error generating ARIMA forecast: {str(e)}")
            return None

    def evaluate(self, test_data):
        """
        Evaluate model on test data.
        
        Args:
            test_data (pd.Series): Test data
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.fitted_model is None:
            st.error("Model must be trained before evaluation.")
            return None
            
        try:
            # Ensure test_data is a pandas Series
            if isinstance(test_data, pd.DataFrame):
                test_data = test_data.iloc[:, 0]
                
            # Generate predictions for test period
            predictions = self.fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - test_data))
            rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
            
            # Handle MAPE with zero values
            if np.any(test_data == 0):
                # Add small epsilon to avoid division by zero
                epsilon = np.finfo(float).eps
                mape = np.mean(np.abs((predictions - test_data) / (test_data + epsilon))) * 100
            else:
                mape = np.mean(np.abs((predictions - test_data) / test_data)) * 100
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            
        except Exception as e:
            st.error(f"Error evaluating ARIMA model: {str(e)}")
            return None

    def get_prediction_intervals(self, steps=30, alpha=0.05):
        """
        Get prediction intervals.
        
        Args:
            steps (int): Number of steps to forecast
            alpha (float): Significance level for intervals
            
        Returns:
            dict: Dictionary with lower and upper bounds
        """
        if self.fitted_model is None:
            st.error("Model must be trained before getting prediction intervals.")
            return None
            
        try:
            pred = self.fitted_model.get_forecast(steps=steps)
            pred_conf = pred.conf_int(alpha=alpha)
            
            return {
                'lower': pred_conf.iloc[:, 0].values,
                'upper': pred_conf.iloc[:, 1].values
            }
            
        except Exception as e:
            st.error(f"Error getting prediction intervals: {str(e)}")
            return None

    def get_parameters(self):
        """Get model parameters."""
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q
        }

    def save(self, path):
        """Save model to file."""
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.fitted_model,
                    'params': {
                        'p': self.p,
                        'd': self.d,
                        'q': self.q
                    }
                }, f)
            st.success(f"Model saved to {path}")
        except Exception as e:
            st.error(f"Error saving ARIMA model: {str(e)}")

    def load(self, path):
        """Load model from file."""
        try:
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.fitted_model = data['model']
                params = data['params']
                self.p = params['p']
                self.d = params['d']
                self.q = params['q']
            st.success(f"Model loaded from {path}")
        except Exception as e:
            st.error(f"Error loading ARIMA model: {str(e)}")


class SARIMAModel(TimeSeriesModel):
    """SARIMA model implementation."""

    def __init__(self, name="SARIMA", p=1, d=1, q=1, P=0, D=0, Q=0, s=12):
        """
        Initialize SARIMA model.
        
        Args:
            name (str): Model name
            p (int): AR order
            d (int): Differencing order
            q (int): MA order
            P (int): Seasonal AR order
            D (int): Seasonal differencing order
            Q (int): Seasonal MA order
            s (int): Seasonal period
        """
        super().__init__(name)
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.fitted_model = None

    def build(self, p=None, d=None, q=None, P=None, D=None, Q=None, s=None):
        """
        Build model with specified parameters.
        
        Args:
            p (int, optional): AR order
            d (int, optional): Differencing order
            q (int, optional): MA order
            P (int, optional): Seasonal AR order
            D (int, optional): Seasonal differencing order
            Q (int, optional): Seasonal MA order
            s (int, optional): Seasonal period
            
        Returns:
            self: Model instance
        """
        if p is not None:
            self.p = p
        if d is not None:
            self.d = d
        if q is not None:
            self.q = q
        if P is not None:
            self.P = P
        if D is not None:
            self.D = D
        if Q is not None:
            self.Q = Q
        if s is not None:
            self.s = s
            
        return self

    def train(self, train_data, val_data=None, **kwargs):
        """
        Train SARIMA model.
        
        Args:
            train_data (pd.Series): Training data
            val_data: Not used for SARIMA
            
        Returns:
            self: Trained model
        """
        try:
            # Create progress indicators
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            status_text.text("Fitting SARIMA model...")
            progress_bar.progress(0.2)
            
            # Ensure data is a pandas Series
            if isinstance(train_data, pd.DataFrame):
                train_data = train_data.iloc[:, 0]
                
            # Fit SARIMA model
            self.model = SARIMAX(
                train_data, 
                order=(self.p, self.d, self.q),
                seasonal_order=(self.P, self.D, self.Q, self.s)
            )
            self.fitted_model = self.model.fit(disp=False)
            
            progress_bar.progress(1.0)
            status_text.text("SARIMA model training complete!")
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return self
            
        except Exception as e:
            st.error(f"Error training SARIMA model: {str(e)}")
            return None

    # The rest of the methods (predict, evaluate, etc.) are similar to ARIMAModel
    # with minor adjustments for SARIMA specifics. You can extend this class with those methods.
    # For brevity, I'm not including them here.

    def predict(self, steps=30, return_conf_int=True, alpha=0.05):
        """
        Generate forecasts.
        
        Args:
            steps (int): Number of steps to forecast
            return_conf_int (bool): Whether to return confidence intervals
            alpha (float): Significance level for confidence intervals
            
        Returns:
            np.array: Forecasted values
        """
        if self.fitted_model is None:
            st.error("Model must be trained before making predictions.")
            return None
            
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            
            if return_conf_int:
                # Get prediction intervals
                pred = self.fitted_model.get_forecast(steps=steps)
                pred_conf = pred.conf_int(alpha=alpha)
                
                return forecast, pred_conf
            
            return forecast
            
        except Exception as e:
            st.error(f"Error generating SARIMA forecast: {str(e)}")
            return None

    def evaluate(self, test_data):
        """
        Evaluate model on test data.
        
        Args:
            test_data (pd.Series): Test data
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.fitted_model is None:
            st.error("Model must be trained before evaluation.")
            return None
            
        try:
            # Ensure test_data is a pandas Series
            if isinstance(test_data, pd.DataFrame):
                test_data = test_data.iloc[:, 0]
                
            # Generate predictions for test period
            predictions = self.fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - test_data))
            rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
            
            # Handle MAPE with zero values
            if np.any(test_data == 0):
                # Add small epsilon to avoid division by zero
                epsilon = np.finfo(float).eps
                mape = np.mean(np.abs((predictions - test_data) / (test_data + epsilon))) * 100
            else:
                mape = np.mean(np.abs((predictions - test_data) / test_data)) * 100
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            
        except Exception as e:
            st.error(f"Error evaluating SARIMA model: {str(e)}")
            return None

    def get_parameters(self):
        """Get model parameters."""
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'P': self.P,
            'D': self.D,
            'Q': self.Q,
            's': self.s
        }


def create_traditional_model(model_type, **kwargs):
    """
    Factory function to create traditional time series models.
    
    Args:
        model_type (str): Type of model ('ARIMA' or 'SARIMA')
        **kwargs: Additional parameters for model construction
        
    Returns:
        TimeSeriesModel: Instance of the requested model
    """
    if model_type.upper() == 'ARIMA':
        return ARIMAModel(**kwargs)
    elif model_type.upper() == 'SARIMA':
        return SARIMAModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")