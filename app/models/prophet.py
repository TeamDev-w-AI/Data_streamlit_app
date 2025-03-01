"""Facebook Prophet model implementation."""

import pandas as pd
import numpy as np
from prophet import Prophet
from .base import TimeSeriesModel
import streamlit as st


class ProphetModel(TimeSeriesModel):
    """Facebook Prophet model implementation."""

    def __init__(self, name="Prophet"):
        super().__init__(name)
        self.model = None
        self.forecast = None
        self.forecast_df = None
        self.changepoints = None
        self.seasonality_mode = "additive"
        self.growth = "linear"
        self.interval_width = 0.95
        self.trained = False

    def build(self, 
              seasonality_mode="additive", 
              growth="linear",
              changepoints=None,
              yearly_seasonality="auto",
              weekly_seasonality="auto",
              daily_seasonality="auto",
              interval_width=0.95):
        """
        Build Prophet model with specified parameters.
        
        Args:
            seasonality_mode (str): 'additive' or 'multiplicative'
            growth (str): 'linear' or 'logistic'
            changepoints (list): List of dates at which to include potential changepoints
            yearly_seasonality: 'auto', True, False, or number of Fourier terms
            weekly_seasonality: 'auto', True, False, or number of Fourier terms
            daily_seasonality: 'auto', True, False, or number of Fourier terms
            interval_width (float): Width of the uncertainty intervals
        """
        # Validate inputs
        if seasonality_mode not in ["additive", "multiplicative"]:
            st.warning(f"Invalid seasonality_mode: {seasonality_mode}. Using 'additive' instead.")
            seasonality_mode = "additive"
            
        if growth not in ["linear", "logistic"]:
            st.warning(f"Invalid growth: {growth}. Using 'linear' instead.")
            growth = "linear"
            
        if not (0 < interval_width < 1):
            st.warning(f"Invalid interval_width: {interval_width}. Using 0.95 instead.")
            interval_width = 0.95
        
        self.seasonality_mode = seasonality_mode
        self.growth = growth
        self.changepoints = changepoints
        self.interval_width = interval_width
        
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            growth=growth,
            changepoints=changepoints,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=interval_width
        )
        
        return self
    
    def add_country_holidays(self, country_name):
        """Add country-specific holidays to the model."""
        try:
            self.model.add_country_holidays(country_name=country_name)
            st.success(f"Added holidays for {country_name}")
        except Exception as e:
            st.error(f"Error adding holidays: {str(e)}")
    
    def add_seasonality(self, name, period, fourier_order):
        """Add custom seasonality to the model."""
        try:
            self.model.add_seasonality(
                name=name,
                period=period,
                fourier_order=fourier_order
            )
            st.success(f"Added {name} seasonality with period {period}")
        except Exception as e:
            st.error(f"Error adding seasonality: {str(e)}")

    def train(self, train_data, val_data=None, **kwargs):
        """
        Train Prophet model.
        
        Args:
            train_data (pd.DataFrame): Training data with 'ds' (datetime) and 'y' (target) columns
            val_data: Not used for Prophet
            
        Returns:
            self: Trained model
        """
        try:
            # Ensure data is in the right format
            if not isinstance(train_data, pd.DataFrame):
                if isinstance(train_data, tuple) and len(train_data) == 2:
                    # Convert from (X_train, y_train) format
                    X_train, y_train = train_data
                    if isinstance(X_train, pd.DataFrame) and X_train.index.is_all_dates:
                        train_data = pd.DataFrame({
                            'ds': X_train.index,
                            'y': y_train
                        })
                    else:
                        st.error("Prophet requires datetime index")
                        return None
                else:
                    st.error("Unsupported data format for Prophet")
                    return None
            
            # If train_data is DataFrame but doesn't have 'ds' and 'y' columns
            if not all(col in train_data.columns for col in ['ds', 'y']):
                if isinstance(train_data.index, pd.DatetimeIndex):
                    # Assume the first column is the target if not specified
                    target_col = kwargs.get('target_col', train_data.columns[0])
                    train_data = pd.DataFrame({
                        'ds': train_data.index,
                        'y': train_data[target_col]
                    })
                else:
                    st.error("Prophet requires datetime index or 'ds' column")
                    return None

            # Handle additional regressors if provided
            regressors = kwargs.get('regressors', None)
            if regressors:
                for regressor in regressors:
                    self.model.add_regressor(regressor)

            # Handle custom cap and floor for logistic growth
            if self.growth == 'logistic':
                if 'cap' not in train_data.columns:
                    cap = kwargs.get('cap', train_data['y'].max() * 1.5)
                    train_data['cap'] = cap
                    st.info(f"Using automatic cap: {cap}")
                
                if 'floor' not in train_data.columns and kwargs.get('use_floor', False):
                    floor = kwargs.get('floor', train_data['y'].min() * 0.5)
                    train_data['floor'] = floor
                    st.info(f"Using automatic floor: {floor}")
            
            # Fit the model
            with st.spinner("Training Prophet model..."):
                self.model.fit(train_data)
                self.trained = True
                
            st.success("Prophet model trained successfully!")
            return self
            
        except Exception as e:
            st.error(f"Error training Prophet model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def predict(self, periods=30, freq='D', include_history=True, regressors=None):
        """
        Generate forecasts.
        
        Args:
            periods (int): Number of periods to forecast
            freq (str): Frequency of forecast (D=daily, W=weekly, M=monthly, etc.)
            include_history (bool): Include the historical data in the forecast
            regressors (pd.DataFrame): Additional regressors for the forecast period
            
        Returns:
            pd.DataFrame: Forecast dataframe
        """
        if not self.trained or self.model is None:
            st.error("Model must be trained before making predictions.")
            return None
            
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods, 
                freq=freq,
                include_history=include_history
            )
            
            # Add regressor values if provided
            if regressors is not None:
                for col in regressors.columns:
                    future[col] = regressors[col]
                    
            # For logistic growth, ensure cap and floor are in future df
            if self.growth == 'logistic':
                if 'cap' not in future.columns and hasattr(self.model, 'train_cap'):
                    future['cap'] = self.model.train_cap
                elif 'cap' not in future.columns:
                    # Set a default cap if not provided
                    st.warning("Logistic growth requires a cap. Using default cap of 1.5x the max historical value.")
                    future['cap'] = 1.5 * self.model.history['y'].max()
                    
                if 'floor' not in future.columns and hasattr(self.model, 'train_floor'):
                    future['floor'] = self.model.train_floor
                    
            # Make prediction
            self.forecast_df = self.model.predict(future)
            return self.forecast_df
            
        except Exception as e:
            st.error(f"Error in Prophet prediction: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def evaluate(self, test_data):
        """
        Evaluate model on test data.
        
        Args:
            test_data (pd.DataFrame): Test data with 'ds' and 'y' columns
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not self.trained or self.model is None:
            st.error("Model must be trained before evaluation.")
            return None
            
        try:
            # Ensure test_data is in the right format
            if not isinstance(test_data, pd.DataFrame):
                if isinstance(test_data, tuple) and len(test_data) == 2:
                    # Convert from (X_test, y_test) format
                    X_test, y_test = test_data
                    if isinstance(X_test, pd.DataFrame) and X_test.index.is_all_dates:
                        test_data = pd.DataFrame({
                            'ds': X_test.index,
                            'y': y_test
                        })
                    else:
                        test_data = pd.DataFrame({
                            'ds': pd.date_range(start='2020-01-01', periods=len(y_test)),
                            'y': y_test
                        })
                else:
                    st.error("Unsupported test_data format for Prophet evaluation.")
                    return None
            
            # Check if required columns exist
            if not all(col in test_data.columns for col in ['ds', 'y']):
                st.error("Test data must contain 'ds' and 'y' columns.")
                return None
                
            # Make predictions on test period
            if self.forecast_df is not None:
                # Check if test dates are in the forecast
                test_ds = pd.to_datetime(test_data['ds'])
                forecast_ds = pd.to_datetime(self.forecast_df['ds'])
                
                # Find matching dates
                matching_dates = test_ds[test_ds.isin(forecast_ds)]
                
                if len(matching_dates) > 0:
                    # Extract predictions for test dates
                    predictions = self.forecast_df[self.forecast_df['ds'].isin(matching_dates)]['yhat'].values
                    actuals = test_data[test_data['ds'].isin(matching_dates)]['y'].values
                else:
                    # Need to regenerate predictions for test period
                    future = pd.DataFrame({'ds': test_data['ds']})
                    
                    # For logistic growth, ensure cap and floor are included
                    if self.growth == 'logistic':
                        if 'cap' in test_data.columns:
                            future['cap'] = test_data['cap']
                        elif hasattr(self.model, 'train_cap'):
                            future['cap'] = self.model.train_cap
                        else:
                            future['cap'] = test_data['y'].max() * 1.5
                            
                        if 'floor' in test_data.columns:
                            future['floor'] = test_data['floor']
                        elif hasattr(self.model, 'train_floor'):
                            future['floor'] = self.model.train_floor
                    
                    predictions = self.model.predict(future)['yhat'].values
                    actuals = test_data['y'].values
            else:
                # Generate predictions if not already done
                future = pd.DataFrame({'ds': test_data['ds']})
                
                # Handle logistic growth
                if self.growth == 'logistic':
                    if 'cap' in test_data.columns:
                        future['cap'] = test_data['cap']
                    elif hasattr(self.model, 'train_cap'):
                        future['cap'] = self.model.train_cap
                    else:
                        future['cap'] = test_data['y'].max() * 1.5
                        
                    if 'floor' in test_data.columns:
                        future['floor'] = test_data['floor']
                    elif hasattr(self.model, 'train_floor'):
                        future['floor'] = self.model.train_floor
                
                predictions = self.model.predict(future)['yhat'].values
                actuals = test_data['y'].values
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            
            # Handle MAPE with zero values
            if np.any(actuals == 0):
                # Add small epsilon to avoid division by zero
                epsilon = np.finfo(float).eps
                mape = np.mean(np.abs((predictions - actuals) / (actuals + epsilon))) * 100
            else:
                mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            
        except Exception as e:
            st.error(f"Error evaluating Prophet model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def plot_components(self):
        """
        Plot the components of the forecast.
        
        Returns:
            tuple: (fig1, fig2) matplotlib figures for components and forecast
        """
        try:
            if self.forecast_df is None:
                st.warning("No forecast available. Run predict() first.")
                return None, None
                
            # Create the plots
            fig1 = self.model.plot(self.forecast_df)
            fig2 = self.model.plot_components(self.forecast_df)
            
            return fig1, fig2
            
        except Exception as e:
            st.error(f"Error plotting Prophet components: {str(e)}")
            return None, None
    
    def get_parameters(self):
        """Get the model's parameters."""
        params = {
            'seasonality_mode': self.seasonality_mode,
            'growth': self.growth,
            'interval_width': self.interval_width
        }
        
        # Add additional parameters from the model
        if self.model:
            for key in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality']:
                if hasattr(self.model, key):
                    params[key] = getattr(self.model, key)
        
        return params

    def save(self, path):
        """Save model to file."""
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            st.success(f"Model saved to {path}")
        except Exception as e:
            st.error(f"Error saving Prophet model: {str(e)}")

    def load(self, path):
        """Load model from file."""
        try:
            import pickle
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            st.success(f"Model loaded from {path}")
            self.trained = True
        except Exception as e:
            st.error(f"Error loading Prophet model: {str(e)}")


def create_prophet_model(**kwargs):
    """Factory function to create a Prophet model."""
    return ProphetModel(**kwargs)