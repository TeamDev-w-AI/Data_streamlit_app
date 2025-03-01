# In models/machine_learning.py

"""Machine learning model implementations for time series forecasting."""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from .base import TimeSeriesModel


class MLTimeSeriesModel(TimeSeriesModel):
    """Base class for machine learning time series models."""

    def __init__(self, name):
        """
        Initialize ML time series model.
        
        Args:
            name (str): Model name
        """
        super().__init__(name)
        self.model = None
        self.feature_importances = None

    def save(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save model
        """
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            st.success(f"Model saved to {path}")
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")

    def load(self, path):
        """
        Load the model.
        
        Args:
            path (str): Path to load model from
        """
        try:
            import pickle
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            st.success(f"Model loaded from {path}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    def get_feature_importances(self):
        """
        Get feature importances from the model.
        
        Returns:
            pd.DataFrame: DataFrame with feature importances
        """
        if self.model is None or self.feature_importances is None:
            return None
            
        return self.feature_importances


class RandomForestModel(MLTimeSeriesModel):
    """Random Forest regressor for time series forecasting."""

    def __init__(self, name="RandomForest", n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize Random Forest model.
        
        Args:
            name (str): Model name
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
            random_state (int): Random state for reproducibility
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def build(self, n_estimators=None, max_depth=None, min_samples_split=2, random_state=None):
        """
        Build the model with specified parameters.
        
        Args:
            n_estimators (int, optional): Number of trees
            max_depth (int, optional): Maximum depth of trees
            min_samples_split (int): Minimum samples to split an internal node
            random_state (int, optional): Random state for reproducibility
            
        Returns:
            self: Model instance
        """
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if max_depth is not None:
            self.max_depth = max_depth
        if random_state is not None:
            self.random_state = random_state
            
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        return self


    def train(self, train_data, **kwargs):
        """
        Train the model.
        
        Args:
            train_data: Training data (X_train, y_train) tuple, nested tuple from prepare_ml_data, or pd.DataFrame
            **kwargs: Additional training parameters
            
        Returns:
            self: Trained model
        """
        try:
            # Create progress indicators
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            status_text.text("Training Random Forest model...")
            progress_bar.progress(0.2)
            
            # Handle different input formats
            if isinstance(train_data, tuple):
                if len(train_data) == 2:
                    # Check if this is a nested tuple from prepare_ml_data
                    if isinstance(train_data[0], tuple) and len(train_data[0]) == 2:
                        # Unpack the first tuple which contains (X_train, y_train)
                        X_train, y_train = train_data[0]
                    else:
                        # Direct (X_train, y_train) tuple
                        X_train, y_train = train_data
                else:
                    st.error("Tuple format not recognized. Expected (X_train, y_train) or ((X_train, y_train), (X_test, y_test))")
                    return None
            elif isinstance(train_data, pd.DataFrame):
                # Assume last column is target or use target_col if specified
                target_col = kwargs.get('target_col', train_data.columns[-1])
                X_train = train_data.drop(columns=[target_col])
                y_train = train_data[target_col]
            else:
                st.error("Unsupported data format. Expected tuple or DataFrame")
                return None
                
            # Debug information
            st.write("X_train shape:", X_train.shape if hasattr(X_train, 'shape') else "no shape")
            st.write("X_train dtypes:", X_train.dtypes if hasattr(X_train, 'dtypes') else "no dtypes")
            st.write("y_train shape:", y_train.shape if hasattr(y_train, 'shape') else "no shape")
            
            # Train the model
            self.model.fit(X_train, y_train)
        
            
            # Extract feature importances
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False)
            
            progress_bar.progress(1.0)
            status_text.text("Random Forest model training complete!")
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return self
            
        except Exception as e:
            st.error(f"Error training Random Forest model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def predict(self, X):
        """
        Generate predictions.
        
        Args:
            X: Input features
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            st.error("Model must be trained before making predictions.")
            return None
            
        try:
            return self.model.predict(X)
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return None

    def evaluate(self, test_data):
        """
        Evaluate the model.
        
        Args:
            test_data: Test data (X_test, y_test) tuple or DataFrame
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            st.error("Model must be trained before evaluation.")
            return None
            
        try:
            # Handle different input formats
            if isinstance(test_data, tuple) and len(test_data) == 2:
                X_test, y_test = test_data
            elif isinstance(test_data, pd.DataFrame):
                # Assume last column is target or use target_col if provided in model context
                if hasattr(self, 'target_col') and self.target_col in test_data.columns:
                    target_col = self.target_col
                else:
                    target_col = test_data.columns[-1]
                X_test = test_data.drop(columns=[target_col])
                y_test = test_data[target_col]
            else:
                st.error("Unsupported data format for evaluation")
                return None
            
            # Generate predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_pred - y_test))
            rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
            
            # Handle MAPE with zero values
            if np.any(y_test == 0):
                # Add small epsilon to avoid division by zero
                epsilon = np.finfo(float).eps
                mape = np.mean(np.abs((y_pred - y_test) / (y_test + epsilon))) * 100
            else:
                mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
            
            # Store predictions for later analysis
            self.test_predictions = y_pred
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            
        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")
            return None

    def get_parameters(self):
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }


class XGBoostModel(MLTimeSeriesModel):
    """XGBoost regressor for time series forecasting."""

    def __init__(self, name="XGBoost", n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8):
        """
        Initialize XGBoost model.
        
        Args:
            name (str): Model name
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Learning rate
            max_depth (int): Maximum tree depth
            subsample (float): Subsample ratio
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

    def build(self, n_estimators=None, learning_rate=None, max_depth=None, subsample=None):
        """
        Build the model with specified parameters.
        
        Args:
            n_estimators (int, optional): Number of boosting rounds
            learning_rate (float, optional): Learning rate
            max_depth (int, optional): Maximum tree depth
            subsample (float, optional): Subsample ratio
            
        Returns:
            self: Model instance
        """
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if max_depth is not None:
            self.max_depth = max_depth
        if subsample is not None:
            self.subsample = subsample
            
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            n_jobs=-1
        )
        
        return self

    # The train, predict, evaluate methods are similar to RandomForestModel
    # For brevity, I'll provide one example method and you can implement the rest similarly

    def train(self, train_data, **kwargs):
        """
        Train the model.
        
        Args:
            train_data: Training data (X_train, y_train) tuple or pd.DataFrame
            **kwargs: Additional training parameters
            
        Returns:
            self: Trained model
        """
        try:
            # Create progress indicators
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            status_text.text("Training XGBoost model...")
            progress_bar.progress(0.2)
            
            # Handle different input formats
            if isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data
            elif isinstance(train_data, pd.DataFrame):
                target_col = kwargs.get('target_col', train_data.columns[-1])
                X_train = train_data.drop(columns=[target_col])
                y_train = train_data[target_col]
            else:
                st.error("Unsupported data format. Expected (X_train, y_train) tuple or DataFrame")
                return None
                
            # Add data validation before training the model
            if X_train is None or y_train is None:
                st.error("Features or target data is None")
                return None
                
            # Verify shapes are compatible
            if len(X_train) != len(y_train):
                st.error(f"Shape mismatch: X_train has {len(X_train)} samples but y_train has {len(y_train)}")
                return None

            # Verify no NaN values in training data
            if hasattr(X_train, 'isna') and X_train.isna().any().any():
                st.warning(f"X_train contains {X_train.isna().sum().sum()} NaN values which may affect model performance")
                
            if hasattr(y_train, 'isna') and y_train.isna().any():
                st.error(f"y_train contains {y_train.isna().sum()} NaN values. Please handle missing values first.")
                return None
                
            # Check for infinite values
            if hasattr(X_train, 'select_dtypes'):
                numeric_cols = X_train.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if np.isinf(X_train[col]).any():
                        st.warning(f"Column {col} contains infinite values which may affect model performance")
            
            # Train the model - FIXED INDENTATION HERE
            self.model.fit(X_train, y_train)
            
            # Extract feature importances
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False)
            
            progress_bar.progress(1.0)
            status_text.text("XGBoost model training complete!")
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return self
            
        except Exception as e:
            st.error(f"Error training XGBoost model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def get_parameters(self):
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample
        }


def create_ml_model(model_type, **kwargs):
    """
    Factory function to create machine learning models.
    
    Args:
        model_type (str): Type of model ('Random Forest' or 'XGBoost')
        **kwargs: Additional parameters for model construction
        
    Returns:
        MLTimeSeriesModel: Instance of the requested model
    """
    if model_type.upper() == 'RANDOM FOREST':
        return RandomForestModel(**kwargs)
    elif model_type.upper() == 'XGBOOST':
        return XGBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")