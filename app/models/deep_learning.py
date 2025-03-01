"""Deep learning model implementations."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
import numpy as np

from .base import TimeSeriesModel


class DeepLearningModel(TimeSeriesModel):
    """Base class for deep learning models."""

    def __init__(self, name, sequence_length, n_features=1):
        """
        Initialize deep learning model.
        
        Args:
            name (str): Model name
            sequence_length (int): Length of input sequences
            n_features (int): Number of features
        """
        super().__init__(name)
        self.sequence_length = sequence_length
        self.n_features = n_features

    def save(self, path):
        """
        Save the Keras model.
        
        Args:
            path (str): Path to save model
        """
        try:
            self.model.save(path)
            st.success(f"Model saved to {path}")
        except Exception as e:
            st.error(f"Error saving model: {e}")

    def load(self, path):
        """
        Load the Keras model.
        
        Args:
            path (str): Path to load model from
        """
        try:
            self.model = keras.models.load_model(path)
            st.success(f"Model loaded from {path}")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    def train(self, train_data, val_data=None, epochs=10, callbacks=None):
        """
        Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            epochs (int): Number of epochs
            callbacks (list): List of callbacks
            
        Returns:
            History object
        """
        try:
            return self.model.fit(
                train_data,
                epochs=epochs,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=0
            )
        except Exception as e:
            st.error(f"Error training model: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def evaluate(self, test_data):
        """
        Evaluate the model.
        
        Args:
            test_data: Test data
            
        Returns:
            float or list: Evaluation metrics
        """
        try:
            return self.model.evaluate(test_data, verbose=0)
        except Exception as e:
            st.error(f"Error evaluating model: {e}")
            return None
            
    def predict(self, data):
        """
        Generate predictions.
        
        Args:
            data: Input data for prediction
            
        Returns:
            np.array: Predictions
        """
        try:
            return self.model.predict(data, verbose=0)
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            return None


class SimpleRNNModel(DeepLearningModel):
    """Simple RNN model implementation."""

    def build(self, units=64, dropout_rate=0.1):
        """
        Build Simple RNN model.
        
        Args:
            units (int): Number of RNN units
            dropout_rate (float): Dropout rate
            
        Returns:
            self: Model instance
        """
        try:
            inputs = keras.Input(shape=(self.sequence_length, self.n_features))
            x = layers.SimpleRNN(units, dropout=dropout_rate)(inputs)
            outputs = layers.Dense(1)(x)
            self.model = keras.Model(inputs, outputs)
            return self
        except Exception as e:
            st.error(f"Error building RNN model: {e}")
            return None


class LSTMModel(DeepLearningModel):
    """LSTM model implementation."""

    def build(self, units=64, dropout_rate=0.1):
        """
        Build LSTM model.
        
        Args:
            units (int): Number of LSTM units
            dropout_rate (float): Dropout rate
            
        Returns:
            self: Model instance
        """
        try:
            inputs = keras.Input(shape=(self.sequence_length, self.n_features))
            x = layers.LSTM(units, dropout=dropout_rate)(inputs)
            outputs = layers.Dense(1)(x)
            self.model = keras.Model(inputs, outputs)
            return self
        except Exception as e:
            st.error(f"Error building LSTM model: {e}")
            return None


class StackedModel(DeepLearningModel):
    """Stacked LSTM+RNN model implementation."""

    def build(self, lstm_units=128, rnn_units=64, dropout_rate=0.1):
        """
        Build stacked LSTM+RNN model.
        
        Args:
            lstm_units (int): Number of LSTM units
            rnn_units (int): Number of RNN units
            dropout_rate (float): Dropout rate
            
        Returns:
            self: Model instance
        """
        try:
            inputs = keras.Input(shape=(self.sequence_length, self.n_features))
            x = layers.LSTM(lstm_units, dropout=dropout_rate, return_sequences=True)(inputs)
            x = layers.SimpleRNN(rnn_units, dropout=dropout_rate)(x)
            x = layers.Dropout(dropout_rate)(x)
            outputs = layers.Dense(1)(x)
            self.model = keras.Model(inputs, outputs)
            return self
        except Exception as e:
            st.error(f"Error building stacked model: {e}")
            return None


def create_dl_model(model_type, **kwargs):
    """
    Factory function to create deep learning models.
    
    Args:
        model_type (str): Type of model ('SimpleRNN', 'LSTM', or 'StackedLSTMRNN')
        **kwargs: Additional parameters for model construction
        
    Returns:
        DeepLearningModel: Instance of the requested model
    """
    if model_type.upper() == 'SIMPLE RNN':
        return SimpleRNNModel(**kwargs)
    elif model_type.upper() == 'LSTM':
        return LSTMModel(**kwargs)
    elif model_type.upper() == 'STACKED LSTM+RNN':
        return StackedModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")