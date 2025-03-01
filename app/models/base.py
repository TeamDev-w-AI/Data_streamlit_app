"""Base model class for time series forecasting."""

from abc import ABC, abstractmethod


class TimeSeriesModel(ABC):
    """Abstract base class for all time series models."""

    def __init__(self, name):
        """
        Initialize the time series model.
        
        Args:
            name (str): Name of the model
        """
        self.name = name
        self.model = None

    @abstractmethod
    def build(self, **kwargs):
        """
        Build the model architecture.
        
        Args:
            **kwargs: Additional parameters for model configuration
            
        Returns:
            self: The model instance
        """
        pass

    @abstractmethod
    def train(self, train_data, val_data=None, **kwargs):
        """
        Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        pass

    @abstractmethod
    def evaluate(self, test_data):
        """
        Evaluate the model.
        
        Args:
            test_data: Test data
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load the model.
        
        Args:
            path (str): Path to load the model from
        """
        pass