# models/__init__.py
"""Model implementations for time series forecasting."""

# Import for easier access
from .base import TimeSeriesModel
from .deep_learning import SimpleRNNModel, LSTMModel, StackedModel, create_dl_model
from .trainer import ModelTrainer

# Import other model types if available
try:
    from .traditional import ARIMAModel, SARIMAModel, create_traditional_model
except ImportError:
    pass

try:
    from .machine_learning import RandomForestModel, XGBoostModel, create_ml_model
except ImportError:
    pass

try:
    from .prophet import ProphetModel, create_prophet_model
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from .autogluon import AutoGluonModel, create_autogluon_model
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False