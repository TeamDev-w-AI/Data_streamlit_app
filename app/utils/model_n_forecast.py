"""Module containing the modeling and forecasting functions."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json
import traceback

# Import from your project structure
from config import (
    MODEL_CATEGORIES, TRADITIONAL_MODELS, ML_MODELS, DL_MODELS, 
    PROPHET_MODELS, AUTOGLUON_MODELS, DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_RNN_UNITS, DEFAULT_DROPOUT_RATE, DEFAULT_VAL_SPLIT,
    DEFAULT_TEST_SPLIT, PROPHET_SEASONALITY_MODES,
    PROPHET_GROWTH_MODELS, PROPHET_SEASONALITY_OPTIONS,
    AUTOGLUON_FREQUENCIES, AUTOGLUON_DEFAULT_TIME_LIMIT
)

from data.preprocessor import TimeSeriesPreprocessor
from utils.visualization import DataVisualizer
from models.deep_learning import SimpleRNNModel, LSTMModel, StackedModel, create_dl_model
from models.trainer import ModelTrainer

# Import optional models with fallback
try:
    from models.traditional import ARIMAModel, SARIMAModel, create_traditional_model
except ImportError:
    def create_traditional_model(*args, **kwargs):
        st.error("Traditional models not available. Missing dependencies.")
        return None

try:
    from models.machine_learning import RandomForestModel, XGBoostModel, create_ml_model
except ImportError:
    def create_ml_model(*args, **kwargs):
        st.error("Machine learning models not available. Missing dependencies.")
        return None

try:
    from models.prophet import ProphetModel, create_prophet_model
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    def create_prophet_model(*args, **kwargs):
        st.error("Prophet not available. Please install prophet package.")
        return None

try:
    from models.autogluon import AutoGluonModel, create_autogluon_model
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    def create_autogluon_model(*args, **kwargs):
        st.error("AutoGluon not available. Please install autogluon.timeseries package.")
        return None


def modeling_and_forecasting():
    """Third stage: Model selection, training, and forecasting."""
    st.markdown('<div class="section-header"><h2>🤖 Modeling & Forecasting</h2></div>', unsafe_allow_html=True)

    if st.session_state.preprocessed_data is None:
        st.warning("No preprocessed data available. Please complete the Data Preprocessing stage first.")
        if st.button("Go to Data Preprocessing"):
            st.session_state.current_stage = "Data Preprocessing"
            st.rerun()
        return

    df = st.session_state.preprocessed_data
    target_column = st.session_state.target_column
    
    # Create tabs for different modeling steps
    tabs = st.tabs(["Model Selection", "Training & Validation", "Forecasting", "Evaluation"])
    
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Select Models for Time Series Forecasting")
        
        # Model category selection
        model_category = st.selectbox(
            "Select Model Category",
            MODEL_CATEGORIES,
            help="Choose the type of models you want to use for forecasting"
        )
        
        # Model options based on category
        if model_category == 'Traditional':
            available_models = TRADITIONAL_MODELS
        elif model_category == 'Machine Learning':
            available_models = ML_MODELS
        elif model_category == 'Deep Learning':
            available_models = DL_MODELS
        elif model_category == 'Prophet':
            available_models = PROPHET_MODELS
            if not PROPHET_AVAILABLE:
                st.warning("Prophet is not installed. Please install it with `pip install prophet`")
        elif model_category == 'AutoGluon':
            available_models = AUTOGLUON_MODELS
            if not AUTOGLUON_AVAILABLE:
                st.warning("AutoGluon is not installed. Please install it with `pip install autogluon.timeseries`")
        
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose a specific model for forecasting"
        )
        
        # Store selected model in session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = selected_model
        else:
            st.session_state.selected_model = selected_model
        
        # Configure model hyperparameters based on selection
        st.subheader("Configure Model Hyperparameters")
        
        # Dictionary to store hyperparameters
        if 'hyperparameters' not in st.session_state:
            st.session_state.hyperparameters = {}
        
        # Configure hyperparameters based on selected model
        if model_category == 'Traditional':
            if selected_model == 'ARIMA':
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.number_input("p (AR order)", min_value=0, max_value=10, value=1)
                with col2:
                    d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
                with col3:
                    q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1)
                
                st.session_state.hyperparameters = {
                    'p': p, 'd': d, 'q': q
                }
            
            elif selected_model == 'SARIMA':
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.number_input("p (AR order)", min_value=0, max_value=10, value=1)
                    P = st.number_input("P (Seasonal AR)", min_value=0, max_value=10, value=1)
                with col2:
                    d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
                    D = st.number_input("D (Seasonal Diff)", min_value=0, max_value=2, value=1)
                with col3:
                    q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1)
                    Q = st.number_input("Q (Seasonal MA)", min_value=0, max_value=10, value=1)
                
                s = st.number_input("s (Seasonal Period)", min_value=1, max_value=52, value=12)
                
                st.session_state.hyperparameters = {
                    'p': p, 'd': d, 'q': q,
                    'P': P, 'D': D, 'Q': Q, 's': s
                }
        
        elif model_category == 'Machine Learning':
            if selected_model == 'Random Forest':
                col1, col2 = st.columns(2)
                with col1:
                    n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
                    max_depth = st.slider("Max Depth", min_value=1, max_value=30, value=10)
                with col2:
                    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2)
                    random_state = st.slider("Random State", min_value=0, max_value=100, value=42)
                
                st.session_state.hyperparameters = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'random_state': random_state
                }
            
            elif selected_model == 'XGBoost':
                col1, col2 = st.columns(2)
                with col1:
                    n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
                    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                with col2:
                    max_depth = st.slider("Max Depth", min_value=1, max_value=15, value=6)
                    subsample = st.slider("Subsample Ratio", min_value=0.5, max_value=1.0, value=0.8, step=0.1)
                
                st.session_state.hyperparameters = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'subsample': subsample
                }
        
        elif model_category == 'Deep Learning':
            col1, col2 = st.columns(2)
            with col1:
                sequence_length = st.slider("Sequence Length", min_value=5, max_value=100, value=DEFAULT_SEQUENCE_LENGTH)
                epochs = st.slider("Number of Epochs", min_value=10, max_value=100, value=DEFAULT_EPOCHS)
            with col2:
                batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=DEFAULT_BATCH_SIZE, step=8)
                learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=DEFAULT_LEARNING_RATE, format="%.4f")
            
            if selected_model in ['Simple RNN', 'LSTM']:
                units = st.slider("Hidden Units", min_value=16, max_value=256, value=DEFAULT_RNN_UNITS, step=16)
                dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=DEFAULT_DROPOUT_RATE)
                
                st.session_state.hyperparameters = {
                    'sequence_length': sequence_length,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'units': units,
                    'dropout_rate': dropout_rate
                }
            
            elif selected_model == 'Stacked LSTM+RNN':
                lstm_units = st.slider("LSTM Units", min_value=32, max_value=256, value=128, step=32)
                rnn_units = st.slider("RNN Units", min_value=16, max_value=128, value=64, step=16)
                dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=DEFAULT_DROPOUT_RATE)
                
                st.session_state.hyperparameters = {
                    'sequence_length': sequence_length,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'lstm_units': lstm_units,
                    'rnn_units': rnn_units,
                    'dropout_rate': dropout_rate
                }
        
        elif model_category == 'Prophet' and PROPHET_AVAILABLE:
            col1, col2 = st.columns(2)
            with col1:
                seasonality_mode = st.selectbox(
                    "Seasonality Mode",
                    PROPHET_SEASONALITY_MODES,
                    index=0
                )
                yearly_seasonality = st.selectbox(
                    "Yearly Seasonality",
                    PROPHET_SEASONALITY_OPTIONS,
                    index=0
                )
            with col2:
                growth = st.selectbox(
                    "Growth Model",
                    PROPHET_GROWTH_MODELS,
                    index=0
                )
                weekly_seasonality = st.selectbox(
                    "Weekly Seasonality",
                    PROPHET_SEASONALITY_OPTIONS,
                    index=0
                )
            
            include_holidays = st.checkbox("Include Country Holidays", value=False)
            if include_holidays:
                country_name = st.selectbox(
                    "Select Country",
                    ["US", "UK", "CA", "DE", "FR", "JP", "CN", "BR", "IN", "RU", "AU"]
                )
            else:
                country_name = None
            
            st.session_state.hyperparameters = {
                'seasonality_mode': seasonality_mode,
                'growth': growth,
                'yearly_seasonality': yearly_seasonality,
                'weekly_seasonality': weekly_seasonality,
                'include_holidays': include_holidays,
                'country_name': country_name
            }
        
        elif model_category == 'AutoGluon' and AUTOGLUON_AVAILABLE:
            col1, col2 = st.columns(2)
            with col1:
                prediction_length = st.slider("Prediction Length", min_value=1, max_value=100, value=30)
                time_limit = st.slider("Time Limit (seconds)", min_value=60, max_value=900, value=AUTOGLUON_DEFAULT_TIME_LIMIT, step=60)
            with col2:
                freq = st.selectbox(
                    "Data Frequency",
                    AUTOGLUON_FREQUENCIES,
                    index=0
                )
                enable_ensemble = st.checkbox("Enable Model Ensemble", value=True)
            
            st.session_state.hyperparameters = {
                'prediction_length': prediction_length,
                'freq': freq,
                'time_limit': time_limit,
                'enable_ensemble': enable_ensemble
            }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data split settings
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Split Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            train_size = st.slider("Training Data Size (%)", min_value=50, max_value=90, value=70)
            val_size = st.slider("Validation Data Size (%)", min_value=0, max_value=30, value=int(DEFAULT_VAL_SPLIT * 100))
        with col2:
            test_size = st.slider("Test Data Size (%)", min_value=10, max_value=40, value=int(DEFAULT_TEST_SPLIT * 100), step=1)
            st.write(f"Split Ratio: {train_size}/{val_size}/{test_size}")
        
        # Ensure splits add up to 100%
        total = train_size + val_size + test_size
        if total != 100:
            st.warning(f"Split percentages sum to {total}%. Please adjust to total 100%.")
        
        st.session_state.split_config = {
            'train_size': train_size / 100,
            'val_size': val_size / 100,
            'test_size': test_size / 100
        }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Forecast horizons
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Forecast Horizon")
        
        forecast_horizon = st.slider("Forecast Horizon (periods ahead)", min_value=1, max_value=100, value=30)
        st.session_state.forecast_horizon = forecast_horizon
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Train and Validate Model")
        
        if 'selected_model' not in st.session_state:
            st.warning("Please select a model in the Model Selection tab first.")
        else:
            model_category = next((category for category, models in zip(
                MODEL_CATEGORIES, 
                [TRADITIONAL_MODELS, ML_MODELS, DL_MODELS, PROPHET_MODELS, AUTOGLUON_MODELS]
            ) if st.session_state.selected_model in models), None)
            
            st.write(f"Selected Model: **{st.session_state.selected_model}** (Category: {model_category})")
            
            # Display current hyperparameters
            if 'hyperparameters' in st.session_state:
                with st.expander("Model Hyperparameters", expanded=False):
                    st.json(st.session_state.hyperparameters)
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner(f"Training {st.session_state.selected_model} model..."):
                    # Prepare data splits
                    try:
                        # Split data based on configuration
                        train_end_idx = int(len(df) * st.session_state.split_config['train_size'])
                        val_end_idx = train_end_idx + int(len(df) * st.session_state.split_config['val_size'])
                        
                        train_data = df[:train_end_idx]
                        val_data = df[train_end_idx:val_end_idx] if st.session_state.split_config['val_size'] > 0 else None
                        test_data = df[val_end_idx:]
                        
                        # Store data splits in session state
                        st.session_state.train_data = train_data
                        st.session_state.val_data = val_data
                        st.session_state.test_data = test_data
                        
                        # Initialize preprocessing depending on model type
                        preprocessor = TimeSeriesPreprocessor()
                        
                        # Initialize and train model based on category and type
                        if model_category == 'Traditional':
                            # Traditional models like ARIMA, SARIMA
                            model = create_traditional_model(
                                st.session_state.selected_model, 
                                **st.session_state.hyperparameters
                            )
                            
                            # Prepare features if needed
                            features = preprocessor.create_features(
                                train_data, 
                                target_column
                            )
                            
                            # Train the model
                            train_result = model.train(
                                train_data[target_column], 
                                val_data=val_data[target_column] if val_data is not None else None
                            )
                            
                        elif model_category == 'Machine Learning':
                            # ML models like Random Forest, XGBoost
                            init_params = {k: v for k, v in st.session_state.hyperparameters.items() 
                                        if k in ['name', 'n_estimators', 'max_depth', 'random_state']}
                            model = create_ml_model(st.session_state.selected_model, **init_params)

                            # Then call build with the remaining parameters
                            build_params = {k: v for k, v in st.session_state.hyperparameters.items() 
                                            if k in ['n_estimators', 'max_depth', 'min_samples_split', 'random_state']}
                            model.build(**build_params)
                            
                            # Create features
                            features = preprocessor.create_features(
                                train_data, 
                                target_column
                            )

                            # Check feature quality before model training
                            if features is not None:
                                # Log feature statistics
                                st.write(f"Feature set contains {len(features.columns)} features and {len(features)} samples")
                                
                                # Check for high correlation between features
                                if len(features.columns) > 1:
                                    corr_matrix = features.corr().abs()
                                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                                    high_corr = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]) 
                                                for i, j in zip(*np.where(upper > 0.95))]
                                    if high_corr:
                                        st.info("High correlation detected between some features (correlation > 0.95):")
                                        for feat1, feat2, corr in high_corr[:5]:  # Show first 5 to avoid cluttering UI
                                            st.info(f"  - {feat1} and {feat2}: {corr:.3f}")
                                        if len(high_corr) > 5:
                                            st.info(f"  - And {len(high_corr) - 5} more...")
                            
                            # Prepare ML data
                            ml_data = preprocessor.prepare_ml_data(
                                features, 
                                target_column,
                                test_size=0 # No need for test split here
                            )
                            
                            # Train the model
                            train_result = model.train(ml_data)
                            
                        elif model_category == 'Deep Learning':
                            # Create and prepare datasets for DL models
                            hp = st.session_state.hyperparameters
                            sequence_length = hp.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
                            batch_size = hp.get('batch_size', DEFAULT_BATCH_SIZE)
                            
                            # Prepare deep learning datasets
                            train_dataset, val_dataset, test_dataset, scaler = preprocessor.prepare_dl_datasets(
                                train_data[target_column],
                                sequence_length=sequence_length,
                                batch_size=batch_size,
                                val_split=st.session_state.split_config['val_size'] / st.session_state.split_config['train_size'] if val_data is None else 0,
                                test_split=0  # We already have test data
                            )
                            
                            # If val_data is provided separately
                            if val_data is not None and val_dataset is None:
                                _, val_dataset, _, _ = preprocessor.prepare_dl_datasets(
                                    val_data[target_column],
                                    sequence_length=sequence_length,
                                    batch_size=batch_size,
                                    val_split=0,
                                    test_split=0
                                )
                            
                            # Store scaler in session state
                            st.session_state.scaler = scaler
                            
                            # Create model
                            if st.session_state.selected_model == 'Simple RNN':
                                model = SimpleRNNModel(
                                    name='SimpleRNN',
                                    sequence_length=sequence_length,
                                    n_features=1
                                )
                                
                                model.build(
                                    units=hp.get('units', DEFAULT_RNN_UNITS),
                                    dropout_rate=hp.get('dropout_rate', DEFAULT_DROPOUT_RATE)
                                )
                                
                            elif st.session_state.selected_model == 'LSTM':
                                model = LSTMModel(
                                    name='LSTM',
                                    sequence_length=sequence_length,
                                    n_features=1
                                )
                                
                                model.build(
                                    units=hp.get('units', DEFAULT_RNN_UNITS),
                                    dropout_rate=hp.get('dropout_rate', DEFAULT_DROPOUT_RATE)
                                )
                                
                            elif st.session_state.selected_model == 'Stacked LSTM+RNN':
                                model = StackedModel(
                                    name='StackedLSTMRNN',
                                    sequence_length=sequence_length,
                                    n_features=1
                                )
                                
                                model.build(
                                    lstm_units=hp.get('lstm_units', 128),
                                    rnn_units=hp.get('rnn_units', 64),
                                    dropout_rate=hp.get('dropout_rate', DEFAULT_DROPOUT_RATE)
                                )
                            
                            # Compile model
                            model.model.compile(
                                optimizer=tf.keras.optimizers.Adam(learning_rate=hp.get('learning_rate', DEFAULT_LEARNING_RATE)),
                                loss='mse',
                                metrics=['mae']
                            )
                            
                            # Create trainer and train model
                            trainer = ModelTrainer(model, target_column)
                            train_result = trainer.train_and_evaluate(
                                train_dataset,
                                val_dataset,
                                test_dataset,
                                epochs=hp.get('epochs', DEFAULT_EPOCHS)
                            )
                            
                        elif model_category == 'Prophet' and PROPHET_AVAILABLE:
                            # Initialize Prophet model
                            model = create_prophet_model()
                            
                            hp = st.session_state.hyperparameters
                            model.build(
                                seasonality_mode=hp.get('seasonality_mode', 'additive'),
                                growth=hp.get('growth', 'linear'),
                                yearly_seasonality=hp.get('yearly_seasonality', 'auto'),
                                weekly_seasonality=hp.get('weekly_seasonality', 'auto')
                            )
                            
                            # Add country holidays if selected
                            if hp.get('include_holidays', False) and hp.get('country_name'):
                                model.add_country_holidays(hp.get('country_name'))
                            
                            # Prepare Prophet data format
                            train_prophet = pd.DataFrame({
                                'ds': train_data.index,
                                'y': train_data[target_column]
                            })
                            
                            # Train model
                            train_result = model.train(train_prophet)
                            
                        elif model_category == 'AutoGluon' and AUTOGLUON_AVAILABLE:
                            # Initialize AutoGluon model
                            hp = st.session_state.hyperparameters
                            model = create_autogluon_model()
                            
                            model.build(
                                prediction_length=hp.get('prediction_length', 30),
                                freq=hp.get('freq', 'D'),
                                time_limit=hp.get('time_limit', AUTOGLUON_DEFAULT_TIME_LIMIT)
                            )
                            
                            # Train model
                            train_result = model.train(
                                train_data,
                                val_data,
                                time_column=None,  # Use index
                                target_column=target_column,
                                enable_ensemble=hp.get('enable_ensemble', True)
                            )
                        
                        # Store trained model in session state
                        st.session_state.model = model
                        st.session_state.model_category = model_category
                        st.session_state.training_complete = True
                        
                        st.success(f"{st.session_state.selected_model} model trained successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Training results and model information
        if 'training_complete' in st.session_state and st.session_state.training_complete:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Training Results")
            
            # Display model information
            st.write(f"Model: **{st.session_state.selected_model}**")
            
            # Display data splits
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Set", f"{len(st.session_state.train_data)} samples")
            with col2:
                if st.session_state.val_data is not None:
                    st.metric("Validation Set", f"{len(st.session_state.val_data)} samples")
                else:
                    st.metric("Validation Set", "Not used")
            with col3:
                st.metric("Test Set", f"{len(st.session_state.test_data)} samples")
            
            # Evaluate on test data
            if st.button("Evaluate on Test Data"):
                with st.spinner("Evaluating model on test data..."):
                    try:
                        model = st.session_state.model
                        model_category = st.session_state.model_category
                        test_data = st.session_state.test_data
                        
                        # Evaluate based on model category
                        if model_category == 'Traditional':
                            metrics = model.evaluate(test_data[target_column])
                            
                        elif model_category == 'Machine Learning':
                            # Create features for test data
                            test_features = preprocessor.create_features(
                                test_data, 
                                target_column
                            )
                            
                            # Evaluate
                            metrics = model.evaluate(test_features)
                            
                        elif model_category == 'Deep Learning':
                            # Prepare test dataset
                            hp = st.session_state.hyperparameters
                            sequence_length = hp.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
                            batch_size = hp.get('batch_size', DEFAULT_BATCH_SIZE)
                            
                            # Create preprocessor here
                            preprocessor = TimeSeriesPreprocessor()
                            
                            # Ensure st.session_state.scaler is used if available
                            scaler = st.session_state.get('scaler', None)
                            
                            _, _, test_dataset, _ = preprocessor.prepare_dl_datasets(
                                test_data[target_column],
                                sequence_length=sequence_length,
                                batch_size=batch_size,
                                val_split=0,
                                test_split=0
                            )
                            
                            # Evaluate
                            metrics = model.evaluate(test_dataset)

                        elif model_category == 'Prophet':
                            # Prepare Prophet test data format
                            test_prophet = pd.DataFrame({
                                'ds': test_data.index,
                                'y': test_data[target_column]
                            })
                            
                            # Evaluate
                            metrics = model.evaluate(test_prophet)
                            
                        elif model_category == 'AutoGluon':
                            # Evaluate
                            metrics = model.evaluate(test_data)
                        
                        # Store metrics in session state
                        st.session_state.test_metrics = metrics
                        
                        # Display metrics
                        st.subheader("Test Metrics")
                        
                        if isinstance(metrics, dict):
                            metrics_df = pd.DataFrame({
                                'Metric': list(metrics.keys()),
                                'Value': list(metrics.values())
                            })
                            st.dataframe(metrics_df, use_container_width=True)
                        else:
                            st.write(f"Test Loss: {metrics}")
                            
                    except Exception as e:
                        st.error(f"Error during model evaluation: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Generate Forecasts")
        
        if 'training_complete' not in st.session_state or not st.session_state.training_complete:
            st.warning("Please train a model first in the Training & Validation tab.")
        else:
            model = st.session_state.model
            model_category = st.session_state.model_category
            forecast_horizon = st.session_state.forecast_horizon
            
            st.write(f"Model: **{st.session_state.selected_model}**")
            st.write(f"Forecast Horizon: **{forecast_horizon} periods**")
            
            # Generate forecast button
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    try:
                        # Generate forecasts based on model category
                        if model_category == 'Traditional':
                            # Forecast for traditional models
                            forecast = model.predict(steps=forecast_horizon)
                            
                            # Create forecast DataFrame
                            last_date = df.index[-1]
                            # First check if the index is a datetime index
                            if isinstance(last_date, pd.Timestamp):
                                # Try to infer the frequency from the data
                                inferred_freq = pd.infer_freq(df.index)
                                if inferred_freq:
                                    # Use the inferred frequency
                                    forecast_index = pd.date_range(
                                        start=last_date, 
                                        periods=forecast_horizon + 1, 
                                        freq=inferred_freq
                                    )[1:]  # Skip the first point which is the last observed point
                                else:
                                    # Fallback to daily frequency if we can't infer
                                    forecast_index = pd.date_range(
                                        start=last_date, 
                                        periods=forecast_horizon + 1, 
                                        freq='D'
                                    )[1:]
                            else:
                                # For non-datetime indices, use simple integer indexing
                                last_idx = df.index[-1] if isinstance(df.index[-1], (int, float)) else len(df)
                                forecast_index = pd.RangeIndex(start=last_idx + 1, stop=last_idx + forecast_horizon + 1)
                            
                            forecast_df = pd.DataFrame({
                                'forecast': forecast
                            }, index=forecast_index)
                            
                            # Calculate confidence intervals if available
                            if hasattr(model, 'get_prediction_intervals'):
                                intervals = model.get_prediction_intervals(steps=forecast_horizon)
                                if intervals is not None:
                                    forecast_df['lower_bound'] = intervals['lower']
                                    forecast_df['upper_bound'] = intervals['upper']
                        
                        elif model_category == 'Machine Learning':
                            # Generate future features
                            future_features = preprocessor.create_features(
                                df, 
                                target_column,
                                future_periods=forecast_horizon
                            )
                            
                            # Get only the future rows
                            future_features = future_features.iloc[-forecast_horizon:]
                            
                            # Generate predictions
                            forecast = model.predict(future_features)
                            
                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'forecast': forecast
                            }, index=future_features.index)
                            
                        elif model_category == 'Deep Learning':
                            # Get the last sequence from the data
                            scaler = st.session_state.get('scaler')
                            if scaler is None:
                                st.error("No scaler found. Cannot generate forecast.")
                                # Instead of continue, set forecast_df to None
                                forecast_df = None
                            else:
                                try:
                                    sequence_length = st.session_state.hyperparameters['sequence_length']
                                    
                                    # Extract the last sequence from the scaled data
                                    last_sequence = df[target_column].values[-sequence_length:]
                                    
                                    # Scale the sequence
                                    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
                                    
                                    # Reshape for prediction
                                    scaled_sequence = scaled_sequence.reshape(1, sequence_length, 1)
                                    
                                    # Initialize array for storing predictions
                                    forecasts = []
                                    
                                    # Generate predictions recursively
                                    curr_sequence = scaled_sequence.copy()
                                    for _ in range(forecast_horizon):
                                        # Predict next value
                                        next_pred = model.predict(curr_sequence)[0][0]
                                        forecasts.append(next_pred)
                                        
                                        # Update sequence for next prediction
                                        curr_sequence = np.append(
                                            curr_sequence[:, 1:, :], 
                                            [[next_pred]], 
                                            axis=1
                                        )
                                    
                                    # Inverse transform predictions
                                    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
                                    
                                    # Create forecast DataFrame
                                    last_date = df.index[-1]
                                    forecast_index = pd.date_range(
                                        start=last_date + pd.Timedelta(days=1), 
                                        periods=forecast_horizon, 
                                        freq='D'
                                    )
                                    
                                    forecast_df = pd.DataFrame({
                                        'forecast': forecasts
                                    }, index=forecast_index)

                                except Exception as e:
                                    st.error(f"Error during deep learning forecast generation: {str(e)}")
                                    import traceback
                                    st.error(traceback.format_exc())
                                    forecast_df = None



                        elif model_category == 'Prophet':
                            # Generate forecast with Prophet
                            future = model.model.make_future_dataframe(periods=forecast_horizon)
                            forecast = model.model.predict(future)
                            
                            # Create forecast DataFrame with confidence intervals
                            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-forecast_horizon:]
                            forecast_df = forecast_df.rename(columns={
                                'ds': 'date',
                                'yhat': 'forecast',
                                'yhat_lower': 'lower_bound',
                                'yhat_upper': 'upper_bound'
                            })
                            forecast_df.set_index('date', inplace=True)
                        
                        elif model_category == 'AutoGluon':
                            # Generate forecast with AutoGluon
                            forecast = model.predict(prediction_length=forecast_horizon)
                            
                            # Format the forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'forecast': forecast['mean'].values if 'mean' in forecast.columns else forecast['0.5'].values,
                            }, index=forecast.index)
                            
                            # Add confidence intervals if available
                            if '0.1' in forecast.columns and '0.9' in forecast.columns:
                                forecast_df['lower_bound'] = forecast['0.1'].values
                                forecast_df['upper_bound'] = forecast['0.9'].values
                        
                        # Store forecast in session state
                        st.session_state.forecast = forecast_df
                        
                        # Display forecast
                        st.subheader("Forecast Results")
                        st.dataframe(forecast_df.head(10), use_container_width=True)
                        
                        # Plot forecast
                        visualizer = DataVisualizer()
                        
                        # Create historical + forecast plot
                        history = df[target_column][-60:] if len(df) > 60 else df[target_column]
                        
                        fig = plt.figure(figsize=(12, 6))
                        plt.plot(history.index, history.values, label='Historical Data', color='blue')
                        plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red', linestyle='--')
                        
                        # Plot confidence intervals if available
                        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                            plt.fill_between(
                                forecast_df.index,
                                forecast_df['lower_bound'],
                                forecast_df['upper_bound'],
                                color='red',
                                alpha=0.2,
                                label='95% Confidence Interval'
                            )
                        
                        plt.title(f'{st.session_state.selected_model} Forecast for {target_column}')
                        plt.xlabel('Date')
                        plt.ylabel(target_column)
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Create download button for forecast
                        csv = forecast_df.to_csv()
                        st.download_button(
                            label="Download Forecast CSV",
                            data=csv,
                            file_name=f"{st.session_state.selected_model}_forecast.csv",
                            mime="text/csv"
                        )
                        
                        # Add to model results
                        st.session_state.model_results[st.session_state.selected_model] = {
                            'model': model,
                            'forecast': forecast_df,
                            'test_metrics': st.session_state.test_metrics if 'test_metrics' in st.session_state else None
                        }
                        
                        st.success("Forecast generated successfully! You can compare multiple models in the Evaluation tab.")
                        
                    except Exception as e:
                        st.error(f"Error during forecasting: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Evaluation and Comparison")
        
        if 'model_results' not in st.session_state or not st.session_state.model_results:
            st.warning("No models have been trained and evaluated yet. Please train at least one model.")
        else:
            # Display available models
            st.write("Models available for comparison:")
            models_list = list(st.session_state.model_results.keys())
            
            for i, model_name in enumerate(models_list):
                st.write(f"- **{model_name}**")
            
            # If only one model, just display its results
            if len(models_list) == 1:
                model_name = models_list[0]
                model_result = st.session_state.model_results[model_name]
                
                st.subheader(f"Evaluation Results for {model_name}")
                
                if model_result['test_metrics'] is not None:
                    # Display metrics
                    if isinstance(model_result['test_metrics'], dict):
                        metrics_df = pd.DataFrame({
                            'Metric': list(model_result['test_metrics'].keys()),
                            'Value': list(model_result['test_metrics'].values())
                        })
                        st.dataframe(metrics_df, use_container_width=True)
                    else:
                        st.write(f"Test Loss: {model_result['test_metrics']}")
                
                # Display forecast plot
                if 'forecast' in model_result:
                    forecast_df = model_result['forecast']
                    history = df[target_column][-60:] if len(df) > 60 else df[target_column]
                    
                    fig = plt.figure(figsize=(12, 6))
                    plt.plot(history.index, history.values, label='Historical Data', color='blue')
                    plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red', linestyle='--')
                    
                    # Plot confidence intervals if available
                    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                        plt.fill_between(
                            forecast_df.index,
                            forecast_df['lower_bound'],
                            forecast_df['upper_bound'],
                            color='red',
                            alpha=0.2,
                            label='95% Confidence Interval'
                        )
                    
                    plt.title(f'{model_name} Forecast for {target_column}')
                    plt.xlabel('Date')
                    plt.ylabel(target_column)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            # If multiple models, allow comparison
            else:
                st.subheader("Model Comparison")
                
                # Select models to compare
                selected_models = st.multiselect(
                    "Select Models to Compare",
                    models_list,
                    default=models_list
                )
                
                if selected_models:
                    # Compare selected models
                    
                    # 1. Metrics comparison
                    st.write("### Performance Metrics Comparison")
                    
                    metrics_data = []
                    common_metrics = set()
                    
                    # Find common metrics across all selected models
                    for model_name in selected_models:
                        model_result = st.session_state.model_results[model_name]
                        if model_result['test_metrics'] is not None and isinstance(model_result['test_metrics'], dict):
                            common_metrics.update(model_result['test_metrics'].keys())
                    
                    # Create data for metrics comparison
                    for model_name in selected_models:
                        model_result = st.session_state.model_results[model_name]
                        if model_result['test_metrics'] is not None:
                            if isinstance(model_result['test_metrics'], dict):
                                metrics_dict = model_result['test_metrics']
                                metrics_row = {'Model': model_name}
                                
                                for metric in common_metrics:
                                    metrics_row[metric] = metrics_dict.get(metric, None)
                                
                                metrics_data.append(metrics_row)
                            else:
                                # Handle single metric case
                                metrics_data.append({
                                    'Model': model_name,
                                    'Test Loss': model_result['test_metrics']
                                })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Create a bar chart for each metric
                        for metric in common_metrics:
                            if metric in metrics_df.columns:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                metrics_df.plot(x='Model', y=metric, kind='bar', ax=ax)
                                plt.title(f'Comparison of {metric} Across Models')
                                plt.ylabel(metric)
                                plt.grid(True, alpha=0.3)
                                st.pyplot(fig)
                    
                    # 2. Forecast comparison
                    st.write("### Forecast Comparison")
                    
                    # Plot forecasts from different models on the same chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data
                    history = df[target_column][-60:] if len(df) > 60 else df[target_column]
                    ax.plot(history.index, history.values, label='Historical Data', color='black')
                    
                    # Plot forecasts
                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                    
                    for i, model_name in enumerate(selected_models):
                        model_result = st.session_state.model_results[model_name]
                        if 'forecast' in model_result:
                            forecast_df = model_result['forecast']
                            color = colors[i % len(colors)]
                            
                            ax.plot(
                                forecast_df.index, 
                                forecast_df['forecast'], 
                                label=f'{model_name} Forecast',
                                color=color,
                                linestyle='--'
                            )
                    
                    plt.title(f'Forecast Comparison for {target_column}')
                    plt.xlabel('Date')
                    plt.ylabel(target_column)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # 3. Forecast error visualization
                    if 'test_data' in st.session_state:
                        st.write("### Forecast Error Visualization")
                        
                        test_data = st.session_state.test_data
                        
                        # Only use models that have test predictions available
                        models_with_test_preds = []
                        
                        for model_name in selected_models:
                            model_result = st.session_state.model_results[model_name]
                            if 'test_predictions' in model_result:
                                models_with_test_preds.append(model_name)
                        
                        if models_with_test_preds:
                            # Plot actual vs. predicted for test data
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot actual values
                            ax.plot(
                                test_data.index, 
                                test_data[target_column], 
                                label='Actual',
                                color='black'
                            )
                            
                            # Plot predictions for each model
                            for i, model_name in enumerate(models_with_test_preds):
                                model_result = st.session_state.model_results[model_name]
                                test_preds = model_result['test_predictions']
                                
                                color = colors[i % len(colors)]
                                
                                ax.plot(
                                    test_data.index[:len(test_preds)], 
                                    test_preds, 
                                    label=f'{model_name} Prediction',
                                    color=color,
                                    linestyle=':'
                                )
                            
                            plt.title('Actual vs. Predicted Values on Test Data')
                            plt.xlabel('Date')
                            plt.ylabel(target_column)
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        else:
                            st.info("Test predictions not available for any selected model.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export results
        if 'model_results' in st.session_state and st.session_state.model_results:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Export Results")
            
            export_options = st.multiselect(
                "Select what to export",
                ["Forecasts", "Performance Metrics", "Model Parameters"],
                default=["Forecasts", "Performance Metrics"]
            )
            
            if st.button("Export Selected Data"):
                try:
                    # Create export data
                    export_data = {}
                    
                    for model_name, model_result in st.session_state.model_results.items():
                        export_data[model_name] = {}
                        
                        if "Forecasts" in export_options and 'forecast' in model_result:
                            export_data[model_name]['forecast'] = model_result['forecast'].to_dict()
                        
                        if "Performance Metrics" in export_options and 'test_metrics' in model_result:
                            export_data[model_name]['metrics'] = model_result['test_metrics']
                        
                        if "Model Parameters" in export_options:
                            if hasattr(model_result['model'], 'get_parameters'):
                                export_data[model_name]['parameters'] = model_result['model'].get_parameters()
                            elif hasattr(model_result['model'], 'model') and hasattr(model_result['model'].model, 'get_config'):
                                export_data[model_name]['parameters'] = model_result['model'].model.get_config()
                    
                    # Convert to JSON
                    import json
                    json_data = json.dumps(export_data, indent=2, default=str)
                    
                    # Create download button
                    st.download_button(
                        label="Download JSON Results",
                        data=json_data,
                        file_name="time_series_forecasting_results.json",
                        mime="application/json"
                    )
                    
                    # Create Excel export for forecasts
                    if "Forecasts" in export_options:
                        # Combine all forecasts into a single DataFrame
                        combined_forecasts = pd.DataFrame()
                        
                        for model_name, model_result in st.session_state.model_results.items():
                            if 'forecast' in model_result:
                                df = model_result['forecast'].copy()
                                df.columns = [f'{model_name}_{col}' for col in df.columns]
                                
                                if combined_forecasts.empty:
                                    combined_forecasts = df
                                else:
                                    combined_forecasts = pd.concat([combined_forecasts, df], axis=1)
                        
                        # Create Excel download button
                        csv = combined_forecasts.to_csv()
                        st.download_button(
                            label="Download Combined Forecasts (CSV)",
                            data=csv,
                            file_name="combined_forecasts.csv",
                            mime="text/csv"
                        )
                    
                    st.success("Export complete! Click the download buttons to save the data.")
                    
                except Exception as e:
                    st.error(f"Error during export: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
            
            st.markdown('</div>', unsafe_allow_html=True)