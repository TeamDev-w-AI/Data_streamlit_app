"""Model training and evaluation utilities."""

import tensorflow as tf
from tensorflow import keras
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


class ModelTrainer:
    """Class for handling model training and evaluation."""

    def __init__(self, model, target_column):
        """
        Initialize the model trainer.
        
        Args:
            model: The model to train
            target_column: Name of the target column
        """
        self.model = model
        self.target_column = target_column

    def create_callbacks(self, model_name):
        """
        Create training callbacks for TensorFlow models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            list: List of callbacks
        """
        class CustomCallback(keras.callbacks.Callback):
            def __init__(self, progress_bar, status_text, metrics_container):
                super().__init__()
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.metrics_container = metrics_container

            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.params['epochs']
                self.progress_bar.progress(progress)
                self.status_text.text(f"Training epoch {epoch + 1}/{self.params['epochs']}")
                if logs:
                    metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                    self.metrics_container.text(f"Current metrics: {metrics_str}")

        # Create Streamlit progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

        return [
            CustomCallback(progress_bar, status_text, metrics_container),
            keras.callbacks.ModelCheckpoint(
                f"{self.target_column}_{model_name.lower().replace(' ', '_')}.keras",
                save_best_only=True,
                monitor='loss',
                mode='min'
            ),
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
        ]

    def train_and_evaluate(self, train_dataset, val_dataset, test_dataset, epochs):
        """
        Train and evaluate the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            epochs: Number of epochs to train
            
        Returns:
            float: Test loss or test MAE
        """
        try:
            # Debug information
            if hasattr(train_dataset, 'element_spec'):
                st.write("Debug: Dataset element spec:", train_dataset.element_spec)

            # Get a sample batch to check shapes
            try:
                for x, y in train_dataset.take(1):
                    st.write("Debug: Sample batch shapes - X:", x.shape, "y:", y.shape)
            except:
                st.write("Debug: Could not extract sample batch")
            
            # Train model
            callbacks = self.create_callbacks(self.model.name)
            
            if hasattr(self.model, 'train') and callable(self.model.train):
                # For models with a train method (non-Keras models)
                history = self.model.train(
                    train_dataset,
                    val_dataset,
                    epochs=epochs,
                    callbacks=callbacks
                )
            else:
                # For Keras models
                history = self.model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0
                )

            # Evaluate
            if hasattr(self.model, 'evaluate') and callable(self.model.evaluate):
                # For models with an evaluate method
                test_loss = self.model.evaluate(test_dataset)
            else:
                # For Keras models
                test_loss = self.model.evaluate(test_dataset, verbose=0)
                
            # Extract MAE if available
            test_mae = test_loss[1] if isinstance(test_loss, list) else test_loss

            # Plot history
            self.plot_training_history(history)

            return test_mae

        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: History object from model training
        """
        if hasattr(history, 'history'):
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                ax.plot(history.history['val_loss'], label='Validation Loss')
            
            if 'mae' in history.history:
                ax.plot(history.history['mae'], label='Training MAE')
            if 'val_mae' in history.history:
                ax.plot(history.history['val_mae'], label='Validation MAE')
                
            ax.set_title(f'{self.model.name} Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss/Metric')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        elif isinstance(history, dict) and 'loss' in history:
            # For custom history dictionaries
            fig, ax = plt.subplots()
            ax.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                ax.plot(history['val_loss'], label='Validation Loss')
            ax.set_title(f'{self.model.name} Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)