<<<<<<< HEAD
# Time Series Forecasting Platform

An advanced Streamlit application for time series forecasting with support for Prophet and AutoGluon.

## Features

- 📊 **Multiple Data Sources**: Upload your data from CSV files or connect to online repositories
- 🧹 **Data Preprocessing**: Clean and transform your time series data
- 🤖 **Multiple Models**: Choose from statistical, machine learning, and deep learning models
- 📈 **Advanced Forecasting**: Use Prophet and AutoGluon for state-of-the-art forecasting
- 📊 **Visualization**: Compare model results with interactive charts
- 📋 **Export Results**: Save your forecasts in various formats

## Project Structure

```
app/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── main.py              # Original main application (old structure)
├── main_v2.py           # New main application with improved structure
├── simple_main.py       # Simplified main app for testing
├── data/
│   ├── __init__.py      # Data package initialization
│   ├── loader.py        # Data loading functions
│   └── preprocessor.py  # Data preprocessing utilities
├── models/
│   ├── __init__.py      # Models package initialization
│   ├── base.py          # Base model class
│   ├── deep_learning.py # Deep learning model implementations
│   ├── trainer.py       # Model training utilities
│   ├── prophet.py       # Facebook Prophet implementation
│   └── autogluon.py     # AutoGluon implementation
└── utils/
    ├── __init__.py      # Utilities package initialization
    ├── helpers.py       # Helper functions
    └── visualization.py # Visualization utilities
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/time-series-forecasting.git
cd time-series-forecasting
```

2. Create a conda environment:

```bash
conda create -n ts_forecasting python=3.10
conda activate ts_forecasting
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. (Optional) Install Facebook Prophet:

```bash
pip install prophet
```

5. (Optional) Install AutoGluon:

```bash
pip install autogluon.timeseries
```

## Running the Application

To run the application:

```bash
streamlit run app/simple_main.py
```

For the full-featured application:

```bash
streamlit run app/main_v2.py
```

## Getting Started

1. Go to the **Data Upload** page to load your time series data
2. Clean and preprocess your data
3. Select models and generate forecasts
4. Compare and export results

## Dependencies

Core dependencies:
- streamlit
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- tensorflow

Optional dependencies:
- prophet
- autogluon.timeseries

## Troubleshooting

If you encounter the error "No module named 'models.trainer'", try one of these solutions:

1. Use absolute imports by adding this to the beginning of your main script:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

2. Run the application from the project root directory:
```bash
streamlit run app/main_v2.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# Data_streamlit_app
>>>>>>> 3fd957619fd234861f6b7212d36b7c209aad111b
