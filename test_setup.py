import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import xgboost as xgb
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

def test_imports():
    print("Testing NumPy:", np.__version__)
    print("Testing Pandas:", pd.__version__)
    print("Testing TensorFlow:", tf.__version__)
    print("Testing PyTorch:", torch.__version__)
    print("Testing XGBoost:", xgb.__version__)
    print("Testing scikit-learn:", RandomForestRegressor().__class__.__module__)
    
    # Test YFinance by fetching some data
    print("\nTesting YFinance data fetching:")
    msft = yf.Ticker("MSFT")
    hist = msft.history(period="1d")
    print("Successfully fetched Microsoft stock data:", hist.shape)

if __name__ == "__main__":
    test_imports()
