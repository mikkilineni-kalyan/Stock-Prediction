import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class AdvancedStockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = self._build_lstm()
        self.rf_model = RandomForestRegressor(n_estimators=100)
        self.xgb_model = XGBRegressor(objective='reg:squarederror')
        
    def _build_lstm(self):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(60, 1)),
            Dropout(0.3),
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(100)),
            Dropout(0.3),
            Dense(100, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers
            metrics=['mae', 'mse']
        )
        return model
    
    def prepare_features(self, data):
        # Technical indicators
        df = pd.DataFrame(data, columns=['close'])
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['close'])
        df['MACD'] = self._calculate_macd(df['close'])
        df['VOL_20'] = df['close'].rolling(window=20).std()
        
        return df.fillna(method='bfill')
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2
    
    def train(self, data, sentiment_scores=None):
        features_df = self.prepare_features(data)
        if sentiment_scores is not None:
            features_df['sentiment'] = sentiment_scores
            
        X, y = self._prepare_sequences(features_df)
        
        # Train LSTM
        self.lstm_model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ReduceLROnPlateau()
            ]
        )
        
        # Train RF and XGB
        flat_X = X.reshape(X.shape[0], -1)
        self.rf_model.fit(flat_X, y)
        self.xgb_model.fit(flat_X, y)
    
    def predict(self, data, sentiment_score=None):
        features_df = self.prepare_features(data)
        if sentiment_score is not None:
            features_df['sentiment'] = sentiment_score
            
        X, _ = self._prepare_sequences(features_df)
        flat_X = X.reshape(X.shape[0], -1)
        
        # Get predictions from all models
        lstm_pred = self.lstm_model.predict(X)
        rf_pred = self.rf_model.predict(flat_X)
        xgb_pred = self.xgb_model.predict(flat_X)
        
        # Ensemble predictions (weighted average)
        final_pred = (0.5 * lstm_pred + 0.25 * rf_pred.reshape(-1, 1) + 
                     0.25 * xgb_pred.reshape(-1, 1))
        
        return self.scaler.inverse_transform(final_pred) 