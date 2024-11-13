import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
import xgboost as xgb
from prophet import Prophet
import torch
import torch.nn as nn

class HybridStockPredictor:
    def __init__(self):
        self.models = {
            'lstm': self.create_lstm_model(),
            'transformer': self.create_transformer_model(),
            'prophet': Prophet(),
            'xgboost': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.1
            ),
            'ensemble': None  # Will be created after training individual models
        }
        self.scaler = MinMaxScaler()

    def create_lstm_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def create_transformer_model(self):
        class TransformerModel(nn.Module):
            def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2):
                super().__init__()
                self.embedding = nn.Linear(input_dim, d_model)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead),
                    num_layers
                )
                self.decoder = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = self.decoder(x)
                return x

        return TransformerModel()

    def prepare_data(self, data, sequence_length=60):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)

    def train_models(self, stock_data):
        X, y = self.prepare_data(stock_data)
        X_train, y_train = X[:-30], y[:-30]  # Save last 30 days for testing

        # Train LSTM
        self.models['lstm'].fit(
            X_train.reshape(-1, 60, 1),
            y_train,
            epochs=50,
            batch_size=32,
            verbose=0
        )

        # Train Transformer
        transformer_data = torch.FloatTensor(X_train.reshape(-1, 60, 1))
        self.models['transformer'].train()
        optimizer = torch.optim.Adam(self.models['transformer'].parameters())
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            optimizer.zero_grad()
            output = self.models['transformer'](transformer_data)
            loss = criterion(output, torch.FloatTensor(y_train).reshape(-1, 1))
            loss.backward()
            optimizer.step()

        # Train Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(end=pd.Timestamp.now(), periods=len(stock_data)),
            'y': stock_data
        })
        self.models['prophet'].fit(df)

        # Train XGBoost
        self.models['xgboost'].fit(X_train, y_train)

        # Create and train ensemble model
        self.create_ensemble_model(X_train, y_train)

    def create_ensemble_model(self, X_train, y_train):
        # Get predictions from all models
        predictions = {
            'lstm': self.models['lstm'].predict(X_train.reshape(-1, 60, 1)).flatten(),
            'transformer': self.models['transformer'](torch.FloatTensor(X_train.reshape(-1, 60, 1))).detach().numpy().flatten(),
            'xgboost': self.models['xgboost'].predict(X_train)
        }
        
        # Create ensemble model (weighted average based on performance)
        ensemble_X = np.column_stack([predictions[model] for model in predictions])
        self.models['ensemble'] = xgb.XGBRegressor()
        self.models['ensemble'].fit(ensemble_X, y_train)

    def predict(self, stock_symbol, days_ahead=30):
        # Get historical data
        stock_data = self.get_stock_data(stock_symbol)
        X_test = self.prepare_data(stock_data)[-1].reshape(1, 60, 1)

        predictions = {}
        confidence_intervals = {}

        # LSTM prediction
        lstm_pred = self.models['lstm'].predict(X_test)
        predictions['lstm'] = self.scaler.inverse_transform(lstm_pred)

        # Transformer prediction
        transformer_pred = self.models['transformer'](torch.FloatTensor(X_test))
        predictions['transformer'] = self.scaler.inverse_transform(transformer_pred.detach().numpy())

        # Prophet prediction
        future_dates = self.models['prophet'].make_future_dataframe(periods=days_ahead)
        prophet_forecast = self.models['prophet'].predict(future_dates)
        predictions['prophet'] = prophet_forecast['yhat'].values[-days_ahead:]

        # XGBoost prediction
        xgb_pred = self.models['xgboost'].predict(X_test.reshape(1, -1))
        predictions['xgboost'] = self.scaler.inverse_transform(xgb_pred.reshape(-1, 1))

        # Ensemble prediction
        ensemble_X = np.column_stack([
            predictions['lstm'],
            predictions['transformer'],
            predictions['xgboost']
        ])
        predictions['ensemble'] = self.models['ensemble'].predict(ensemble_X)

        # Calculate confidence intervals using bootstrap
        for model_name in predictions:
            confidence_intervals[model_name] = self.calculate_confidence_intervals(
                predictions[model_name],
                stock_data
            )

        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals
        }

    def calculate_confidence_intervals(self, predictions, historical_data, confidence=0.95):
        # Bootstrap to calculate confidence intervals
        n_bootstrap = 1000
        bootstrap_predictions = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            sample_idx = np.random.choice(len(historical_data), size=len(historical_data))
            sample_data = historical_data[sample_idx]
            
            # Calculate prediction on bootstrap sample
            X_bootstrap = self.prepare_data(sample_data)[-1].reshape(1, 60, 1)
            bootstrap_pred = self.models['lstm'].predict(X_bootstrap)  # Using LSTM for bootstrap
            bootstrap_predictions.append(bootstrap_pred)

        bootstrap_predictions = np.array(bootstrap_predictions)
        lower = np.percentile(bootstrap_predictions, ((1 - confidence) / 2) * 100, axis=0)
        upper = np.percentile(bootstrap_predictions, (1 - (1 - confidence) / 2) * 100, axis=0)

        return {
            'lower': self.scaler.inverse_transform(lower),
            'upper': self.scaler.inverse_transform(upper)
        } 