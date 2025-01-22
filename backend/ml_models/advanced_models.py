import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D

import yfinance as yf

import logging

from transformers import TransformerModel

from wavenet import WaveNetModel

from attention_model import AttentionModel



logger = logging.getLogger(__name__)



class LSTMPredictor:

    def __init__(self):

        self.model = self._build_model()

        self.scaler = MinMaxScaler()

        

    def _build_model(self):

        model = Sequential([

            LSTM(100, return_sequences=True, input_shape=(60, 1)),

            Dropout(0.2),

            LSTM(100, return_sequences=False),

            Dropout(0.2),

            Dense(50),

            Dense(1)

        ])

        model.compile(optimizer='adam', loss='mse')

        return model



    def predict(self, data, sequence_length=60):

        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        sequences = self._create_sequences(scaled_data, sequence_length)

        predictions = self.model.predict(sequences)

        return self.scaler.inverse_transform(predictions)



class XGBoostPredictor:

    def __init__(self):

        self.model = xgb.XGBRegressor(

            objective='reg:squarederror',

            n_estimators=1000,

            learning_rate=0.01,

            max_depth=7

        )

        

    def predict(self, features):

        return self.model.predict(features)



class ProphetPredictor:

    def __init__(self):

        self.model = Prophet(

            changepoint_prior_scale=0.05,

            yearly_seasonality=True,

            weekly_seasonality=True,

            daily_seasonality=True

        )

        

    def predict(self, data, periods):

        self.model.fit(data)

        future = self.model.make_future_dataframe(periods=periods)

        return self.model.predict(future) 



class AdvancedPredictor:

    def __init__(self):

        self.models = {

            'lstm': self._build_lstm(),

            'transformer': self._build_transformer(),

            'wavenet': self._build_wavenet(),

            'attention': self._build_attention_model(),

            'gru': self._build_gru(),

            'cnn_lstm': self._build_cnn_lstm()

        }

        

    def _build_transformer(self):

        return TransformerModel(

            n_layers=6,

            d_model=512,

            n_head=8,

            d_ff=2048,

            dropout=0.1

        )

        

    def _build_wavenet(self):

        return WaveNetModel(

            layers=10,

            channels=32,

            kernel_size=2,

            dilation_depth=8

        )

        

    def _build_attention_model(self):

        return AttentionModel(

            input_dim=60,

            hidden_dim=128,

            num_heads=4,

            num_layers=3

        )

        

    def _build_gru(self):

        model = Sequential([

            LSTM(100, return_sequences=True, input_shape=(60, 1)),

            Dropout(0.2),

            LSTM(100, return_sequences=False),

            Dropout(0.2),

            Dense(50),

            Dense(1)

        ])

        model.compile(optimizer='adam', loss='mse')

        return model

        

    def _build_cnn_lstm(self):

        model = Sequential([

            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(60, 1)),

            MaxPooling1D(pool_size=2),

            LSTM(100, return_sequences=True),

            Dropout(0.2),

            LSTM(50),

            Dense(25),

            Dense(1)

        ])

        model.compile(optimizer='adam', loss='mse')

        return model

        

    def predict(self, data, features):

        predictions = {}

        for name, model in self.models.items():

            try:

                pred = model.predict(data)

                predictions[name] = pred

            except Exception as e:

                logger.error(f"Error in {name} prediction: {str(e)}")

                

        return self._ensemble_predictions(predictions)



    def _ensemble_predictions(self, predictions):

        try:

            valid_predictions = {}

            weights = {

                'lstm': 0.3,

                'transformer': 0.2,

                'wavenet': 0.15,

                'attention': 0.15,

                'gru': 0.1,

                'cnn_lstm': 0.1

            }

            

            # Filter out None predictions

            for model_name, pred in predictions.items():

                if pred is not None:

                    valid_predictions[model_name] = pred

            

            if not valid_predictions:

                return None

            

            # Normalize weights for available predictions

            total_weight = sum(weights[model] for model in valid_predictions.keys())

            normalized_weights = {

                model: weights[model]/total_weight 

                for model in valid_predictions.keys()

            }

            

            # Combine predictions

            ensemble_pred = np.zeros_like(list(valid_predictions.values())[0])

            for model_name, pred in valid_predictions.items():

                ensemble_pred += pred * normalized_weights[model_name]

            

            return ensemble_pred

            

        except Exception as e:

            logger.error(f"Ensemble prediction error: {str(e)}")

            return None
