import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import logging
from datetime import datetime, timedelta
import os
from .alpha_vantage_client import AlphaVantageClient
import finnhub

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = self._build_lstm()
        self.prophet_model = Prophet(daily_seasonality=True)
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6
        )
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.real_time_data = {}
        self.update_interval = 5  # seconds
        self.last_update = {}

    async def get_real_time_data(self, symbol):
        current_time = datetime.now().timestamp()
        
        if (symbol not in self.last_update or 
            current_time - self.last_update.get(symbol, 0) > self.update_interval):
            try:
                alpha_vantage_data = await self._get_alpha_vantage_data(symbol)
                finnhub_data = await self._get_finnhub_data(symbol)
                yahoo_data = await self._get_yahoo_data(symbol)
                
                self.real_time_data[symbol] = self._combine_real_time_data(
                    alpha_vantage_data,
                    finnhub_data,
                    yahoo_data
                )
                self.last_update[symbol] = current_time
            except Exception as e:
                logger.error(f"Real-time data update error: {str(e)}")
                
        return self.real_time_data.get(symbol)

    def predict(self, symbol, start_date, end_date):
        try:
            # Get historical data
            hist_data = self.prepare_data(symbol)
            if hist_data.empty:
                logger.error(f"No historical data available for {symbol}")
                return None

            # Get historical and real-time data
            hist_data = self.prepare_data(symbol)
            real_time = self.real_time_data.get(symbol, {})
            
            if hist_data.empty:
                raise ValueError("No historical data available")
            
            # Calculate time points
            time_delta = (end_date - start_date).total_seconds() / 60
            num_points = max(int(time_delta / 5), 2)  # 5-minute intervals
            
            # Get stock-specific characteristics
            current_price = float(hist_data['Close'].iloc[-1])
            volatility = float(hist_data['Close'].pct_change().std() * np.sqrt(252))
            avg_volume = float(hist_data['Volume'].mean())
            
            # Calculate stock-specific trends
            short_term_trend = hist_data['Close'].pct_change().tail(5).mean()
            medium_term_trend = hist_data['Close'].pct_change().tail(20).mean()
            long_term_trend = hist_data['Close'].pct_change().mean()
            
            # Calculate momentum indicators
            rsi = self._calculate_rsi(hist_data['Close'])[-1]
            volume_trend = hist_data['Volume'].pct_change().tail(5).mean()
            
            predictions = []
            base_prediction = current_price
            
            for i in range(num_points):
                time_factor = i / num_points
                
                # Combine multiple factors for prediction
                trend_factor = (
                    short_term_trend * 0.5 + 
                    medium_term_trend * 0.3 + 
                    long_term_trend * 0.2
                )
                
                # Add momentum effect
                momentum = 0.0
                if rsi > 70:  # Overbought
                    momentum = -0.0002
                elif rsi < 30:  # Oversold
                    momentum = 0.0002
                    
                # Add volume impact
                volume_impact = np.sign(volume_trend) * min(abs(volume_trend), 0.0001)
                
                # Calculate random walk with stock-specific factors
                random_walk = np.random.normal(
                    trend_factor + momentum + volume_impact,
                    volatility/np.sqrt(252)
                )
                
                # Add cyclical component based on time of day
                hour = (start_date + timedelta(minutes=i*5)).hour
                minute = (start_date + timedelta(minutes=i*5)).minute
                time_of_day = hour + minute/60
                
                # Market typically more volatile at open and close
                if time_of_day < 10.5 or time_of_day > 15:  # Before 10:30 AM or after 3 PM
                    random_walk *= 1.2
                
                next_price = base_prediction * (1 + random_walk)
                predictions.append(float(next_price))
                base_prediction = next_price
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {str(e)}")
            return None

    def _prophet_predict(self, data, start_date, end_date):
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': data.index,
                'y': data['Close']
            })
            
            self.prophet_model.fit(df)
            
            # Create future dates
            future_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq='5min'
            )
            future = pd.DataFrame({'ds': future_dates})
            
            # Make prediction
            forecast = self.prophet_model.predict(future)
            return forecast['yhat'].values
        except Exception as e:
            logger.error(f"Prophet prediction error: {str(e)}")
            return None

    def _build_lstm(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, symbol, lookback=60):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='60d')
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            data = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            return data.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value

    def _lstm_predict(self, data):
        if len(data) < 60:
            return None
        
        sequence_length = 60
        sequences = []
        
        close_values = data['Close'].values
        
        for i in range(len(close_values) - sequence_length):
            sequences.append(close_values[i:(i + sequence_length)])
        
        if not sequences:
            return None
        
        X = np.array(sequences)
        
        if X.size == 0:
            return None
        
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1))
        X_scaled = X_scaled.reshape(-1, sequence_length, 1)
        
        predictions = self.lstm_model.predict(X_scaled)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()

    def _xgb_predict(self, data):
        try:
            features = self._prepare_features(data)
            return self.xgb_model.predict(features)
        except Exception as e:
            logger.error(f"XGBoost prediction error: {str(e)}")
            return None

    def _rf_predict(self, data):
        try:
            features = self._prepare_features(data)
            return self.rf_model.predict(features)
        except Exception as e:
            logger.error(f"Random Forest prediction error: {str(e)}")
            return None

    def _get_alpha_vantage_data(self, symbol):
        try:
            client = AlphaVantageClient()
            return client.get_stock_data(symbol)
        except Exception as e:
            logger.error(f"Alpha Vantage data error: {str(e)}")
            return None

    def _get_finnhub_data(self, symbol):
        try:
            client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
            return client.quote(symbol)
        except Exception as e:
            logger.error(f"Finnhub data error: {str(e)}")
            return None

    def _get_yahoo_data(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            return stock.history(period='1d', interval='5m')
        except Exception as e:
            logger.error(f"Yahoo data error: {str(e)}")
            return None

    def _combine_real_time_data(self, alpha_vantage_data, finnhub_data, yahoo_data):
        combined_data = {}
        try:
            if alpha_vantage_data:
                combined_data['alpha_vantage'] = alpha_vantage_data
            if finnhub_data:
                combined_data['finnhub'] = finnhub_data
            if yahoo_data is not None and not yahoo_data.empty:
                combined_data['yahoo'] = yahoo_data.to_dict()
            return combined_data
        except Exception as e:
            logger.error(f"Data combination error: {str(e)}")
            return None

    def _prepare_features(self, data):
        # Implement feature preparation logic
        pass































