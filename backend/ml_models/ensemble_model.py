import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
from utils.market_calendar import MarketCalendar
from ml_models.confidence_adjuster import ConfidenceAdjuster
from indicators.technical_indicators import TechnicalIndicators

class HybridPredictionSystem:
    def __init__(self, data_fetcher, news_analyzer):
        self.data_fetcher = data_fetcher
        self.news_analyzer = news_analyzer
        
        # Initialize models for different timeframes
        self.models = {
            'hourly': {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1),
                'lstm': self._build_lstm(sequence_length=24)  # 24 hours
            },
            'daily': {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1),
                'lstm': self._build_lstm(sequence_length=30)  # 30 days
            },
            'weekly': {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1),
                'lstm': self._build_lstm(sequence_length=52)  # 52 weeks
            }
        }
        
        # Initialize technical indicators
        self.tech_indicators = TechnicalIndicators()
        
    def _build_lstm(self, sequence_length: int):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def predict(self, ticker: str, company_name: str) -> Dict[str, Any]:
        # Get historical data
        stock_data = self.data_fetcher.get_stock_data(ticker, period="2y")
        
        # Get technical indicators
        indicators = self.tech_indicators.calculate_all(stock_data)
        
        # Get news sentiment
        news_impact = self.news_analyzer.analyze(ticker, company_name)
        
        # Generate predictions for each timeframe
        predictions = {}
        for timeframe in ['hourly', 'daily', 'weekly']:
            predictions[timeframe] = self._generate_prediction(
                stock_data, 
                indicators,
                news_impact,
                timeframe
            )
        
        return {
            'ticker': ticker,
            'company': company_name,
            'predictions': predictions,
            'technical_analysis': indicators,
            'news_sentiment': news_impact,
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'confidence_score': self._calculate_confidence(predictions, news_impact)
            }
        }
    
    def _generate_prediction(self, stock_data: pd.DataFrame, indicators: Dict, news_impact: Dict, timeframe: str) -> Dict[str, Any]:
        """Generate predictions for specific timeframe"""
        # Get ML prediction
        ml_pred = self._get_ensemble_prediction(stock_data, timeframe)
        
        # Adjust confidence based on various factors
        adjusted_confidence = ConfidenceAdjuster.adjust_confidence(
            ml_pred,
            days_ahead,
            indicators['volatility'],
            news_impact['score']
        )
        
        # Calculate prediction intervals
        intervals = ConfidenceAdjuster.calculate_prediction_intervals(
            ml_pred['prediction'],
            adjusted_confidence,
            indicators['volatility'],
            days_ahead
        )
        
        # Combine with news sentiment
        combined = self._combine_predictions(ml_pred, news_impact, timeframe)
        combined['confidence'] = adjusted_confidence
        combined['price_range'] = {
            'low': intervals['lower'],
            'high': intervals['upper']
        }
        
        return {
            'target_price': combined['target_price'],
            'price_range': combined['price_range'],
            'confidence': combined['confidence'],
            'direction': combined['direction'],
            'expected_change_percent': combined['expected_change_percent'],
            'days_ahead': days_ahead,
            'is_trading_day': True,
            'trading_hours': market_cal.get_trading_hours(target_date)
        }
    
    def _get_ensemble_prediction(self, stock_data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Get predictions from all models for specific timeframe"""
        # Prepare features based on timeframe
        features = self._prepare_features(stock_data, timeframe)
        
        # Get predictions from each model
        models = self.models[timeframe]
        rf_pred = models['rf'].predict(features)
        xgb_pred = models['xgb'].predict(features)
        
        # Prepare LSTM data
        sequence_length = 30 if timeframe == '1d' else 60
        lstm_data = self._prepare_lstm_data(stock_data, sequence_length)
        lstm_pred = models['lstm'].predict(lstm_data)
        
        # Combine predictions with weights
        weights = [0.3, 0.3, 0.4]  # RF, XGB, LSTM weights
        final_prediction = (
            rf_pred * weights[0] + 
            xgb_pred * weights[1] + 
            lstm_pred * weights[2]
        )
        
        # Calculate confidence based on model agreement
        predictions = [rf_pred[0], xgb_pred[0], lstm_pred[0][0]]
        confidence = 1 - (np.std(predictions) / np.mean(predictions))
        
        # Calculate expected price range
        std_dev = np.std(predictions)
        current_price = stock_data['Close'].iloc[-1]
        
        return {
            'prediction': float(final_prediction[0]),
            'confidence': float(confidence),
            'timeframe': timeframe,
            'price_range': {
                'low': float(final_prediction[0] - std_dev),
                'high': float(final_prediction[0] + std_dev)
            },
            'expected_change_percent': float((final_prediction[0] - current_price) / current_price * 100),
            'model_predictions': {
                'random_forest': float(rf_pred[0]),
                'xgboost': float(xgb_pred[0]),
                'lstm': float(lstm_pred[0][0])
            }
        }
    
    def _prepare_features(self, stock_data: pd.DataFrame, timeframe: str) -> np.ndarray:
        """Prepare features based on timeframe"""
        df = stock_data.copy()
        
        # Calculate basic features
        df['Returns'] = df['Close'].pct_change()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Add timeframe-specific features
        if timeframe == '1w':
            df['Weekly_Return'] = df['Close'].pct_change(5)
            df['Weekly_Volatility'] = df['Weekly_Return'].rolling(window=20).std()
            features = df[['Returns', 'MA20', 'MA50', 'Volatility', 'Volume', 
                         'Weekly_Return', 'Weekly_Volatility']].values
        else:
            features = df[['Returns', 'MA20', 'MA50', 'Volatility', 'Volume']].values
        
        return features[-1:]  # Return latest data point
    
    def _combine_predictions(self, ml_pred: Dict, news_pred: Dict, timeframe: str) -> Dict[str, Any]:
        """Combine ML and news predictions for specific timeframe"""
        # Adjust news impact based on timeframe
        news_weight = 0.3 if timeframe == '1d' else 0.2  # Less news weight for longer timeframes
        
        # Get prediction directions
        ml_direction = 'up' if ml_pred['prediction'] > ml_pred['model_predictions']['random_forest'] else 'down'
        news_direction = 'up' if news_pred['impact'] == 'positive' else 'down'
        
        # Calculate combined confidence
        combined_confidence = (
            ml_pred['confidence'] * (1 - news_weight) +
            news_pred['confidence'] * news_weight
        )
        
        # Boost confidence if directions agree
        if ml_direction == news_direction:
            combined_confidence *= 1.2
        
        # Calculate target price range
        current_price = float(ml_pred['model_predictions']['random_forest'])
        news_adjustment = (news_pred['score'] - 3) / 10
        
        price_range = {
            'low': ml_pred['price_range']['low'] * (1 + news_adjustment),
            'high': ml_pred['price_range']['high'] * (1 + news_adjustment)
        }
        
        return {
            'direction': ml_direction if ml_pred['confidence'] > news_pred['confidence'] else news_direction,
            'confidence': min(combined_confidence, 1.0),
            'target_price': float(ml_pred['prediction'] * (1 + news_adjustment)),
            'price_range': price_range,
            'expected_change_percent': ml_pred['expected_change_percent'],
            'timeframe': timeframe
        }
    
    def _prepare_lstm_data(self, stock_data: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Prepare data for LSTM model"""
        # Use last sequence_length days of normalized closing prices
        data = stock_data['Close'].values[-sequence_length:]
        data = (data - np.mean(data)) / np.std(data)
        return data.reshape(1, sequence_length, 1)
    
    def _calculate_confidence(self, predictions: Dict, news_impact: Dict) -> float:
        """Calculate confidence score based on predictions and news sentiment"""
        # Calculate average confidence
        average_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
        
        # Calculate news sentiment impact
        news_impact_score = news_impact['score']
        
        # Calculate confidence score
        confidence_score = average_confidence + news_impact_score
        
        return float(confidence_score)

class NewsAnalyzer:
    def __init__(self):
        self.news_sources = []
        self.historical_correlations = {}
        
    def analyze(self, ticker: str):
        # Implement news analysis logic
        pass