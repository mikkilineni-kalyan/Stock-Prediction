from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..ml_models.ensemble_model import HybridPredictionSystem
from ..news_analyzer.sentiment_analyzer import EnhancedNewsAnalyzer

class BacktestEngine:
    def __init__(self, prediction_system: HybridPredictionSystem, data_fetcher, news_analyzer: EnhancedNewsAnalyzer):
        self.prediction_system = prediction_system
        self.data_fetcher = data_fetcher
        self.news_analyzer = news_analyzer
        
    def run_backtest(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest for a specific period"""
        # Get historical data
        stock_data = self.data_fetcher.get_stock_data(ticker, start=start_date, end=end_date)
        
        # Store predictions and actual results
        daily_results = []
        weekly_results = []
        
        # Iterate through each day
        for i in range(len(stock_data) - 5):  # -5 to have enough data for weekly validation
            current_date = stock_data.index[i]
            
            # Get data up to current date
            historical_data = stock_data.iloc[:i+1]
            
            try:
                # Get predictions
                prediction = self._get_prediction_for_date(ticker, historical_data, current_date)
                
                # Get actual prices
                next_day_price = stock_data['Close'].iloc[i+1]
                next_week_price = stock_data['Close'].iloc[i+5]
                
                # Record results
                daily_results.append(self._evaluate_prediction(
                    prediction['predictions']['daily'],
                    next_day_price,
                    current_date,
                    'daily'
                ))
                
                weekly_results.append(self._evaluate_prediction(
                    prediction['predictions']['weekly'],
                    next_week_price,
                    current_date,
                    'weekly'
                ))
                
            except Exception as e:
                print(f"Error in backtest for date {current_date}: {str(e)}")
        
        # Calculate performance metrics
        return self._calculate_backtest_metrics(daily_results, weekly_results)
    
    def _get_prediction_for_date(self, ticker: str, historical_data: pd.DataFrame, 
                               current_date: datetime) -> Dict[str, Any]:
        """Get prediction for a specific historical date"""
        # Create a copy of the prediction system with historical data
        prediction = self.prediction_system.predict(ticker, historical_data)
        return prediction
    
    def _evaluate_prediction(self, prediction: Dict[str, Any], actual_price: float,
                           date: datetime, timeframe: str) -> Dict[str, Any]:
        """Evaluate a single prediction"""
        predicted_direction = prediction['direction']
        actual_direction = 'up' if actual_price > prediction['target_price'] else 'down'
        
        return {
            'date': date,
            'timeframe': timeframe,
            'predicted_price': prediction['target_price'],
            'actual_price': actual_price,
            'predicted_direction': predicted_direction,
            'actual_direction': actual_direction,
            'confidence': prediction['confidence'],
            'price_error': abs(prediction['target_price'] - actual_price),
            'direction_correct': predicted_direction == actual_direction,
            'expected_change_percent': prediction['expected_change_percent'],
            'actual_change_percent': ((actual_price - prediction['target_price']) / 
                                    prediction['target_price'] * 100)
        }
    
    def _calculate_backtest_metrics(self, daily_results: List[Dict], 
                                  weekly_results: List[Dict]) -> Dict[str, Any]:
        """Calculate overall backtest performance metrics"""
        def calculate_timeframe_metrics(results):
            if not results:
                return {}
            
            df = pd.DataFrame(results)
            
            direction_accuracy = df['direction_correct'].mean()
            avg_price_error = df['price_error'].mean()
            avg_confidence = df['confidence'].mean()
            
            # Calculate risk metrics
            price_errors = df['price_error']
            max_drawdown = price_errors.max()
            sharpe_ratio = (df['actual_change_percent'].mean() / 
                          df['actual_change_percent'].std() * np.sqrt(252))
            
            return {
                'direction_accuracy': float(direction_accuracy),
                'avg_price_error': float(avg_price_error),
                'avg_confidence': float(avg_confidence),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'total_predictions': len(results),
                'profitable_predictions': int(df['direction_correct'].sum())
            }
        
        return {
            'daily_metrics': calculate_timeframe_metrics(daily_results),
            'weekly_metrics': calculate_timeframe_metrics(weekly_results),
            'test_period': {
                'start': min(r['date'] for r in daily_results),
                'end': max(r['date'] for r in daily_results)
            }
        } 