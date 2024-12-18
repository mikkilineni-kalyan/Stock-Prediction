from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import sqlite3
from dataclasses import dataclass
from api.config import Config

@dataclass
class HistoricalPrediction:
    ticker: str
    timestamp: datetime
    sentiment_score: float
    price_before: float
    price_after: float
    price_change: float
    prediction_direction: str
    actual_direction: str
    news_sources: List[str]
    confidence: float

class HistoricalTracker:
    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    ticker TEXT,
                    timestamp DATETIME,
                    sentiment_score REAL,
                    price_before REAL,
                    price_after REAL,
                    price_change REAL,
                    prediction_direction TEXT,
                    actual_direction TEXT,
                    news_sources TEXT,
                    confidence REAL
                )
            ''')
    
    def add_prediction(self, prediction: HistoricalPrediction):
        """Add a new prediction to the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.ticker,
                prediction.timestamp.isoformat(),
                prediction.sentiment_score,
                prediction.price_before,
                prediction.price_after,
                prediction.price_change,
                prediction.prediction_direction,
                prediction.actual_direction,
                ','.join(prediction.news_sources),
                prediction.confidence
            ))
    
    def get_historical_accuracy(self, ticker: str, days: int = 30) -> Dict:
        """Get historical prediction accuracy for a ticker"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE ticker = ? AND timestamp > ?
            ''', conn, params=(ticker, cutoff_date.isoformat()))
        
        if df.empty:
            return {
                'accuracy': 0.0,
                'correlation': 0.0,
                'total_predictions': 0,
                'correct_predictions': 0
            }
        
        correct_predictions = df[df['prediction_direction'] == df['actual_direction']]
        accuracy = len(correct_predictions) / len(df)
        
        correlation = df[['sentiment_score', 'price_change']].corr().iloc[0, 1]
        
        return {
            'accuracy': float(accuracy),
            'correlation': float(correlation),
            'total_predictions': len(df),
            'correct_predictions': len(correct_predictions),
            'average_confidence': float(df['confidence'].mean()),
            'recent_trend': self._calculate_recent_trend(df)
        }
    
    def _calculate_recent_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate recent prediction trends"""
        df = df.sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate 7-day rolling accuracy
        df['correct'] = df['prediction_direction'] == df['actual_direction']
        rolling_accuracy = df['correct'].rolling(7).mean()
        
        # Get recent accuracy and trend
        recent_accuracy = float(rolling_accuracy.iloc[-1]) if not rolling_accuracy.empty else 0.0
        
        # Determine trend direction using proper ternary syntax
        accuracy_trend = (
            'improving' if rolling_accuracy.iloc[-1] > rolling_accuracy.iloc[-7]
            else 'declining'
        ) if len(rolling_accuracy) >= 7 else 'neutral'
        
        return {
            'recent_accuracy': recent_accuracy,
            'accuracy_trend': accuracy_trend
        }
    
    def update_actual_price(self, ticker: str, timestamp: datetime, actual_price: float):
        """Update the actual price and direction after prediction"""
        with sqlite3.connect(self.db_path) as conn:
            # Get the prediction record
            prediction = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE ticker = ? AND timestamp = ?
            ''', conn, params=(ticker, timestamp.isoformat()))
            
            if not prediction.empty:
                price_change = actual_price - prediction['price_before'].iloc[0]
                actual_direction = 'up' if price_change > 0 else 'down'
                
                conn.execute('''
                    UPDATE predictions 
                    SET price_after = ?, price_change = ?, actual_direction = ?
                    WHERE ticker = ? AND timestamp = ?
                ''', (actual_price, price_change, actual_direction, ticker, timestamp.isoformat())) 
    
    def get_todays_predictions(self) -> List[HistoricalPrediction]:
        """Get all predictions from today that haven't been updated"""
        today = date.today()
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE date(timestamp) = ? 
                AND price_after IS NULL
            ''', conn, params=(today.isoformat(),))
        
        if df.empty:
            return []
        
        return [
            HistoricalPrediction(
                ticker=row['ticker'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                sentiment_score=row['sentiment_score'],
                price_before=row['price_before'],
                price_after=row['price_after'],
                price_change=row['price_change'],
                prediction_direction=row['prediction_direction'],
                actual_direction=row['actual_direction'],
                news_sources=row['news_sources'].split(','),
                confidence=row['confidence']
            )
            for _, row in df.iterrows()
        ]
    
    def remove_old_predictions(self, cutoff_date: datetime):
        """Remove predictions older than cutoff_date"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                DELETE FROM predictions 
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))