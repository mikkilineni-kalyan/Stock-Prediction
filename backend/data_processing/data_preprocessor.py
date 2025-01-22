import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from datetime import datetime, timedelta
import logging
from ..config.database import session, Stock, StockPrice, NewsArticle
import json
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.session = session
        self.scalers = {}
        self.feature_columns = [
            'open_price', 'high_price', 'low_price', 'close_price', 
            'volume', 'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd'
        ]
        self.data_quality_thresholds = {
            'missing_value_threshold': 0.1,  # Max 10% missing values
            'outlier_threshold': 3,  # Z-score threshold for outliers
            'min_data_points': 100,  # Minimum required data points
            'price_jump_threshold': 0.2,  # Max 20% price change
            'volume_spike_threshold': 5,  # Max 5x volume change
            'stale_data_threshold': 24  # Max hours for data freshness
        }
        self._load_scalers()

    def _load_scalers(self):
        """Load or initialize scalers for each feature"""
        scaler_dir = Path("models/scalers")
        scaler_dir.mkdir(parents=True, exist_ok=True)

        for feature in self.feature_columns:
            scaler_path = scaler_dir / f"{feature}_scaler.joblib"
            if scaler_path.exists():
                self.scalers[feature] = joblib.load(scaler_path)
            else:
                self.scalers[feature] = RobustScaler()

    def _save_scalers(self):
        """Save trained scalers"""
        scaler_dir = Path("models/scalers")
        for feature, scaler in self.scalers.items():
            scaler_path = scaler_dir / f"{feature}_scaler.joblib"
            joblib.dump(scaler, scaler_path)

    async def preprocess_stock_data(self, symbol: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
        """
        Preprocess stock data for the given symbol
        Returns preprocessed DataFrame or None if error occurs
        """
        try:
            # Fetch data from database
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                logger.warning(f"Stock {symbol} not found in database")
                return None

            # Get stock prices
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            prices = self.session.query(StockPrice).filter(
                StockPrice.stock_id == stock.id,
                StockPrice.timestamp >= start_date,
                StockPrice.timestamp <= end_date
            ).order_by(StockPrice.timestamp.asc()).all()

            if not prices:
                logger.warning(f"No price data found for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': p.timestamp,
                'open_price': p.open_price,
                'high_price': p.high_price,
                'low_price': p.low_price,
                'close_price': p.close_price,
                'volume': p.volume,
                'sma_20': p.sma_20,
                'sma_50': p.sma_50,
                'sma_200': p.sma_200,
                'rsi_14': p.rsi_14,
                'macd': p.macd
            } for p in prices])

            # Process the data
            df = self._clean_data(df)
            df = self._handle_missing_values(df)
            df = self._calculate_additional_features(df)
            df = self._normalize_features(df)
            df = self._add_sentiment_features(df, stock.id)

            return df

        except Exception as e:
            logger.error(f"Error preprocessing data for {symbol}: {str(e)}")
            return None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by removing outliers and invalid values"""
        try:
            # Remove rows with any infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Remove outliers using IQR method for each numerical column
            for col in self.feature_columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            return df

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df

    def check_data_quality(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Comprehensive data quality checks
        Returns: (is_valid, list of quality messages)
        """
        try:
            messages = []
            is_valid = True

            # 1. Check data completeness
            missing_pct = df.isnull().mean()
            for col in self.feature_columns:
                if missing_pct[col] > self.data_quality_thresholds['missing_value_threshold']:
                    messages.append(f"High missing values in {col}: {missing_pct[col]:.2%}")
                    is_valid = False

            # 2. Check data volume
            if len(df) < self.data_quality_thresholds['min_data_points']:
                messages.append(f"Insufficient data points: {len(df)}")
                is_valid = False

            # 3. Check data freshness
            latest_timestamp = df['timestamp'].max()
            freshness = (datetime.utcnow() - latest_timestamp).total_seconds() / 3600
            if freshness > self.data_quality_thresholds['stale_data_threshold']:
                messages.append(f"Data is stale. Latest point is {freshness:.1f} hours old")
                is_valid = False

            # 4. Check for anomalous patterns
            price_changes = df['close_price'].pct_change().abs()
            volume_changes = df['volume'].pct_change().abs()

            if price_changes.max() > self.data_quality_thresholds['price_jump_threshold']:
                messages.append(f"Unusual price jump detected: {price_changes.max():.2%}")
                is_valid = False

            if volume_changes.max() > self.data_quality_thresholds['volume_spike_threshold']:
                messages.append(f"Unusual volume spike detected: {volume_changes.max():.2f}x")
                is_valid = False

            # 5. Check data consistency
            if not (df['high_price'] >= df['low_price']).all():
                messages.append("Inconsistent high/low prices detected")
                is_valid = False

            if not (df['high_price'] >= df['close_price']).all() or \
               not (df['high_price'] >= df['open_price']).all() or \
               not (df['low_price'] <= df['close_price']).all() or \
               not (df['low_price'] <= df['open_price']).all():
                messages.append("Price relationship violation detected")
                is_valid = False

            return is_valid, messages

        except Exception as e:
            logger.error(f"Error in data quality check: {str(e)}")
            return False, [f"Error in quality check: {str(e)}"]

    def detect_outliers(self, df: pd.DataFrame, method: str = 'ensemble') -> pd.DataFrame:
        """
        Advanced outlier detection using multiple methods
        """
        try:
            outlier_masks = {}
            
            # 1. Z-score method
            for col in self.feature_columns:
                if col in df.columns:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_masks[f'{col}_zscore'] = z_scores > self.data_quality_thresholds['outlier_threshold']

            # 2. IQR method
            for col in self.feature_columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_masks[f'{col}_iqr'] = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))

            # 3. Rolling median method
            for col in self.feature_columns:
                if col in df.columns:
                    rolling_median = df[col].rolling(window=20, center=True).median()
                    rolling_std = df[col].rolling(window=20, center=True).std()
                    outlier_masks[f'{col}_rolling'] = np.abs(df[col] - rolling_median) > (3 * rolling_std)

            # Combine outlier detection methods based on specified method
            if method == 'ensemble':
                # Mark as outlier if detected by at least 2 methods
                for col in self.feature_columns:
                    if col in df.columns:
                        outlier_count = (outlier_masks[f'{col}_zscore'].astype(int) + 
                                       outlier_masks[f'{col}_iqr'].astype(int) + 
                                       outlier_masks[f'{col}_rolling'].astype(int))
                        df.loc[outlier_count >= 2, f'{col}_outlier'] = True

            return df

        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced missing value handling with multiple strategies"""
        try:
            # 1. Forward fill for technical indicators (short gaps)
            technical_indicators = ['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd']
            df[technical_indicators] = df[technical_indicators].fillna(method='ffill', limit=5)

            # 2. Interpolation for price data (short gaps)
            price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
            for col in price_columns:
                # Use linear interpolation for small gaps
                df[col] = df[col].interpolate(method='linear', limit=3)
                # Use polynomial interpolation for slightly larger gaps
                df[col] = df[col].interpolate(method='polynomial', order=2, limit=5)

            # 3. Moving average for volume
            df['volume'] = df['volume'].fillna(df['volume'].rolling(window=5, min_periods=1).mean())

            # 4. Handle remaining missing values
            for col in df.columns:
                if df[col].isnull().any():
                    if col in technical_indicators:
                        # Use exponential moving average for remaining technical indicators
                        df[col] = df[col].fillna(df[col].ewm(span=20).mean())
                    elif col in price_columns:
                        # Use previous day's values for remaining price data
                        df[col] = df[col].fillna(method='ffill')
                    else:
                        # Use column mean for any remaining missing values
                        df[col] = df[col].fillna(df[col].mean())

            return df

        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df

    def _calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional technical features"""
        try:
            # Price changes
            df['price_change'] = df['close_price'].pct_change()
            df['price_change_5d'] = df['close_price'].pct_change(periods=5)
            df['price_change_20d'] = df['close_price'].pct_change(periods=20)

            # Volatility
            df['volatility'] = df['price_change'].rolling(window=20).std()

            # Volume features
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma20']

            # Trend indicators
            df['trend_20'] = np.where(df['close_price'] > df['sma_20'], 1, -1)
            df['trend_50'] = np.where(df['close_price'] > df['sma_50'], 1, -1)

            # Price position
            df['price_position'] = (df['close_price'] - df['low_price']) / (df['high_price'] - df['low_price'])

            return df

        except Exception as e:
            logger.error(f"Error calculating additional features: {str(e)}")
            return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all features using robust scaling"""
        try:
            # Update scalers with new data
            for feature in self.feature_columns:
                if feature in df.columns:
                    data = df[feature].values.reshape(-1, 1)
                    self.scalers[feature] = self.scalers[feature].fit(data)
                    df[f"{feature}_normalized"] = self.scalers[feature].transform(data)

            # Save updated scalers
            self._save_scalers()

            return df

        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return df

    def _add_sentiment_features(self, df: pd.DataFrame, stock_id: int) -> pd.DataFrame:
        """Add sentiment analysis features from news articles"""
        try:
            # Get news articles for the stock
            news = self.session.query(NewsArticle).filter(
                NewsArticle.stock_id == stock_id,
                NewsArticle.published_at >= df['timestamp'].min(),
                NewsArticle.published_at <= df['timestamp'].max()
            ).all()

            # Create sentiment features
            sentiment_data = []
            for index, row in df.iterrows():
                date = row['timestamp']
                
                # Get articles for the day
                daily_articles = [
                    n for n in news 
                    if n.published_at.date() == date.date()
                ]
                
                if daily_articles:
                    avg_sentiment = np.mean([n.sentiment_score for n in daily_articles])
                    avg_impact = np.mean([n.impact_score for n in daily_articles])
                    article_count = len(daily_articles)
                else:
                    avg_sentiment = 0
                    avg_impact = 3  # Neutral impact
                    article_count = 0
                
                sentiment_data.append({
                    'avg_sentiment': avg_sentiment,
                    'avg_impact': avg_impact,
                    'article_count': article_count
                })

            # Add to DataFrame
            sentiment_df = pd.DataFrame(sentiment_data)
            df = pd.concat([df, sentiment_df], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error adding sentiment features: {str(e)}")
            return df

    def prepare_training_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        try:
            # Select features for training
            feature_cols = [col for col in df.columns if 'normalized' in col or col in ['avg_sentiment', 'avg_impact']]
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(len(df) - sequence_length):
                sequence = df[feature_cols].iloc[i:(i + sequence_length)].values
                target = df['close_price'].iloc[i + sequence_length]
                sequences.append(sequence)
                targets.append(target)

            return np.array(sequences), np.array(targets)

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([])
