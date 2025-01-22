import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import json
from dataclasses import dataclass
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for stock data"""
    missing_values_pct: float
    outliers_pct: float
    data_freshness_hours: float
    price_continuity_score: float
    volume_consistency_score: float
    technical_indicator_coverage: float

class StockDataValidator(BaseModel):
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    timestamp: datetime

    @validator('open_price', 'high_price', 'low_price', 'close_price')
    def validate_prices(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        return v

    @validator('volume')
    def validate_volume(cls, v):
        if v < 0:
            raise ValueError("Volume cannot be negative")
        return v

    @validator('high_price')
    def validate_high_price(cls, v, values):
        if 'low_price' in values and v < values['low_price']:
            raise ValueError("High price cannot be less than low price")
        return v

class DataValidator:
    def __init__(self):
        self.session = session
        self.validation_rules = {
            'price_change_threshold': 0.2,  # 20% max price change
            'volume_change_threshold': 5,    # 5x max volume change
            'min_price': 0.01,              # Minimum valid price
            'max_price_gap_days': 5,        # Maximum days between price points
            'min_daily_volume': 1000,       # Minimum daily volume
        }
        self.stock_data_validator = StockDataValidator()

    async def validate_stock_data(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate stock data and return cleaned DataFrame and list of validation messages
        """
        messages = []
        original_len = len(df)

        try:
            # Basic data structure validation
            required_columns = ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            if len(df) < original_len:
                messages.append(f"Removed {original_len - len(df)} duplicate entries")

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Validate each row
            valid_rows = []
            for index, row in df.iterrows():
                try:
                    self.stock_data_validator(
                        symbol=symbol,
                        open_price=row['open_price'],
                        high_price=row['high_price'],
                        low_price=row['low_price'],
                        close_price=row['close_price'],
                        volume=row['volume'],
                        timestamp=row['timestamp']
                    )
                    valid_rows.append(row)
                except Exception as e:
                    messages.append(f"Invalid row at {row['timestamp']}: {str(e)}")

            df = pd.DataFrame(valid_rows)
            if len(df) == 0:
                raise ValueError("No valid data points after validation")

            # Check for gaps in data
            df = self._validate_data_continuity(df, messages)

            # Check for anomalies
            df = self._validate_price_changes(df, messages)
            df = self._validate_volume_changes(df, messages)

            # Validate technical indicators
            df = self._validate_technical_indicators(df, messages)

            # Comprehensive data quality validation
            is_valid, metrics, quality_messages = self._validate_data_quality(df)
            messages.extend(quality_messages)

            if not is_valid:
                df = self._clean_data(df)

            return df, messages

        except Exception as e:
            logger.error(f"Error validating stock data: {str(e)}")
            messages.append(f"Validation error: {str(e)}")
            return df, messages

    async def validate_news_data(self, news_items: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Validate news data and return cleaned list and validation messages
        """
        messages = []
        valid_news = []

        try:
            for item in news_items:
                try:
                    NewsDataValidator(**item)
                    valid_news.append(item)
                except Exception as e:
                    messages.append(f"Invalid news item: {str(e)}")

            # Check for duplicates
            seen_urls = set()
            unique_news = []
            for item in valid_news:
                if item['url'] not in seen_urls:
                    seen_urls.add(item['url'])
                    unique_news.append(item)
                else:
                    messages.append(f"Removed duplicate news item: {item['url']}")

            return unique_news, messages

        except Exception as e:
            logger.error(f"Error validating news data: {str(e)}")
            messages.append(f"Validation error: {str(e)}")
            return valid_news, messages

    def _validate_data_continuity(self, df: pd.DataFrame, messages: List[str]) -> pd.DataFrame:
        """Check for gaps in time series data"""
        try:
            # Check for missing days
            date_range = pd.date_range(start=df['timestamp'].min(),
                                     end=df['timestamp'].max(),
                                     freq='B')  # Business days
            missing_dates = date_range.difference(df['timestamp'])
            
            if len(missing_dates) > 0:
                gap_sizes = np.diff(missing_dates)
                large_gaps = gap_sizes[gap_sizes > timedelta(days=self.validation_rules['max_price_gap_days'])]
                
                if len(large_gaps) > 0:
                    messages.append(f"Found {len(large_gaps)} large gaps in data")
                    
                    # Interpolate small gaps
                    df = df.set_index('timestamp')
                    df = df.reindex(date_range)
                    df = df.interpolate(method='time', limit=self.validation_rules['max_price_gap_days'])
                    df = df.dropna()
                    df = df.reset_index()
                    messages.append("Interpolated small gaps in data")

            return df

        except Exception as e:
            logger.error(f"Error validating data continuity: {str(e)}")
            return df

    def _validate_price_changes(self, df: pd.DataFrame, messages: List[str]) -> pd.DataFrame:
        """Validate price changes and remove anomalies"""
        try:
            # Calculate price changes
            df['price_change'] = df['close_price'].pct_change()

            # Identify anomalous price changes
            anomalies = df[abs(df['price_change']) > self.validation_rules['price_change_threshold']]
            if len(anomalies) > 0:
                messages.append(f"Found {len(anomalies)} anomalous price changes")
                
                # Remove extreme anomalies
                df = df[abs(df['price_change']) <= self.validation_rules['price_change_threshold']]
                messages.append("Removed extreme price anomalies")

            return df

        except Exception as e:
            logger.error(f"Error validating price changes: {str(e)}")
            return df

    def _validate_volume_changes(self, df: pd.DataFrame, messages: List[str]) -> pd.DataFrame:
        """Validate volume changes and remove anomalies"""
        try:
            # Calculate volume changes
            df['volume_change'] = df['volume'] / df['volume'].shift(1)

            # Remove low volume days
            low_volume = df[df['volume'] < self.validation_rules['min_daily_volume']]
            if len(low_volume) > 0:
                messages.append(f"Found {len(low_volume)} low volume days")
                df = df[df['volume'] >= self.validation_rules['min_daily_volume']]

            # Identify anomalous volume changes
            volume_anomalies = df[df['volume_change'] > self.validation_rules['volume_change_threshold']]
            if len(volume_anomalies) > 0:
                messages.append(f"Found {len(volume_anomalies)} anomalous volume changes")
                
                # Replace extreme volumes with moving average
                ma_volume = df['volume'].rolling(window=5, min_periods=1).mean()
                df.loc[df['volume_change'] > self.validation_rules['volume_change_threshold'], 'volume'] = \
                    ma_volume[df['volume_change'] > self.validation_rules['volume_change_threshold']]
                messages.append("Adjusted extreme volume anomalies")

            return df

        except Exception as e:
            logger.error(f"Error validating volume changes: {str(e)}")
            return df

    def _validate_technical_indicators(self, df: pd.DataFrame, messages: List[str]) -> pd.DataFrame:
        """Validate technical indicators"""
        try:
            # Check for invalid technical indicators
            tech_indicators = ['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd']
            
            for indicator in tech_indicators:
                if indicator in df.columns:
                    # Replace invalid values with NaN
                    df[indicator] = pd.to_numeric(df[indicator], errors='coerce')
                    
                    # Count invalid values
                    invalid_count = df[indicator].isna().sum()
                    if invalid_count > 0:
                        messages.append(f"Found {invalid_count} invalid values in {indicator}")
                        
                        # Interpolate missing values
                        df[indicator] = df[indicator].interpolate(method='linear')
                        messages.append(f"Interpolated missing values in {indicator}")

            return df

        except Exception as e:
            logger.error(f"Error validating technical indicators: {str(e)}")
            return df

    def _validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, DataQualityMetrics, List[str]]:
        """
        Validate stock data quality
        Returns: (is_valid, metrics, validation_messages)
        """
        messages = []
        
        # Check for minimum required data
        if len(df) < 100:
            messages.append("Insufficient data points (minimum 100 required)")
            return False, None, messages

        # Calculate quality metrics
        metrics = self._calculate_quality_metrics(df)
        
        # Validate against thresholds
        is_valid = True
        
        if metrics.missing_values_pct > 0.05:
            is_valid = False
            messages.append(f"Too many missing values: {metrics.missing_values_pct:.2%}")
            
        if metrics.outliers_pct > 0.10:
            is_valid = False
            messages.append(f"Too many outliers detected: {metrics.outliers_pct:.2%}")
            
        if metrics.data_freshness_hours > 24:
            is_valid = False
            messages.append(f"Data too old: {metrics.data_freshness_hours:.1f} hours")
            
        if metrics.price_continuity_score < 0.95:
            is_valid = False
            messages.append(f"Poor price continuity: {metrics.price_continuity_score:.2%}")
            
        if metrics.volume_consistency_score < 0.90:
            is_valid = False
            messages.append(f"Poor volume consistency: {metrics.volume_consistency_score:.2%}")
            
        if metrics.technical_indicator_coverage < 0.98:
            is_valid = False
            messages.append(f"Insufficient technical indicator coverage: {metrics.technical_indicator_coverage:.2%}")

        return is_valid, metrics, messages

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics"""
        
        # Missing values
        missing_pct = df.isnull().mean().mean()
        
        # Outliers using ensemble method
        outliers_mask = self._detect_outliers(df)
        outliers_pct = outliers_mask.mean()
        
        # Data freshness
        latest_timestamp = pd.to_datetime(df.index[-1])
        data_age = datetime.now() - latest_timestamp
        freshness_hours = data_age.total_seconds() / 3600
        
        # Price continuity
        price_changes = df['close_price'].pct_change().abs()
        price_continuity = 1 - (price_changes > 0.1).mean()
        
        # Volume consistency
        volume_changes = df['volume'].pct_change().abs()
        volume_consistency = 1 - (volume_changes > 1.0).mean()
        
        # Technical indicator coverage
        indicator_cols = [col for col in df.columns if any(ind in col.lower() for ind in ['sma', 'ema', 'rsi', 'macd'])]
        indicator_coverage = 1 - df[indicator_cols].isnull().mean().mean()
        
        return DataQualityMetrics(
            missing_values_pct=missing_pct,
            outliers_pct=outliers_pct,
            data_freshness_hours=freshness_hours,
            price_continuity_score=price_continuity,
            volume_consistency_score=volume_consistency,
            technical_indicator_coverage=indicator_coverage
        )

    def _detect_outliers(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect outliers using ensemble method
        Returns boolean mask where True indicates outlier
        """
        feature_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        X = df[feature_cols].values
        
        # Get predictions from each detector
        predictions = []
        for detector in self.outlier_detectors.values():
            try:
                pred = detector.fit_predict(X)
                predictions.append(pred == -1)  # -1 indicates outlier
            except Exception as e:
                logger.warning(f"Outlier detector failed: {str(e)}")
                continue
        
        # Combine predictions (majority vote)
        if predictions:
            outlier_mask = np.mean(predictions, axis=0) >= 0.5
        else:
            outlier_mask = np.zeros(len(df), dtype=bool)
        
        return outlier_mask

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling outliers and missing values
        """
        # Detect outliers
        outlier_mask = self._detect_outliers(df)
        
        # Handle outliers by interpolation
        df_clean = df.copy()
        df_clean.loc[outlier_mask] = np.nan
        
        # Interpolate missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='time')
        
        # Forward/backward fill any remaining missing values
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        return df_clean

    def get_validation_summary(self, messages: List[str]) -> Dict:
        """Generate a summary of validation results"""
        try:
            return {
                'total_messages': len(messages),
                'warnings': [msg for msg in messages if 'warning' in msg.lower()],
                'errors': [msg for msg in messages if 'error' in msg.lower()],
                'info': [msg for msg in messages if not ('warning' in msg.lower() or 'error' in msg.lower())],
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating validation summary: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate detailed data quality report
        """
        is_valid, metrics, messages = self._validate_data_quality(df)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'is_valid': is_valid,
            'metrics': {
                'missing_values_pct': metrics.missing_values_pct,
                'outliers_pct': metrics.outliers_pct,
                'data_freshness_hours': metrics.data_freshness_hours,
                'price_continuity_score': metrics.price_continuity_score,
                'volume_consistency_score': metrics.volume_consistency_score,
                'technical_indicator_coverage': metrics.technical_indicator_coverage
            },
            'validation_messages': messages,
            'data_summary': {
                'total_rows': len(df),
                'date_range': f"{df.index[0]} to {df.index[-1]}",
                'columns': list(df.columns)
            }
        }

    def _analyze_data_distribution(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze the statistical distribution of price and volume data
        """
        from scipy import stats
        
        distribution_metrics = {}
        
        for column in ['close_price', 'volume']:
            # Calculate basic statistics
            data = df[column].values
            metrics = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'skew': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
            }
            
            # Perform Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(data)
            metrics['normality_test'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
            
            # Check for seasonality using autocorrelation
            acf = np.correlate(data, data, mode='full')[len(data)-1:]
            metrics['seasonality'] = {
                'daily': bool(acf[1] > 0.7),
                'weekly': bool(acf[5] > 0.7),
                'monthly': bool(acf[20] > 0.7)
            }
            
            distribution_metrics[column] = metrics
            
        return distribution_metrics

    async def cross_validate_with_external(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Cross-validate data with external sources
        """
        messages = []
        is_valid = True
        
        try:
            # Compare with market index (e.g., S&P 500)
            spy_data = await self._fetch_market_index_data('SPY', df.index[0], df.index[-1])
            
            if spy_data is not None:
                # Calculate correlation with market
                price_corr = np.corrcoef(df['close_price'].pct_change().dropna(),
                                       spy_data['close_price'].pct_change().dropna())[0,1]
                
                # Check if correlation is within expected range
                if abs(price_corr) < 0.1:
                    messages.append(f"Warning: Unusually low correlation with market index: {price_corr:.2f}")
                    is_valid = False
                
                # Compare volume patterns
                volume_ratio = df['volume'].mean() / spy_data['volume'].mean()
                if volume_ratio > 10 or volume_ratio < 0.1:
                    messages.append(f"Warning: Unusual volume ratio compared to market: {volume_ratio:.2f}")
                    is_valid = False
            
            # Add distribution analysis
            dist_metrics = self._analyze_data_distribution(df)
            
            # Check for distribution anomalies
            for column, metrics in dist_metrics.items():
                if not metrics['normality_test']['is_normal']:
                    messages.append(f"Warning: {column} data is not normally distributed")
                
                if abs(metrics['skew']) > 2:
                    messages.append(f"Warning: High skewness in {column}: {metrics['skew']:.2f}")
                
                if metrics['seasonality']['daily']:
                    messages.append(f"Info: Detected daily seasonality in {column}")
                if metrics['seasonality']['weekly']:
                    messages.append(f"Info: Detected weekly seasonality in {column}")
                if metrics['seasonality']['monthly']:
                    messages.append(f"Info: Detected monthly seasonality in {column}")
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            messages.append(f"Cross-validation error: {str(e)}")
            is_valid = False
        
        return is_valid, messages

    async def _fetch_market_index_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch market index data for comparison
        """
        try:
            # Implement your market data fetching logic here
            # This could use your existing data fetching infrastructure
            return None  # Placeholder
        except Exception as e:
            logger.error(f"Error fetching market index data: {str(e)}")
            return None
