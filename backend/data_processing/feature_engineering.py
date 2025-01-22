import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
from textblob import TextBlob
import yfinance as yf

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_groups = {
            'price': ['open', 'high', 'low', 'close', 'volume'],
            'technical': ['sma', 'ema', 'rsi', 'macd', 'bbands', 'stoch'],
            'volatility': ['atr', 'natr', 'trange'],
            'momentum': ['roc', 'mom', 'willr'],
            'trend': ['adx', 'dmi', 'aroon'],
            'volume': ['obv', 'ad', 'adosc'],
            'market': ['spy_correlation', 'sector_correlation', 'vix']
        }

    def engineer_features(self, df: pd.DataFrame, include_sentiment: bool = True) -> pd.DataFrame:
        """
        Engineer comprehensive feature set for stock prediction
        """
        try:
            df = df.copy()
            
            # Basic price features
            df = self._calculate_price_features(df)
            
            # Technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Volatility features
            df = self._calculate_volatility_features(df)
            
            # Market regime features
            df = self._calculate_market_regime_features(df)
            
            # Volume profile features
            df = self._calculate_volume_profile_features(df)
            
            # Market correlation features
            df = self._calculate_market_correlation_features(df)
            
            # Microstructure features
            df = self._calculate_microstructure_features(df)
            
            # Alternative data features
            df = self._calculate_alternative_features(df)
            
            # Deep learning-based features
            df = self._calculate_deep_features(df)
            
            # Cross-asset correlation features
            df = self._calculate_cross_asset_features(df)
            
            # If sentiment data is available
            if include_sentiment:
                df = self._calculate_sentiment_features(df)
            
            # Drop rows with NaN values from feature calculation
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        try:
            # Price changes
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            
            # Price ranges
            df['daily_range'] = df['high'] - df['low']
            df['daily_range_pct'] = df['daily_range'] / df['close']
            
            # Gap features
            df['gap'] = df['open'] - df['close'].shift(1)
            df['gap_pct'] = df['gap'] / df['close'].shift(1)
            
            # Price levels
            df['dist_from_high'] = df['high'].rolling(window=20).max() - df['close']
            df['dist_from_low'] = df['close'] - df['low'].rolling(window=20).min()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating price features: {str(e)}")
            return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'])
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            
            # Stochastic
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df

    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features"""
        try:
            # ATR and Normalized ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            df['natr'] = talib.NATR(df['high'], df['low'], df['close'])
            
            # Historical volatility
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Volatility regime
            df['high_volatility'] = df['volatility'] > df['volatility'].rolling(window=100).mean()
            
            # Garman-Klass volatility
            c = np.log(df['close'] / df['open'])
            h = np.log(df['high'] / df['open'])
            l = np.log(df['low'] / df['open'])
            df['gk_volatility'] = np.sqrt(0.5 * (h - l)**2 - (2*np.log(2) - 1) * c**2)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating volatility features: {str(e)}")
            return df

    def _calculate_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features"""
        try:
            # Trend strength
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            
            # Trend direction
            plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'])
            minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'])
            df['trend_direction'] = np.where(plus_di > minus_di, 1, -1)
            
            # Aroon indicators
            aroon_up, aroon_down = talib.AROON(df['high'], df['low'])
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            df['aroon_signal'] = aroon_up - aroon_down
            
            # Market phases
            df['market_phase'] = pd.cut(df['rsi'], 
                                      bins=[0, 30, 45, 55, 70, 100],
                                      labels=['oversold', 'weak', 'neutral', 'strong', 'overbought'])
            
            return df
        except Exception as e:
            logger.error(f"Error calculating market regime features: {str(e)}")
            return df

    def _calculate_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume profile features"""
        try:
            # On-Balance Volume
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Accumulation/Distribution Line
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # Chaikin Money Flow
            df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
            
            # Volume-price trend
            df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
            
            # Volume moving averages
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating volume profile features: {str(e)}")
            return df

    def _calculate_market_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market correlation features"""
        try:
            # Download market data
            spy = yf.download('^GSPC', start=df.index[0], end=df.index[-1])
            vix = yf.download('^VIX', start=df.index[0], end=df.index[-1])
            
            # Calculate correlations
            df['spy_correlation'] = df['close'].rolling(window=20).corr(spy['Close'])
            df['market_beta'] = df['returns'].rolling(window=20).cov(spy['Close'].pct_change()) / \
                               spy['Close'].pct_change().rolling(window=20).var()
            
            # VIX relationship
            df['vix_level'] = vix['Close']
            df['vix_change'] = vix['Close'].pct_change()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating market correlation features: {str(e)}")
            return df

    def _calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment-based features"""
        try:
            if 'news_sentiment' in df.columns:
                # Aggregate daily sentiment
                df['sentiment_ma'] = df['news_sentiment'].rolling(window=5).mean()
                df['sentiment_std'] = df['news_sentiment'].rolling(window=5).std()
                
                # Sentiment regime
                df['sentiment_regime'] = pd.qcut(df['sentiment_ma'], 
                                               q=5, 
                                               labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive'])
                
                # Sentiment momentum
                df['sentiment_momentum'] = df['sentiment_ma'] - df['sentiment_ma'].shift(5)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating sentiment features: {str(e)}")
            return df

    def _calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        try:
            # Bid-ask spread approximation (using high-low range as proxy)
            df['spread_proxy'] = (df['high'] - df['low']) / df['close']
            
            # Trade size analysis
            df['avg_trade_size'] = df['volume'] * df['close'] / df['volume'].rolling(window=20).mean()
            
            # Price impact
            df['amihud_illiquidity'] = abs(df['returns']) / (df['volume'] * df['close'])
            
            # Volume-weighted average price (VWAP)
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
            df['vwap'] = df['vwap'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # Price efficiency ratio
            log_returns = np.log(df['close'] / df['close'].shift(1))
            df['price_efficiency'] = abs(log_returns.rolling(window=20).sum()) / \
                                   log_returns.abs().rolling(window=20).sum()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {str(e)}")
            return df

    def _calculate_alternative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate alternative data features"""
        try:
            # Options market indicators (if available)
            if 'put_volume' in df.columns and 'call_volume' in df.columns:
                df['put_call_ratio'] = df['put_volume'] / df['call_volume']
                df['put_call_ratio_ma'] = df['put_call_ratio'].rolling(window=5).mean()
            
            # Short interest features (if available)
            if 'short_interest' in df.columns:
                df['short_interest_ratio'] = df['short_interest'] / df['volume']
                df['short_interest_days'] = df['short_interest'] / df['volume'].rolling(window=20).mean()
            
            # Institutional ownership features (if available)
            if 'institutional_ownership' in df.columns:
                df['inst_ownership_change'] = df['institutional_ownership'].pct_change()
                df['inst_ownership_zscore'] = (df['institutional_ownership'] - 
                                             df['institutional_ownership'].rolling(window=60).mean()) / \
                                             df['institutional_ownership'].rolling(window=60).std()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating alternative features: {str(e)}")
            return df

    def _calculate_deep_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate deep learning-based features"""
        try:
            # Time series decomposition
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Decompose log prices
            log_prices = np.log(df['close'])
            decomposition = seasonal_decompose(log_prices, period=20, extrapolate_trend='freq')
            
            df['trend_component'] = decomposition.trend
            df['seasonal_component'] = decomposition.seasonal
            df['residual_component'] = decomposition.resid
            
            # Wavelet features
            from pywt import dwt
            
            # Calculate wavelet transforms for different frequencies
            for level in range(1, 4):
                cA, cD = dwt(df['close'].values, 'haar')
                df[f'wavelet_approx_{level}'] = np.pad(cA, (0, len(df) - len(cA)), 'edge')
                df[f'wavelet_detail_{level}'] = np.pad(cD, (0, len(df) - len(cD)), 'edge')
            
            return df
        except Exception as e:
            logger.error(f"Error calculating deep features: {str(e)}")
            return df

    def _calculate_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-asset correlation features"""
        try:
            # Download related asset data
            end_date = df.index[-1]
            start_date = df.index[0]
            
            # Get data for different asset classes
            assets = {
                'bonds': '^TNX',  # 10-year Treasury yield
                'gold': 'GLD',    # Gold ETF
                'oil': 'USO',     # Oil ETF
                'dollar': 'UUP'   # US Dollar ETF
            }
            
            asset_data = {}
            for asset_name, ticker in assets.items():
                try:
                    data = yf.download(ticker, start=start_date, end=end_date)
                    asset_data[asset_name] = data['Close']
                except:
                    logger.warning(f"Could not fetch data for {ticker}")
            
            # Calculate correlations with each asset
            window = 20
            for asset_name, prices in asset_data.items():
                df[f'{asset_name}_corr'] = df['close'].rolling(window=window).corr(prices)
                df[f'{asset_name}_beta'] = (df['returns'].rolling(window=window).cov(prices.pct_change()) / 
                                          prices.pct_change().rolling(window=window).var())
            
            # Calculate regime based on cross-asset correlations
            correlation_cols = [col for col in df.columns if col.endswith('_corr')]
            if correlation_cols:
                df['cross_asset_regime'] = df[correlation_cols].mean(axis=1)
                df['cross_asset_regime'] = pd.qcut(df['cross_asset_regime'], 
                                                 q=5, 
                                                 labels=['very_low', 'low', 'neutral', 'high', 'very_high'])
            
            return df
        except Exception as e:
            logger.error(f"Error calculating cross-asset features: {str(e)}")
            return df

    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            importance_scores = {}
            
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importance_scores = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For linear models
                importance_scores = dict(zip(feature_names, np.abs(model.coef_)))
            
            # Sort by importance
            importance_scores = dict(sorted(importance_scores.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True))
            
            return importance_scores
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
