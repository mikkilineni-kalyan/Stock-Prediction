import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Moving Averages
        indicators.update(TechnicalIndicators.calculate_moving_averages(df))
        
        # Momentum Indicators
        indicators.update(TechnicalIndicators.calculate_momentum_indicators(df))
        
        # Volatility Indicators
        indicators.update(TechnicalIndicators.calculate_volatility_indicators(df))
        
        # Volume Indicators
        indicators.update(TechnicalIndicators.calculate_volume_indicators(df))
        
        # Price Patterns
        indicators.update(TechnicalIndicators.calculate_price_patterns(df))
        
        return indicators
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> Dict[str, float]:
        ma_periods = [5, 10, 20, 50, 200]
        indicators = {}
        
        for period in ma_periods:
            indicators[f'MA{period}'] = df['Close'].rolling(window=period).mean().iloc[-1]
            indicators[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean().iloc[-1]
        
        return indicators
    
    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> Dict[str, float]:
        indicators = {}
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = float(100 - (100 / (1 + rs)).iloc[-1])
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators['MACD'] = float(macd.iloc[-1])
        indicators['MACD_Signal'] = float(signal.iloc[-1])
        
        return indicators
    
    @staticmethod
    def calculate_volatility_indicators(df: pd.DataFrame) -> Dict[str, float]:
        indicators = {}
        
        # Bollinger Bands
        ma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = float(ma20.iloc[-1] + (std20.iloc[-1] * 2))
        indicators['BB_Lower'] = float(ma20.iloc[-1] - (std20.iloc[-1] * 2))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        indicators['ATR'] = float(true_range.rolling(14).mean().iloc[-1])
        
        return indicators
    
    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, float]:
        indicators = {}
        
        # Volume Moving Average
        indicators['Volume_MA'] = float(df['Volume'].rolling(window=20).mean().iloc[-1])
        
        # Price-Volume Trend
        indicators['PVT'] = float((df['Volume'] * ((df['Close'] - df['Close'].shift()) / df['Close'].shift())).cumsum().iloc[-1])
        
        return indicators
    
    @staticmethod
    def calculate_price_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """Identify common price patterns"""
        patterns = {}
        
        # Support and Resistance
        last_price = df['Close'].iloc[-1]
        recent_lows = df['Low'].rolling(window=20).min()
        recent_highs = df['High'].rolling(window=20).max()
        
        patterns['near_support'] = last_price <= recent_lows.iloc[-1] * 1.02
        patterns['near_resistance'] = last_price >= recent_highs.iloc[-1] * 0.98
        
        # Trend Detection
        ma50 = df['Close'].rolling(window=50).mean()
        ma200 = df['Close'].rolling(window=200).mean()
        
        patterns['golden_cross'] = ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]
        patterns['death_cross'] = ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]
        
        return patterns 