import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    def __init__(self):
        self.required_periods = {
            'RSI': 14,
            'MACD': 26,
            'BB': 20,
            'SMA': [20, 50, 200]  # Multiple SMAs
        }

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Basic indicators
        indicators['RSI'] = self.calculate_rsi(data)
        indicators['MACD'] = self.calculate_macd(data)
        indicators['BB'] = self.calculate_bollinger_bands(data)
        indicators['SMA'] = self.calculate_sma(data)
        
        # Volume indicators
        indicators['OBV'] = self.calculate_obv(data)
        indicators['Volume_MA'] = self.calculate_volume_ma(data)
        
        # Trend indicators
        indicators['ADX'] = self.calculate_adx(data)
        
        # Pattern recognition
        indicators['Patterns'] = self.identify_patterns(data)
        
        return indicators

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]

    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': macd.iloc[-1] - signal.iloc[-1]
        }

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1]
        }

    def calculate_sma(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Simple Moving Averages"""
        sma_values = {}
        for period in self.required_periods['SMA']:
            sma_values[f'SMA_{period}'] = data['Close'].rolling(window=period).mean().iloc[-1]
        return sma_values

    def calculate_obv(self, data: pd.DataFrame) -> float:
        """Calculate On-Balance Volume"""
        obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
        return obv.iloc[-1]

    def calculate_volume_ma(self, data: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calculate Volume Moving Average"""
        vol_ma = data['Volume'].rolling(window=period).mean()
        current_vol = data['Volume'].iloc[-1]
        
        return {
            'current_volume': current_vol,
            'volume_ma': vol_ma.iloc[-1],
            'volume_ratio': current_vol / vol_ma.iloc[-1]
        }

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate ADX
        tr_smoothed = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_smoothed)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_smoothed)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1]

    def identify_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Identify common chart patterns"""
        patterns = {
            'double_top': False,
            'double_bottom': False,
            'head_shoulders': False,
            'triangle': False
        }
        
        # Simple pattern detection logic
        close = data['Close'].values
        
        # Double top/bottom detection
        peaks = self._find_peaks(close)
        troughs = self._find_peaks(-close)
        
        if len(peaks) >= 2:
            patterns['double_top'] = self._check_double_pattern(close[peaks])
        if len(troughs) >= 2:
            patterns['double_bottom'] = self._check_double_pattern(close[troughs])
            
        return patterns

    def _find_peaks(self, arr: np.array, threshold: float = 0.02) -> np.array:
        """Find peaks in price data"""
        peaks = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                if not peaks or abs(arr[i] - arr[peaks[-1]]) > threshold:
                    peaks.append(i)
        return np.array(peaks)

    def _check_double_pattern(self, peaks: np.array, threshold: float = 0.02) -> bool:
        """Check if peaks form a double top/bottom pattern"""
        if len(peaks) < 2:
            return False
        return abs(peaks[-1] - peaks[-2]) / peaks[-2] < threshold 