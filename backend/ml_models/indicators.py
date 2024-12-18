import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import talib

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        self.periods = {
            'short': 14,
            'medium': 50,
            'long': 200
        }
    
    def calculate_all(self, data):
        try:
            return {
                'trend': self._calculate_trend_indicators(data),
                'momentum': self._calculate_momentum_indicators(data),
                'volatility': self._calculate_volatility_indicators(data),
                'volume': self._calculate_volume_indicators(data)
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def _calculate_trend_indicators(self, data):
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        return {
            'sma': {
                'short': talib.SMA(close, timeperiod=self.periods['short']),
                'medium': talib.SMA(close, timeperiod=self.periods['medium']),
                'long': talib.SMA(close, timeperiod=self.periods['long'])
            },
            'ema': {
                'short': talib.EMA(close, timeperiod=self.periods['short']),
                'medium': talib.EMA(close, timeperiod=self.periods['medium'])
            },
            'macd': talib.MACD(close),
            'adx': talib.ADX(high, low, close, timeperiod=self.periods['short']),
            'parabolic_sar': talib.SAR(high, low),
            'ichimoku': self._calculate_ichimoku(data)
        }

    def _calculate_momentum_indicators(self, data):
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        return {
            'rsi': talib.RSI(close, timeperiod=self.periods['short']),
            'stoch': talib.STOCH(high, low, close),
            'cci': talib.CCI(high, low, close, timeperiod=self.periods['short']),
            'mfi': talib.MFI(high, low, close, data['Volume'].values, timeperiod=self.periods['short']),
            'williams_r': talib.WILLR(high, low, close, timeperiod=self.periods['short']),
            'ultimate_oscillator': talib.ULTOSC(high, low, close)
        }

    def _calculate_volatility_indicators(self, data):
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        return {
            'bollinger': talib.BBANDS(close),
            'atr': talib.ATR(high, low, close, timeperiod=self.periods['short']),
            'natr': talib.NATR(high, low, close, timeperiod=self.periods['short']),
            'standard_deviation': talib.STDDEV(close, timeperiod=self.periods['short'])
        }

    def _calculate_volume_indicators(self, data):
        close = data['Close'].values
        volume = data['Volume'].values
        
        return {
            'obv': talib.OBV(close, volume),
            'ad': talib.AD(data['High'].values, data['Low'].values, close, volume),
            'adosc': talib.ADOSC(data['High'].values, data['Low'].values, close, volume),
            'volume_sma': talib.SMA(volume, timeperiod=self.periods['short'])
        }

    def _calculate_ichimoku(self, data):
        high = data['High'].values
        low = data['Low'].values
        
        tenkan_period = 9
        kijun_period = 26
        senkou_span_b_period = 52
        
        tenkan_sen = self._calculate_ichimoku_line(high, low, tenkan_period)
        kijun_sen = self._calculate_ichimoku_line(high, low, kijun_period)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = self._calculate_ichimoku_line(high, low, senkou_span_b_period)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b
        }

    def _calculate_ichimoku_line(self, high, low, period):
        highs = pd.Series(high).rolling(window=period).max()
        lows = pd.Series(low).rolling(window=period).min()
        return (highs + lows) / 2

    def _calculate_advanced_indicators(self, data):
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        return {
            'trend': {
                'sma': self._calculate_multiple_smas(close),
                'ema': self._calculate_multiple_emas(close),
                'macd': self._calculate_macd(close),
                'adx': self._calculate_adx(high, low, close),
                'supertrend': self._calculate_supertrend(high, low, close),
                'ichimoku': self._calculate_ichimoku(high, low, close)
            },
            'momentum': {
                'rsi': self._calculate_rsi(close),
                'stochastic': self._calculate_stochastic(high, low, close),
                'cci': self._calculate_cci(high, low, close),
                'williams_r': self._calculate_williams_r(high, low, close),
                'ultimate_oscillator': self._calculate_ultimate_oscillator(high, low, close)
            },
            'volume': {
                'obv': self._calculate_obv(close, volume),
                'cmf': self._calculate_cmf(high, low, close, volume),
                'mfi': self._calculate_mfi(high, low, close, volume),
                'vwap': self._calculate_vwap(high, low, close, volume)
            }
        }
































































