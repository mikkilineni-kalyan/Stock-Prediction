import pandas as pd
import numpy as np

class TechnicalAnalysis:
    def __init__(self, price_data):
        self.df = price_data
        
    def get_indicators(self):
        prices = self.df['Close']
        
        # Calculate moving averages
        ma20 = prices.rolling(window=20).mean().iloc[-1]
        ma50 = prices.rolling(window=50).mean().iloc[-1]
        
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Calculate MACD
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = (exp1 - exp2).iloc[-1]
        
        # Calculate Bollinger Bands
        middle = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        upper = (middle + (std * 2)).iloc[-1]
        lower = (middle - (std * 2)).iloc[-1]
        middle = middle.iloc[-1]
        
        # Calculate volatility
        volatility = prices.pct_change().std()
        
        return {
            'rsi': rsi,
            'macd': macd,
            'bollingerBands': {
                'upper': upper,
                'middle': middle,
                'lower': lower
            },
            'volume': self.df['Volume'].iloc[-1],
            'ma20': ma20,
            'ma50': ma50,
            'volatility': volatility
        }
