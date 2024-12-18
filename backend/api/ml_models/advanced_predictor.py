import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedPredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def calculate_technical_indicators(self, data):
        # Calculate basic technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        return data
    
    def predict(self, symbol):
        try:
            # Get historical data
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            hist_data = stock.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                return {
                    'prediction': 'neutral',
                    'confidence': 0.5,
                    'technical_indicators': {},
                    'price_targets': {'low': 0, 'high': 0}
                }
            
            # Calculate technical indicators
            tech_data = self.calculate_technical_indicators(hist_data.copy())
            
            # Get latest values
            latest = tech_data.iloc[-1]
            prev = tech_data.iloc[-2]
            
            # Simple trading signals
            signals = []
            confidence_factors = []
            
            # SMA signals
            if latest['Close'] > latest['SMA_20'] and latest['Close'] > latest['SMA_50']:
                signals.append('bullish')
                confidence_factors.append(0.6)
            elif latest['Close'] < latest['SMA_20'] and latest['Close'] < latest['SMA_50']:
                signals.append('bearish')
                confidence_factors.append(0.6)
            
            # RSI signals
            if latest['RSI'] < 30:
                signals.append('bullish')  # Oversold
                confidence_factors.append(0.7)
            elif latest['RSI'] > 70:
                signals.append('bearish')  # Overbought
                confidence_factors.append(0.7)
            
            # MACD signals
            if latest['MACD'] > latest['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']:
                signals.append('bullish')
                confidence_factors.append(0.65)
            elif latest['MACD'] < latest['Signal_Line'] and prev['MACD'] >= prev['Signal_Line']:
                signals.append('bearish')
                confidence_factors.append(0.65)
            
            # Make final prediction
            if not signals:
                prediction = 'neutral'
                confidence = 0.5
            else:
                bullish_count = sum(1 for s in signals if s == 'bullish')
                bearish_count = sum(1 for s in signals if s == 'bearish')
                
                if bullish_count > bearish_count:
                    prediction = 'bullish'
                    confidence = sum(cf for s, cf in zip(signals, confidence_factors) if s == 'bullish') / bullish_count
                elif bearish_count > bullish_count:
                    prediction = 'bearish'
                    confidence = sum(cf for s, cf in zip(signals, confidence_factors) if s == 'bearish') / bearish_count
                else:
                    prediction = 'neutral'
                    confidence = 0.5
            
            # Calculate simple price targets
            latest_price = latest['Close']
            volatility = hist_data['Close'].pct_change().std()
            price_targets = {
                'low': round(latest_price * (1 - volatility * 2), 2),
                'high': round(latest_price * (1 + volatility * 2), 2)
            }
            
            # Get technical indicators for response
            technical_indicators = {
                'RSI': round(latest['RSI'], 2),
                'MACD': round(latest['MACD'], 4),
                'Signal_Line': round(latest['Signal_Line'], 4),
                'SMA_20': round(latest['SMA_20'], 2),
                'SMA_50': round(latest['SMA_50'], 2)
            }
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 2),
                'technical_indicators': technical_indicators,
                'price_targets': price_targets
            }
            
        except Exception as e:
            print(f"Error in advanced prediction: {e}")
            return {
                'prediction': 'neutral',
                'confidence': 0.5,
                'technical_indicators': {},
                'price_targets': {'low': 0, 'high': 0}
            }
