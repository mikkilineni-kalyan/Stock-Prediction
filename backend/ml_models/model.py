import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from .indicators import TechnicalAnalysis
import random

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.historical_data = None
        
    def get_historical_data(self):
        return self.historical_data
        
    def predict(self, symbol):
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=120)
            
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise Exception(f"No data found for symbol: {symbol}")
            
            # Store historical data
            self.historical_data = df
            
            # Convert numpy arrays to lists and numpy values to native Python types
            dates = df.index[-30:].strftime('%Y-%m-%d').tolist()
            actual_prices = [float(x) for x in df['Close'].values[-30:]]
            
            # Calculate price momentum and volatility
            price_momentum = float((df['Close'].values[-1] - df['Close'].values[-5]) / df['Close'].values[-5])
            volatility = float(df['Close'].pct_change().std())
            
            # Calculate next day's predicted price
            last_price = float(actual_prices[-1])
            predicted_change = price_momentum + (volatility if price_momentum > 0 else -volatility)
            next_price = float(last_price * (1 + predicted_change))
            
            # Create predicted prices array
            predicted_prices = [float(x) for x in actual_prices[:-1]] + [next_price]
            
            # Calculate confidence interval
            confidence = 0.95
            std_dev = float(df['Close'].pct_change().std())
            lower_bound = float(next_price * (1 - confidence * std_dev))
            upper_bound = float(next_price * (1 + confidence * std_dev))
            
            # Calculate technical indicators
            prices = df['Close']
            ma20 = float(prices.rolling(window=20).mean().iloc[-1])
            ma50 = float(prices.rolling(window=50).mean().iloc[-1])
            
            # Calculate RSI
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])
            
            # Calculate MACD
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = float((exp1 - exp2).iloc[-1])
            
            # Calculate Bollinger Bands
            middle = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            upper_band = float((middle + (std * 2)).iloc[-1])
            lower_band = float((middle - (std * 2)).iloc[-1])
            middle_band = float(middle.iloc[-1])
            
            return {
                'dates': dates,
                'actual_prices': actual_prices,
                'predicted_prices': predicted_prices,
                'symbol': symbol,
                'last_price': last_price,
                'predicted_price': next_price,
                'change_percent': float((next_price - last_price) / last_price * 100),
                'technicalIndicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'bollingerBands': {
                        'upper': upper_band,
                        'middle': middle_band,
                        'lower': lower_band
                    },
                    'volume': float(df['Volume'].iloc[-1]),
                    'ma20': ma20,
                    'ma50': ma50,
                    'volatility': volatility
                },
                'confidence_interval': {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
            }
            
        except Exception as e:
            print(f"Error predicting for {symbol}: {str(e)}")
            raise Exception(f"Prediction error: {str(e)}")

    def get_historical_metrics(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get historical data
            hist_ytd = stock.history(period='ytd')
            hist_1y = stock.history(period='1y')
            hist_3y = stock.history(period='3y')
            
            # Calculate returns with error handling
            ytd_return = float(hist_ytd['Close'].pct_change().sum() * 100) if not hist_ytd.empty else 0
            one_year_return = float(hist_1y['Close'].pct_change().sum() * 100) if not hist_1y.empty else 0
            three_year_return = float(hist_3y['Close'].pct_change().sum() * 100) if not hist_3y.empty else 0
            
            return {
                'yearToDate': ytd_return,
                'oneYear': one_year_return,
                'threeYear': three_year_return,
                'beta': float(info.get('beta', 0) or 0),
                'peRatio': float(info.get('forwardPE', 0) or 0),
                'marketCap': float(info.get('marketCap', 0) or 0),
                'volume': float(info.get('volume', 0) or 0)
            }
        except Exception as e:
            print(f"Error getting historical metrics: {str(e)}")
            # Return default values if there's an error
            return {
                'yearToDate': 0,
                'oneYear': 0,
                'threeYear': 0,
                'beta': 0,
                'peRatio': 0,
                'marketCap': 0,
                'volume': 0
            }

    def get_news(self, symbol):
        stock = yf.Ticker(symbol)
        news = stock.news
        return [
            {
                'title': item.get('title'),
                'summary': item.get('summary'),
                'url': item.get('link'),
                'source': item.get('source'),
                'publishedAt': item.get('providerPublishTime')
            }
            for item in news[:5]  # Get latest 5 news items
        ]

    def get_market_sentiment(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            
            # Get analyst recommendations
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                latest_rec = recommendations.iloc[-1]
            else:
                latest_rec = pd.Series()
            
            # Get news sentiment (simplified example)
            news_sentiment = self.analyze_news_sentiment(symbol)
            
            # Get social sentiment (simplified example)
            social_sentiment = self.analyze_social_sentiment(symbol)
            
            return {
                'newsScore': news_sentiment,
                'socialScore': social_sentiment,
                'analystRating': latest_rec.get('To Grade', 'N/A'),
                'recommendationTrend': {
                    'strongBuy': int(recommendations['To Grade'].value_counts().get('Strong Buy', 0)),
                    'buy': int(recommendations['To Grade'].value_counts().get('Buy', 0)),
                    'hold': int(recommendations['To Grade'].value_counts().get('Hold', 0)),
                    'sell': int(recommendations['To Grade'].value_counts().get('Sell', 0)),
                    'strongSell': int(recommendations['To Grade'].value_counts().get('Strong Sell', 0))
                }
            }
        except Exception as e:
            print(f"Error getting market sentiment: {str(e)}")
            return {
                'newsScore': 0.5,
                'socialScore': 0.5,
                'analystRating': 'N/A',
                'recommendationTrend': {
                    'strongBuy': 0,
                    'buy': 0,
                    'hold': 0,
                    'sell': 0,
                    'strongSell': 0
                }
            }

    def analyze_news_sentiment(self, symbol):
        # Implement news sentiment analysis here
        # This is a placeholder that returns a random score
        return random.uniform(0.3, 0.8)

    def analyze_social_sentiment(self, symbol):
        # Implement social media sentiment analysis here
        # This is a placeholder that returns a random score
        return random.uniform(0.3, 0.8)
