import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import yfinance as yf
from textblob import TextBlob
import json
import os

class NewsAnalyzer:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from configuration file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def fetch_news(self, ticker: str, days: int = 1) -> List[Dict[str, Any]]:
        """Fetch news from multiple sources for a given ticker"""
        news_items = []
        
        # Alpha Vantage News (if API key is available)
        if 'ALPHA_VANTAGE_KEY' in self.api_keys:
            av_news = self._fetch_alpha_vantage_news(ticker)
            news_items.extend(av_news)
        
        # Yahoo Finance News (free)
        yf_news = self._fetch_yahoo_finance_news(ticker, days)
        news_items.extend(yf_news)
        
        return news_items

    def _fetch_yahoo_finance_news(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        """Fetch news from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            processed_news = []
            for item in news:
                processed_news.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': 'Yahoo Finance',
                    'url': item.get('link', ''),
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'sentiment': self._analyze_sentiment(f"{item.get('title', '')} {item.get('summary', '')}")
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {str(e)}")
            return []

    def _fetch_alpha_vantage_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage"""
        try:
            api_key = self.api_keys.get('ALPHA_VANTAGE_KEY')
            if not api_key:
                return []
                
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
            response = requests.get(url)
            data = response.json()
            
            processed_news = []
            for item in data.get('feed', []):
                processed_news.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'published_at': datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                    'sentiment': float(item.get('overall_sentiment_score', 0))
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching Alpha Vantage news: {str(e)}")
            return []

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            analysis = TextBlob(text)
            # Convert polarity (-1 to 1) to our scale (1 to 5)
            return (analysis.sentiment.polarity + 1) * 2 + 1
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return 3.0  # Neutral sentiment as fallback

    def predict_impact(self, ticker: str) -> Dict[str, Any]:
        """Predict market impact based on news analysis"""
        news_items = self.fetch_news(ticker)
        if not news_items:
            return {
                'score': 0,
                'impact': 'neutral',
                'confidence': 0,
                'news_count': 0,
                'average_sentiment': 3.0,
                'latest_news': []
            }

        # Calculate average sentiment and impact score
        sentiments = [item['sentiment'] for item in news_items]
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        # Determine impact
        if avg_sentiment > 3.5:
            impact = 'positive'
        elif avg_sentiment < 2.5:
            impact = 'negative'
        else:
            impact = 'neutral'
        
        # Calculate confidence based on number of news items and sentiment consistency
        news_count_factor = min(len(news_items) / 10, 1)  # Max out at 10 news items
        consistency_factor = 1 - min(sentiment_std / 2, 0.5)  # Lower std = higher consistency
        confidence = (news_count_factor + consistency_factor) / 2
        
        # Calculate impact score (1-5)
        score = min(abs(avg_sentiment - 3) * 2 + 3, 5)
        
        return {
            'score': round(score, 1),
            'impact': impact,
            'confidence': round(confidence * 100, 1),
            'news_count': len(news_items),
            'average_sentiment': round(avg_sentiment, 2),
            'latest_news': sorted(news_items, key=lambda x: x['published_at'], reverse=True)[:5]
        }

    def analyze_historical_correlation(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Analyze correlation between news sentiment and price movements"""
        try:
            # Fetch historical news and price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get stock data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date)
            
            # Get news data
            news_items = self.fetch_news(ticker, days)
            
            if hist_data.empty or not news_items:
                return {
                    'correlation': 0,
                    'accuracy': 0,
                    'sample_size': 0
                }
            
            # Create daily sentiment scores
            daily_sentiments = {}
            for item in news_items:
                date = item['published_at'].date()
                if date not in daily_sentiments:
                    daily_sentiments[date] = []
                daily_sentiments[date].append(item['sentiment'])
            
            # Calculate average daily sentiment
            sentiment_df = pd.DataFrame(
                [(date, np.mean(scores)) for date, scores in daily_sentiments.items()],
                columns=['Date', 'Sentiment']
            ).set_index('Date')
            
            # Calculate daily returns
            hist_data['Returns'] = hist_data['Close'].pct_change()
            
            # Merge data
            merged_data = pd.merge(
                hist_data['Returns'],
                sentiment_df,
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            # Calculate correlation and prediction accuracy
            correlation = merged_data['Returns'].corr(merged_data['Sentiment'])
            
            # Calculate prediction accuracy
            correct_predictions = sum(
                (merged_data['Returns'] > 0) == (merged_data['Sentiment'] > 3)
            )
            accuracy = correct_predictions / len(merged_data) if len(merged_data) > 0 else 0
            
            return {
                'correlation': round(correlation, 2),
                'accuracy': round(accuracy * 100, 1),
                'sample_size': len(merged_data)
            }
            
        except Exception as e:
            print(f"Error analyzing historical correlation: {str(e)}")
            return {
                'correlation': 0,
                'accuracy': 0,
                'sample_size': 0
            }
