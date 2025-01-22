import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from typing import Dict, List
import requests

logger = logging.getLogger(__name__)

class NewsPredictor:
    def __init__(self):
        """Initialize with basic sentiment analysis capabilities"""
        self.positive_words = {
            'surge', 'jump', 'rise', 'gain', 'up', 'high', 'growth', 'profit',
            'bullish', 'outperform', 'beat', 'exceed', 'positive', 'strong',
            'upgrade', 'buy', 'recommend', 'success', 'innovative', 'breakthrough'
        }
        self.negative_words = {
            'fall', 'drop', 'decline', 'loss', 'down', 'low', 'bearish',
            'underperform', 'miss', 'weak', 'negative', 'downgrade', 'sell',
            'avoid', 'risk', 'crash', 'plunge', 'bankruptcy', 'lawsuit', 'debt'
        }
        
        # Initialize default response
        self.default_response = {
            'sentiment': {
                'overall_score': 2.5,  # Neutral score
                'confidence': 0.5,
                'news_count': 0,
                'reddit': None,
                'twitter': None
            },
            'news': []
        }

    def analyze_article(self, article: Dict) -> Dict:
        """Analyze a single news article and return sentiment details"""
        try:
            title = article.get('title', '').lower()
            if not title:
                return None
                
            # Count sentiment words
            pos_count = sum(1 for word in self.positive_words if word in title)
            neg_count = sum(1 for word in self.negative_words if word in title)
            
            # Calculate base sentiment (2.5 is neutral)
            if pos_count == neg_count:
                sentiment = 2.5
            elif pos_count > neg_count:
                # Scale positive sentiment between 2.5 and 4.5
                sentiment = 2.5 + min(2.0, (pos_count - neg_count) * 0.5)
            else:
                # Scale negative sentiment between 0.5 and 2.5
                sentiment = 2.5 - min(2.0, (neg_count - pos_count) * 0.5)
            
            return {
                'title': article.get('title', ''),
                'link': article.get('link', ''),
                'publisher': article.get('publisher', ''),
                'published': article.get('published', ''),
                'sentiment': float(sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing article: {str(e)}", exc_info=True)
            return None

    def predict_from_news(self, symbol: str) -> Dict:
        """
        Predict market sentiment based on recent stock news
        Returns a dictionary with prediction details including a sentiment score from 0.5 to 4.5
        """
        try:
            logger.info(f"Starting news prediction for {symbol}")
            
            # Get news from yfinance with error handling
            try:
                stock = yf.Ticker(symbol)
                news_data = stock.news
                
                # Validate news data
                if not isinstance(news_data, list):
                    logger.warning(f"Invalid news data format for {symbol}")
                    return self.default_response
                    
                # Filter out invalid news items
                news = [item for item in news_data if isinstance(item, dict) and 'title' in item]
                
                if not news:
                    logger.warning(f"No valid news found for {symbol}")
                    return self.default_response
                    
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {str(e)}", exc_info=True)
                return self.default_response
            
            # Analyze sentiment for each news item
            analyzed_articles = []
            for article in news[:10]:  # Analyze up to 10 most recent articles
                result = self.analyze_article(article)
                if result:
                    analyzed_articles.append(result)
            
            if not analyzed_articles:
                logger.warning(f"No articles could be analyzed for {symbol}")
                return self.default_response
            
            # Calculate overall sentiment
            sentiments = [article['sentiment'] for article in analyzed_articles]
            avg_sentiment = float(np.mean(sentiments))
            sentiment_std = float(np.std(sentiments))
            
            # Calculate confidence based on consistency of sentiments
            # Higher standard deviation = lower confidence
            confidence = float(max(0.5, min(1.0, 1.0 - (sentiment_std / 2))))
            
            result = {
                'sentiment': {
                    'overall_score': avg_sentiment,
                    'confidence': confidence,
                    'news_count': len(analyzed_articles),
                    'reddit': None,
                    'twitter': None
                },
                'news': analyzed_articles
            }
            
            logger.info(f"Successfully analyzed news for {symbol}: score={avg_sentiment:.2f}, confidence={confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_from_news: {str(e)}", exc_info=True)
            return self.default_response
