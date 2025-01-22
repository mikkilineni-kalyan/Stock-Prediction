import yfinance as yf
from newsapi import NewsApiClient
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Change relative imports to absolute
from api.config import Config
from utils.api_manager import APIKeyManager

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.news_api = NewsApiClient(api_key=Config.NEWS_API_KEY)
        self.api_manager = APIKeyManager()
        
    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with fallback mechanisms"""
        data = None
        errors = []

        # Try Finnhub if available
        if self.api_manager.can_make_request('finnhub'):
            try:
                # Finnhub data fetching logic...
                pass
            except Exception as e:
                errors.append(f"Finnhub error: {str(e)}")
        
        # Try AlphaVantage if Finnhub failed
        if data is None:
            try:
                # AlphaVantage data fetching logic...
                pass
            except Exception as e:
                errors.append(f"AlphaVantage error: {str(e)}")
        
        # Fallback to yfinance if all else fails
        if data is None:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
            except Exception as e:
                errors.append(f"yfinance error: {str(e)}")
        
        if data is None:
            logger.error(f"All data sources failed: {', '.join(errors)}")
            raise Exception("Unable to fetch stock data from any source")
        
        return data
    
    def get_news(self, ticker: str, company_name: str, days: int = 7) -> List[Dict[Any, Any]]:
        """Fetch news articles with rate limiting"""
        if not self.api_manager.can_make_request('newsapi'):
            logger.warning("Rate limit reached for NewsAPI")
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            articles = self.news_api.get_everything(
                q=f'({ticker} OR "{company_name}") AND (stock OR market OR trading)',
                from_param=from_date,
                language='en',
                sort_by='relevancy'
            )
            
            self.api_manager.log_request('newsapi')
            return articles['articles'] if 'articles' in articles else []
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return [] 