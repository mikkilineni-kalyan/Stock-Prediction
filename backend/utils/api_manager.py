from typing import Dict, Optional
import time
from datetime import datetime, timedelta
import requests
import logging
from api.config import Config

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def can_make_request(self) -> bool:
        current_time = time.time()
        # Remove calls older than 1 minute
        self.calls = [call for call in self.calls if call > current_time - 60]
        
        if len(self.calls) < self.calls_per_minute:
            self.calls.append(current_time)
            return True
        return False

class APIKeyManager:
    def __init__(self):
        self.rate_limiters = {
            'newsapi': RateLimiter(30),      # 30 calls per minute
            'finnhub': RateLimiter(60),      # 60 calls per minute
            'alphavantage': RateLimiter(5),  # 5 calls per minute
            'twitter': RateLimiter(180),     # 180 calls per 15 minutes
            'reddit': RateLimiter(60),       # 60 calls per minute
            'seekingalpha': RateLimiter(5)   # 5 calls per minute
        }
        
        # Initialize key status
        self.key_status = self._validate_all_keys()
    
    def _validate_all_keys(self) -> Dict[str, bool]:
        """Validate all API keys and return their status"""
        return {
            'newsapi': self._validate_newsapi_key(),
            'finnhub': self._validate_finnhub_key(),
            'alphavantage': self._validate_alphavantage_key(),
            'twitter': self._validate_twitter_keys(),
            'reddit': self._validate_reddit_keys(),
            'seekingalpha': self._validate_seekingalpha_key()
        }
    
    def _validate_newsapi_key(self) -> bool:
        """Validate NewsAPI key"""
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'apiKey': Config.NEWS_API_KEY,
                'category': 'business',
                'pageSize': 1
            }
            response = requests.get(url, params=params)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"NewsAPI key validation failed: {str(e)}")
            return False
    
    def _validate_finnhub_key(self) -> bool:
        """Validate Finnhub key"""
        try:
            url = f"https://finnhub.io/api/v1/quote"
            params = {
                'symbol': 'AAPL',
                'token': Config.FINNHUB_API_KEY
            }
            response = requests.get(url, params=params)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Finnhub key validation failed: {str(e)}")
            return False

    def _validate_alphavantage_key(self) -> bool:
        """Validate Alpha Vantage key"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'AAPL',
                'apikey': Config.ALPHA_VANTAGE_API_KEY
            }
            response = requests.get(url, params=params)
            return response.status_code == 200 and 'Error Message' not in response.json()
        except Exception as e:
            logger.error(f"Alpha Vantage key validation failed: {str(e)}")
            return False

    def _validate_twitter_keys(self) -> bool:
        """Validate Twitter API keys"""
        return bool(Config.TWITTER_API_KEY and Config.TWITTER_API_SECRET)

    def _validate_reddit_keys(self) -> bool:
        """Validate Reddit API keys"""
        return bool(Config.REDDIT_CLIENT_ID and Config.REDDIT_CLIENT_SECRET)

    def _validate_seekingalpha_key(self) -> bool:
        """Validate Seeking Alpha API key"""
        return bool(Config.SEEKING_ALPHA_API_KEY)
    
    def can_make_request(self, api_name: str) -> bool:
        """Check if we can make a request to the specified API"""
        if api_name not in self.rate_limiters:
            return True
        
        if not self.key_status.get(api_name, False):
            logger.warning(f"{api_name} key is invalid or not configured")
            return False
        
        return self.rate_limiters[api_name].can_make_request()
    
    def log_request(self, api_name: str):
        """Log a request to the rate limiter"""
        if api_name in self.rate_limiters:
            self.rate_limiters[api_name].calls.append(time.time()) 