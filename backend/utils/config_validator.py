import re
from typing import Dict, List
import requests
from ..api.config import Config

class ConfigValidator:
    @staticmethod
    def validate_api_key_format(key: str, service: str) -> bool:
        """Validate API key format"""
        patterns = {
            'newsapi': r'^[a-f0-9]{32}$',
            'finnhub': r'^[A-Za-z0-9]{20}$',
            'alphavantage': r'^[A-Z0-9]{16}$',
        }
        
        if service not in patterns:
            return True
        
        return bool(re.match(patterns[service], key))
    
    @staticmethod
    def test_api_connectivity() -> Dict[str, bool]:
        """Test connectivity to all configured APIs"""
        results = {}
        
        # Test NewsAPI
        if Config.NEWS_API_KEY:
            try:
                response = requests.get(
                    'https://newsapi.org/v2/top-headlines',
                    params={'apiKey': Config.NEWS_API_KEY, 'category': 'business', 'pageSize': 1}
                )
                results['newsapi'] = response.status_code == 200
            except:
                results['newsapi'] = False
        
        # Add tests for other APIs...
        
        return results 