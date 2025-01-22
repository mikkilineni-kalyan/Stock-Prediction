import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging
from dotenv import load_dotenv
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys, including validation and rotation tracking."""
    
    def __init__(self):
        self.key_history_file = Path(__file__).parent.parent / 'data' / 'key_history.json'
        self.key_history_file.parent.mkdir(exist_ok=True)
        self.load_dotenv()
        self.load_key_history()
    
    def load_dotenv(self):
        """Load environment variables."""
        load_dotenv()
        self.required_keys = {
            'alpha_vantage': 'ALPHA_VANTAGE_API_KEY',
            'news_api': 'NEWS_API_KEY',
            'finnhub': 'FINNHUB_API_KEY',
            'polygon': 'POLYGON_API_KEY'
        }
        
        self.optional_keys = {
            'twitter': ['TWITTER_API_KEY', 'TWITTER_API_SECRET'],
            'seeking_alpha': 'SEEKING_ALPHA_API_KEY',
            'sec': 'SEC_API_KEY'
        }
    
    def load_key_history(self):
        """Load key rotation history."""
        if self.key_history_file.exists():
            with open(self.key_history_file, 'r') as f:
                self.key_history = json.load(f)
        else:
            self.key_history = {}
            self.save_key_history()
    
    def save_key_history(self):
        """Save key rotation history."""
        with open(self.key_history_file, 'w') as f:
            json.dump(self.key_history, f, indent=4)
    
    def validate_key(self, service: str) -> tuple[bool, str]:
        """Validate an API key by making a test request."""
        key_name = self.required_keys.get(service)
        if not key_name:
            return False, f"Unknown service: {service}"
        
        api_key = os.getenv(key_name)
        if not api_key:
            return False, f"Missing API key for {service}"
        
        try:
            if service == 'alpha_vantage':
                response = requests.get(
                    f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={api_key}'
                )
            elif service == 'news_api':
                response = requests.get(
                    f'https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey={api_key}'
                )
            elif service == 'finnhub':
                response = requests.get(
                    f'https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}'
                )
            elif service == 'polygon':
                response = requests.get(
                    f'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey={api_key}'
                )
            
            if response.status_code == 200:
                return True, "API key is valid"
            else:
                return False, f"API key validation failed: {response.status_code}"
                
        except Exception as e:
            return False, f"API key validation failed: {str(e)}"
    
    def check_key_rotation(self) -> Dict[str, Dict]:
        """Check which API keys need rotation."""
        rotation_status = {}
        current_time = datetime.now()
        
        for service in self.required_keys:
            last_rotation = self.key_history.get(service, {}).get('last_rotation')
            if last_rotation:
                last_rotation = datetime.fromisoformat(last_rotation)
                days_since_rotation = (current_time - last_rotation).days
                needs_rotation = days_since_rotation >= 90  # Rotate every 90 days
            else:
                needs_rotation = True
            
            is_valid, message = self.validate_key(service)
            
            rotation_status[service] = {
                'last_rotation': last_rotation.isoformat() if last_rotation else None,
                'needs_rotation': needs_rotation,
                'is_valid': is_valid,
                'message': message
            }
        
        return rotation_status
    
    def record_key_rotation(self, service: str, new_key: str):
        """Record a key rotation event."""
        if service not in self.required_keys and service not in self.optional_keys:
            raise ValueError(f"Unknown service: {service}")
        
        self.key_history[service] = {
            'last_rotation': datetime.now().isoformat(),
            'rotations': self.key_history.get(service, {}).get('rotations', 0) + 1
        }
        self.save_key_history()
    
    def get_key_info(self, service: str) -> Optional[Dict]:
        """Get information about a specific API key."""
        if service not in self.required_keys and service not in self.optional_keys:
            return None
        
        key_name = self.required_keys.get(service) or self.optional_keys.get(service)
        if isinstance(key_name, list):
            key_name = key_name[0]  # Use primary key for services with multiple keys
        
        api_key = os.getenv(key_name)
        is_valid, message = self.validate_key(service) if api_key else (False, "Key not found")
        
        history = self.key_history.get(service, {})
        last_rotation = history.get('last_rotation')
        if last_rotation:
            last_rotation = datetime.fromisoformat(last_rotation)
            days_since_rotation = (datetime.now() - last_rotation).days
        else:
            days_since_rotation = None
        
        return {
            'service': service,
            'key_name': key_name,
            'is_valid': is_valid,
            'validation_message': message,
            'last_rotation': last_rotation.isoformat() if last_rotation else None,
            'days_since_rotation': days_since_rotation,
            'total_rotations': history.get('rotations', 0)
        }

def main():
    """Main function to check API key status."""
    manager = APIKeyManager()
    
    # Check all keys
    logger.info("Checking API key status...")
    rotation_status = manager.check_key_rotation()
    
    for service, status in rotation_status.items():
        logger.info(f"\n{service.upper()} API Key:")
        logger.info(f"Last Rotation: {status['last_rotation'] or 'Never'}")
        logger.info(f"Needs Rotation: {status['needs_rotation']}")
        logger.info(f"Is Valid: {status['is_valid']}")
        logger.info(f"Message: {status['message']}")

if __name__ == '__main__':
    main()
