import os
from dotenv import load_dotenv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Get the backend directory (where your .env file is)
BACKEND_DIR = Path(__file__).parent.parent  # This will go up to Stock-Prediction/backend/

# Load environment variables from backend/.env file
env_path = BACKEND_DIR / '.env'
load_dotenv(env_path)

class Config:
    # Flask Configuration
    FLASK_APP = os.getenv('FLASK_APP', 'api/app.py')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # API URLs
    VITE_API_URL = os.getenv('VITE_API_URL', 'http://localhost:5000')
    
    # Required API keys
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    
    # Social Media APIs
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    
    # Additional APIs
    SEEKING_ALPHA_API_KEY = os.getenv('SEEKING_ALPHA_API_KEY')
    SEC_API_KEY = os.getenv('SEC_API_KEY')
    
    # Database and Cache Configuration
    DB_PATH = os.getenv('DB_PATH', 'stock_predictions.db')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    @staticmethod
    def check_api_keys():
        """Check which API keys are available and log status"""
        api_status = {
            'NewsAPI': bool(Config.NEWS_API_KEY),
            'AlphaVantage': bool(Config.ALPHA_VANTAGE_API_KEY),
            'Finnhub': bool(Config.FINNHUB_API_KEY),
            'Twitter': bool(Config.TWITTER_API_KEY and Config.TWITTER_API_SECRET),
            'Reddit': bool(Config.REDDIT_CLIENT_ID and Config.REDDIT_CLIENT_SECRET),
            'SeekingAlpha': bool(Config.SEEKING_ALPHA_API_KEY),
            'SEC': bool(Config.SEC_API_KEY)
        }
        
        # Log status of each API
        for api, status in api_status.items():
            if status:
                logger.info(f"{api} API key is configured")
            else:
                logger.warning(f"{api} API key is missing")
        
        return api_status

    @staticmethod
    def validate_required_keys():
        """Validate that minimum required keys are present"""
        required_keys = {
            'ALPHA_VANTAGE_API_KEY': Config.ALPHA_VANTAGE_API_KEY
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")