import os
from pathlib import Path
from typing import Dict, Any
import json
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the application"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._load_config()
    
    def _load_config(self):
        """Load configuration from multiple sources"""
        try:
            # Load .env file if it exists
            env_path = Path(__file__).parent.parent.parent / '.env'
            load_dotenv(env_path)
            
            # Load config.json
            config_path = Path(__file__).parent.parent / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
            else:
                self.config = {}
            
            # API Keys - prioritize environment variables
            self.config['ALPHA_VANTAGE_KEY'] = os.getenv('ALPHA_VANTAGE_KEY', self.config.get('ALPHA_VANTAGE_KEY', ''))
            self.config['NEWS_API_KEY'] = os.getenv('NEWS_API_KEY', self.config.get('NEWS_API_KEY', ''))
            self.config['FINNHUB_API_KEY'] = os.getenv('FINNHUB_API_KEY', self.config.get('FINNHUB_API_KEY', ''))
            self.config['POLYGON_API_KEY'] = os.getenv('POLYGON_API_KEY', self.config.get('POLYGON_API_KEY', ''))
            
            # Database Configuration
            self.config['DATABASE'] = {
                'dialect': os.getenv('DB_DIALECT', 'sqlite'),
                'database': os.getenv('DB_NAME', 'stock_prediction.db'),
                'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
                'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '5')),
                'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
                'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '1800'))
            }
            
            # Model Configuration
            self.config['MODEL'] = {
                'version_path': os.getenv('MODEL_VERSION_PATH', 'models/versions'),
                'registry_path': os.getenv('MODEL_REGISTRY_PATH', 'models/registry'),
                'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'stock_prediction'),
                'mlflow_tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
            }
            
            # API Configuration
            self.config['API'] = {
                'rate_limit': int(os.getenv('API_RATE_LIMIT', '100')),
                'jwt_secret': os.getenv('JWT_SECRET', 'your-secret-key'),
                'jwt_algorithm': os.getenv('JWT_ALGORITHM', 'HS256'),
                'token_expire_minutes': int(os.getenv('TOKEN_EXPIRE_MINUTES', '1440'))
            }
            
            # Validate required configurations
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _validate_config(self):
        """Validate required configuration values"""
        required_keys = [
            'ALPHA_VANTAGE_KEY',
            'NEWS_API_KEY'
        ]
        
        missing_keys = [key for key in required_keys if not self.config.get(key)]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service"""
        key_mapping = {
            'alpha_vantage': 'ALPHA_VANTAGE_KEY',
            'news_api': 'NEWS_API_KEY',
            'finnhub': 'FINNHUB_API_KEY',
            'polygon': 'POLYGON_API_KEY'
        }
        
        key = self.config.get(key_mapping.get(service.lower()))
        if not key:
            raise ValueError(f"API key not found for service: {service}")
        return key
    
    def get_database_config(self) -> Dict:
        """Get database configuration"""
        return self.config['DATABASE']
    
    def get_model_config(self) -> Dict:
        """Get model configuration"""
        return self.config['MODEL']
    
    def get_api_config(self) -> Dict:
        """Get API configuration"""
        return self.config['API']
