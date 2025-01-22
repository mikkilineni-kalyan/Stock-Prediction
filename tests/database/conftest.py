import pytest
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta

from backend.database.models import Base
from backend.database.db_config import DatabaseConfig

@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """Create a temporary database file"""
    db_dir = tmp_path_factory.mktemp("test_db")
    return db_dir / "test.db"

@pytest.fixture(scope="session")
def test_db_config(test_db_path):
    """Configure test database"""
    os.environ["APP_ENV"] = "test"
    config = DatabaseConfig()
    config.db_config = {
        'dialect': 'sqlite',
        'database': str(test_db_path)
    }
    return config

@pytest.fixture(scope="function")
def db_session(test_db_config):
    """Create a new database session for a test"""
    engine = create_engine(f"sqlite:///{test_db_config.db_config['database']}")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)

@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing"""
    base_time = datetime.utcnow()
    return [
        {
            'symbol': 'AAPL',
            'timestamp': base_time - timedelta(days=i),
            'open_price': 150.0 + i,
            'high_price': 155.0 + i,
            'low_price': 145.0 + i,
            'close_price': 152.0 + i,
            'volume': 1000000 + i * 1000
        }
        for i in range(10)
    ]

@pytest.fixture
def sample_feature_data():
    """Generate sample feature data for testing"""
    return {
        'name': 'RSI',
        'description': 'Relative Strength Index',
        'category': 'technical',
        'parameters': {'window': 14}
    }

@pytest.fixture
def sample_model_data():
    """Generate sample model data for testing"""
    return {
        'name': 'test_model',
        'version': '1.0.0',
        'type': 'ensemble',
        'parameters': {
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87
        },
        'status': 'active'
    }

@pytest.fixture
def sample_prediction_data():
    """Generate sample prediction data for testing"""
    base_time = datetime.utcnow()
    return [
        {
            'symbol': 'AAPL',
            'timestamp': base_time + timedelta(days=i),
            'prediction': 155.0 + i,
            'confidence': 0.85,
            'features_used': ['RSI', 'MACD', 'BB']
        }
        for i in range(5)
    ]
