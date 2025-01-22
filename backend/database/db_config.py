from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import logging
from typing import Dict, Optional
import os
from pathlib import Path
import json

logger = logging.getLogger(__name__)
Base = declarative_base()

class DatabaseConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._load_config()
        self._setup_engine()
        self._create_session_factory()
    
    def _load_config(self):
        """Load database configuration from config file"""
        try:
            config_path = Path(__file__).parent / 'config' / 'database.json'
            
            if not config_path.exists():
                # Create default config if it doesn't exist
                config_path.parent.mkdir(parents=True, exist_ok=True)
                default_config = {
                    'default': {
                        'dialect': 'sqlite',
                        'database': 'stock_prediction.db',
                        'pool_size': 10,
                        'max_overflow': 5,
                        'pool_timeout': 30,
                        'pool_recycle': 1800
                    },
                    'test': {
                        'dialect': 'sqlite',
                        'database': ':memory:'
                    }
                }
                
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
            
            with open(config_path) as f:
                self.config = json.load(f)
            
            # Get environment-specific config
            env = os.getenv('APP_ENV', 'default')
            self.db_config = self.config.get(env, self.config['default'])
            
        except Exception as e:
            logger.error(f"Error loading database config: {str(e)}")
            raise
    
    def _setup_engine(self):
        """Setup SQLAlchemy engine with connection pooling"""
        try:
            # Get database path
            if self.db_config['database'] == ':memory:':
                db_url = 'sqlite:///:memory:'
            else:
                db_path = Path(__file__).parent.parent.parent / 'data' / self.db_config['database']
                db_path.parent.mkdir(parents=True, exist_ok=True)
                db_url = f"sqlite:///{db_path}"
            
            # Create engine with SQLite-appropriate pooling
            self.engine = create_engine(
                db_url,
                pool_size=self.db_config.get('pool_size', 10),
                max_overflow=self.db_config.get('max_overflow', 5),
                pool_timeout=self.db_config.get('pool_timeout', 30),
                pool_recycle=self.db_config.get('pool_recycle', 1800),
                echo=False
            )
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
        except Exception as e:
            logger.error(f"Error setting up database engine: {str(e)}")
            raise
    
    def _create_session_factory(self):
        """Create scoped session factory"""
        try:
            session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(session_factory)
        except Exception as e:
            logger.error(f"Error creating session factory: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_engine(self):
        """Get the SQLAlchemy engine"""
        return self.engine
    
    def get_connection(self):
        """Get a raw database connection from the pool"""
        return self.engine.connect()
    
    def dispose_engine(self):
        """Dispose of the engine and connection pool"""
        self.engine.dispose()
    
    def health_check(self) -> Dict:
        """Check database connection health"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            # Get pool statistics
            pool_status = {
                'size': self.engine.pool.size(),
                'checkedin': self.engine.pool.checkedin(),
                'overflow': self.engine.pool.overflow(),
                'checkedout': self.engine.pool.checkedout()
            }
            
            return {
                'status': 'healthy',
                'pool': pool_status
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def recreate_tables(self):
        """Recreate all database tables (WARNING: destroys all data)"""
        try:
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.error(f"Error recreating tables: {str(e)}")
            raise
