import pytest
import os
from pathlib import Path
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.database.db_config import DatabaseConfig
from backend.database.models import Base

class TestDatabaseConfig:
    def test_singleton_instance(self):
        """Test that DatabaseConfig is a singleton"""
        config1 = DatabaseConfig()
        config2 = DatabaseConfig()
        assert config1 is config2
    
    def test_default_config_creation(self, tmp_path):
        """Test creation of default config file"""
        # Set up test environment
        os.environ['APP_ENV'] = 'default'
        
        # Initialize config
        config = DatabaseConfig()
        
        # Verify default config values
        assert config.db_config['dialect'] == 'sqlite'
        assert 'database' in config.db_config
        assert config.db_config['pool_size'] == 10
    
    def test_test_environment_config(self):
        """Test configuration in test environment"""
        # Set up test environment
        os.environ['APP_ENV'] = 'test'
        
        # Initialize config
        config = DatabaseConfig()
        
        # Verify test config
        assert config.db_config['dialect'] == 'sqlite'
        assert config.db_config['database'] == ':memory:'
    
    def test_engine_creation(self):
        """Test SQLAlchemy engine creation"""
        config = DatabaseConfig()
        engine = config.get_engine()
        
        assert isinstance(engine, Engine)
        assert str(engine.url).startswith('sqlite://')
    
    def test_session_management(self):
        """Test session creation and cleanup"""
        config = DatabaseConfig()
        
        with config.get_session() as session:
            assert isinstance(session, Session)
            # Test that session is active
            assert session.is_active
        
        # Test that session is closed after context
        assert not session.is_active
    
    def test_connection_pooling(self):
        """Test connection pool configuration"""
        config = DatabaseConfig()
        engine = config.get_engine()
        
        # Verify pool settings
        assert engine.pool.size() == config.db_config['pool_size']
        assert engine.pool.overflow() == config.db_config['max_overflow']
    
    def test_health_check(self):
        """Test database health check functionality"""
        config = DatabaseConfig()
        health_status = config.health_check()
        
        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert 'pool' in health_status
        assert health_status['status'] == 'healthy'
    
    def test_error_handling(self):
        """Test error handling in database operations"""
        config = DatabaseConfig()
        
        # Test with invalid query
        with pytest.raises(Exception):
            with config.get_session() as session:
                session.execute("SELECT * FROM nonexistent_table")
    
    def test_table_creation(self, tmp_path):
        """Test automatic table creation"""
        # Set up test database
        db_path = tmp_path / "test_creation.db"
        os.environ['APP_ENV'] = 'default'
        
        config = DatabaseConfig()
        config.db_config['database'] = str(db_path)
        
        # Initialize database
        engine = config.get_engine()
        
        # Verify all tables are created
        for table_name in Base.metadata.tables.keys():
            assert engine.dialect.has_table(engine, table_name)
    
    def test_database_cleanup(self):
        """Test proper cleanup of database resources"""
        config = DatabaseConfig()
        engine = config.get_engine()
        
        # Create and dispose engine
        config.dispose_engine()
        
        # Verify pool is disposed
        assert engine.pool.status() == 'disposed'
