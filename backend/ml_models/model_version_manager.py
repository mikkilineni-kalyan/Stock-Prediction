import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import json
import mlflow
import mlflow.tensorflow
from pathlib import Path
import hashlib
import shutil
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import joblib

from .advanced_ensemble import AdvancedEnsembleModel
from ..utils.logger import ModelLogger

logger = logging.getLogger(__name__)
Base = declarative_base()

class ModelVersion(Base):
    """Model version tracking table"""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    version_id = Column(String, unique=True)
    model_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)
    parameters = Column(JSON)
    status = Column(String)  # active, archived, failed
    path = Column(String)
    hash = Column(String)

class ModelPerformance(Base):
    """Model performance tracking table"""
    __tablename__ = 'model_performance'

    id = Column(Integer, primary_key=True)
    version_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String)
    metric_value = Column(Float)
    symbol = Column(String)

class ModelVersionManager:
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.model_logger = ModelLogger("model_versioning")
        self.engine = create_engine(self.config['database_uri'])
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config['mlflow_tracking_uri'])
        mlflow.set_experiment(self.config['experiment_name'])
        
    def _get_default_config(self) -> Dict:
        return {
            'database_uri': 'sqlite:///models.db',
            'mlflow_tracking_uri': 'sqlite:///mlflow.db',
            'experiment_name': 'stock_prediction',
            'model_registry_path': 'models/registry',
            'performance_threshold': {
                'min_accuracy': 0.6,
                'max_mse': 0.1,
                'min_sharpe': 0.5
            },
            'online_learning': {
                'batch_size': 100,
                'update_frequency': 24,  # hours
                'min_samples': 1000
            }
        }
    
    def create_version(self, model: AdvancedEnsembleModel, metrics: Dict, 
                      parameters: Dict) -> str:
        """Create a new model version"""
        try:
            # Generate version ID
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            version_id = f"model_{timestamp}"
            
            # Save model
            save_path = Path(self.config['model_registry_path']) / version_id
            model.save_model(str(save_path))
            
            # Calculate model hash
            model_hash = self._calculate_model_hash(save_path)
            
            # Create version record
            version = ModelVersion(
                version_id=version_id,
                model_type='ensemble',
                metrics=metrics,
                parameters=parameters,
                status='active',
                path=str(save_path),
                hash=model_hash
            )
            
            # Deactivate previous versions
            active_versions = self.session.query(ModelVersion).filter_by(status='active').all()
            for v in active_versions:
                v.status = 'archived'
            
            self.session.add(version)
            self.session.commit()
            
            # Log to MLflow
            with mlflow.start_run(run_name=version_id):
                mlflow.log_params(parameters)
                mlflow.log_metrics(metrics)
                mlflow.log_artifacts(str(save_path))
            
            logger.info(f"Created new model version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error creating model version: {str(e)}")
            raise
    
    def get_active_model(self) -> Tuple[str, AdvancedEnsembleModel]:
        """Get the currently active model"""
        try:
            version = self.session.query(ModelVersion).filter_by(status='active').first()
            if not version:
                raise ValueError("No active model version found")
            
            model = AdvancedEnsembleModel()
            model.load_model(version.path)
            
            return version.version_id, model
            
        except Exception as e:
            logger.error(f"Error getting active model: {str(e)}")
            raise
    
    def update_performance(self, version_id: str, metrics: Dict, symbol: str):
        """Update model performance metrics"""
        try:
            timestamp = datetime.utcnow()
            
            for metric_name, metric_value in metrics.items():
                performance = ModelPerformance(
                    version_id=version_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    symbol=symbol
                )
                self.session.add(performance)
            
            self.session.commit()
            
            # Check if performance is below threshold
            if self._check_performance_degradation(version_id):
                logger.warning(f"Performance degradation detected for model {version_id}")
                self.model_logger.log_error(
                    "performance_degradation",
                    f"Model {version_id} performance below threshold"
                )
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
            raise
    
    def _check_performance_degradation(self, version_id: str) -> bool:
        """Check if model performance has degraded"""
        try:
            # Get recent performance metrics
            recent_metrics = self.session.query(ModelPerformance)\
                .filter_by(version_id=version_id)\
                .filter(ModelPerformance.timestamp >= datetime.utcnow() - timedelta(days=7))\
                .all()
            
            if not recent_metrics:
                return False
            
            # Check against thresholds
            accuracy_metrics = [m.metric_value for m in recent_metrics 
                              if m.metric_name == 'accuracy']
            mse_metrics = [m.metric_value for m in recent_metrics 
                         if m.metric_name == 'mse']
            sharpe_metrics = [m.metric_value for m in recent_metrics 
                            if m.metric_name == 'sharpe_ratio']
            
            if accuracy_metrics and np.mean(accuracy_metrics) < self.config['performance_threshold']['min_accuracy']:
                return True
            
            if mse_metrics and np.mean(mse_metrics) > self.config['performance_threshold']['max_mse']:
                return True
            
            if sharpe_metrics and np.mean(sharpe_metrics) < self.config['performance_threshold']['min_sharpe']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {str(e)}")
            return False
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model files for version control"""
        try:
            hasher = hashlib.sha256()
            
            for file_path in sorted(model_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating model hash: {str(e)}")
            raise
    
    def rollback_version(self, target_version_id: str):
        """Rollback to a previous model version"""
        try:
            target_version = self.session.query(ModelVersion)\
                .filter_by(version_id=target_version_id)\
                .first()
            
            if not target_version:
                raise ValueError(f"Version {target_version_id} not found")
            
            # Deactivate current active version
            current_active = self.session.query(ModelVersion)\
                .filter_by(status='active')\
                .first()
            
            if current_active:
                current_active.status = 'archived'
            
            # Activate target version
            target_version.status = 'active'
            self.session.commit()
            
            logger.info(f"Rolled back to version: {target_version_id}")
            
        except Exception as e:
            logger.error(f"Error rolling back version: {str(e)}")
            raise
    
    def cleanup_old_versions(self, keep_days: int = 30):
        """Clean up old model versions"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
            
            old_versions = self.session.query(ModelVersion)\
                .filter(ModelVersion.created_at < cutoff_date)\
                .filter(ModelVersion.status != 'active')\
                .all()
            
            for version in old_versions:
                # Remove files
                path = Path(version.path)
                if path.exists():
                    shutil.rmtree(path)
                
                # Remove from database
                self.session.delete(version)
            
            self.session.commit()
            logger.info(f"Cleaned up {len(old_versions)} old model versions")
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {str(e)}")
            raise
