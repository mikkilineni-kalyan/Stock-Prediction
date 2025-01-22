import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import asyncio
from collections import deque

from .model_version_manager import ModelVersionManager
from .advanced_ensemble import AdvancedEnsembleModel
from ..utils.logger import ModelLogger
from ..data_processing.data_preprocessor import DataPreprocessor
from ..data_processing.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class OnlineLearner:
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.version_manager = ModelVersionManager()
        self.data_preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_logger = ModelLogger("online_learning")
        self.training_buffer = {}  # Buffer for each symbol
        self.last_update = {}  # Track last update time for each symbol
        
    def _get_default_config(self) -> Dict:
        return {
            'buffer_size': 1000,  # Number of samples to keep in buffer
            'update_threshold': 100,  # Minimum samples before update
            'update_frequency': 24,  # Hours between updates
            'performance_window': 7,  # Days to monitor performance
            'learning_rate': 0.001,  # Learning rate for online updates
            'batch_size': 32,
            'validation_split': 0.2
        }
    
    async def process_new_data(self, symbol: str, new_data: pd.DataFrame):
        """Process and potentially learn from new data"""
        try:
            # Initialize buffer for symbol if not exists
            if symbol not in self.training_buffer:
                self.training_buffer[symbol] = deque(maxlen=self.config['buffer_size'])
            
            # Preprocess new data
            processed_data = await self.data_preprocessor.preprocess_stock_data(
                symbol, 
                lookback_days=1
            )
            
            if processed_data is None or processed_data.empty:
                logger.warning(f"No valid data to process for {symbol}")
                return
            
            # Add to buffer
            self.training_buffer[symbol].extend(processed_data.to_dict('records'))
            
            # Check if update is needed
            if self._should_update(symbol):
                await self._update_model(symbol)
            
        except Exception as e:
            logger.error(f"Error processing new data for {symbol}: {str(e)}")
            self.model_logger.log_error("online_learning_error", str(e))
    
    def _should_update(self, symbol: str) -> bool:
        """Determine if model should be updated"""
        try:
            # Check buffer size
            if len(self.training_buffer[symbol]) < self.config['update_threshold']:
                return False
            
            # Check last update time
            last_update = self.last_update.get(symbol)
            if last_update:
                hours_since_update = (datetime.utcnow() - last_update).total_seconds() / 3600
                if hours_since_update < self.config['update_frequency']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking update condition: {str(e)}")
            return False
    
    async def _update_model(self, symbol: str):
        """Update model with new data"""
        try:
            # Get current active model
            version_id, model = self.version_manager.get_active_model()
            
            # Prepare data for training
            buffer_data = pd.DataFrame(list(self.training_buffer[symbol]))
            
            # Engineer features
            features_df = self.feature_engineer.engineer_features(buffer_data)
            
            # Split into training and validation
            split_idx = int(len(features_df) * (1 - self.config['validation_split']))
            train_data = features_df.iloc[:split_idx]
            val_data = features_df.iloc[split_idx:]
            
            # Prepare sequences
            X_train, y_train = model.prepare_training_data(
                train_data.drop('target', axis=1).values,
                train_data['target'].values,
                self.config['batch_size']
            )
            
            X_val, y_val = model.prepare_training_data(
                val_data.drop('target', axis=1).values,
                val_data['target'].values,
                self.config['batch_size']
            )
            
            # Update model
            history = await self._incremental_update(
                model, X_train, y_train, X_val, y_val
            )
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(history, val_data)
            
            # Update version manager
            self.version_manager.update_performance(version_id, metrics, symbol)
            
            # Log update
            self.model_logger.log_training_end({
                'symbol': symbol,
                'metrics': metrics,
                'samples_processed': len(buffer_data)
            })
            
            # Update last update time
            self.last_update[symbol] = datetime.utcnow()
            
            # Clear buffer
            self.training_buffer[symbol].clear()
            
        except Exception as e:
            logger.error(f"Error updating model for {symbol}: {str(e)}")
            self.model_logger.log_error("model_update_error", str(e))
    
    async def _incremental_update(self, model: AdvancedEnsembleModel,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Perform incremental model update"""
        try:
            # Update deep learning model
            dl_history = model.models['deep_learning'].fit(
                X_train, y_train,
                batch_size=self.config['batch_size'],
                epochs=1,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Update traditional ML models
            for name, m in model.models.items():
                if name != 'deep_learning':
                    m.partial_fit(X_train, y_train)
            
            return dl_history.history
            
        except Exception as e:
            logger.error(f"Error in incremental update: {str(e)}")
            raise
    
    def _calculate_metrics(self, history: Dict, val_data: pd.DataFrame) -> Dict:
        """Calculate performance metrics after update"""
        try:
            metrics = {}
            
            # Training metrics
            metrics['train_loss'] = history.get('loss', [0])[-1]
            metrics['val_loss'] = history.get('val_loss', [0])[-1]
            
            # Trading metrics
            predictions = self.version_manager.get_active_model()[1].predict(
                val_data.drop('target', axis=1).values
            )
            
            # Calculate returns
            actual_returns = np.diff(val_data['target'])
            predicted_returns = np.diff(predictions)
            
            # Directional accuracy
            metrics['directional_accuracy'] = np.mean(
                np.sign(actual_returns) == np.sign(predicted_returns)
            )
            
            # Sharpe ratio
            strategy_returns = predicted_returns * actual_returns
            metrics['sharpe_ratio'] = np.sqrt(252) * np.mean(strategy_returns) / \
                                    np.std(strategy_returns)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
