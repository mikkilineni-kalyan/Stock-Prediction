import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import mlflow
import mlflow.tensorflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..data_processing.data_validator import DataValidator
from ..data_processing.feature_engineering import FeatureEngineer
from ..ml_models.advanced_ensemble import AdvancedEnsembleModel
from ..utils.logger import ModelLogger, DataLogger

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.data_validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        self.model = AdvancedEnsembleModel()
        self.model_logger = ModelLogger("stock_prediction")
        self.data_logger = DataLogger()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config['mlflow_tracking_uri'])
        mlflow.set_experiment(self.config['experiment_name'])
    
    def _get_default_config(self) -> Dict:
        return {
            'data_validation': {
                'min_training_points': 1000,
                'max_missing_percentage': 0.1,
                'min_price': 0.01
            },
            'feature_engineering': {
                'sequence_length': 20,
                'include_sentiment': True,
                'technical_indicators': True,
                'market_indicators': True
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'optimize_hyperparameters': True,
                'early_stopping_patience': 10,
                'max_epochs': 100
            },
            'evaluation': {
                'metrics': ['mse', 'mae', 'r2', 'sharpe_ratio'],
                'backtest_periods': 30
            },
            'mlflow_tracking_uri': 'sqlite:///mlflow.db',
            'experiment_name': 'stock_prediction',
            'model_save_path': 'models/stock_prediction',
            'prediction_threshold': 0.6
        }
    
    async def run_pipeline(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Run the complete training pipeline"""
        try:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_logger.log_training_start({
                'run_id': run_id,
                'symbol': symbol,
                'config': self.config
            })
            
            with mlflow.start_run(run_name=f"{symbol}_{run_id}") as run:
                # Log configuration
                mlflow.log_params(self.config)
                
                # 1. Data Validation
                logger.info("Starting data validation...")
                validated_data, validation_messages = await self.data_validator.validate_stock_data(
                    data, symbol
                )
                self.data_logger.log_data_validation(symbol, {
                    'messages': validation_messages,
                    'data_shape': validated_data.shape
                })
                
                # 2. Feature Engineering
                logger.info("Starting feature engineering...")
                features_df = self.feature_engineer.engineer_features(
                    validated_data,
                    include_sentiment=self.config['feature_engineering']['include_sentiment']
                )
                self.data_logger.log_data_processing(symbol, {
                    'feature_count': features_df.shape[1],
                    'feature_names': list(features_df.columns)
                })
                
                # 3. Train/Test Split with Time Series Consideration
                train_data, val_data, test_data = self._time_series_split(features_df)
                
                # 4. Model Training
                logger.info("Starting model training...")
                X_train = train_data.drop('target', axis=1).values
                y_train = train_data['target'].values
                
                training_result = self._train_model(X_train, y_train, val_data)
                
                # 5. Model Evaluation
                logger.info("Evaluating model...")
                evaluation_results = self._evaluate_model(test_data)
                
                # Log metrics to MLflow
                for metric_name, metric_value in evaluation_results.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Save model artifacts
                self._save_model_artifacts(run.info.run_id)
                
                # Log completion
                self.model_logger.log_training_end({
                    'run_id': run_id,
                    'metrics': evaluation_results
                })
                
                return {
                    'run_id': run_id,
                    'validation_messages': validation_messages,
                    'evaluation_results': evaluation_results,
                    'model_artifacts_path': self.config['model_save_path']
                }
                
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            self.model_logger.log_error("pipeline_error", str(e))
            raise
    
    def _time_series_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data with time series consideration"""
        total_rows = len(df)
        train_size = int(total_rows * (1 - self.config['training']['test_size'] - 
                                     self.config['training']['validation_size']))
        val_size = int(total_rows * self.config['training']['validation_size'])
        
        train_data = df.iloc[:train_size]
        val_data = df.iloc[train_size:train_size + val_size]
        test_data = df.iloc[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                    val_data: pd.DataFrame) -> Dict:
        """Train the model with validation"""
        try:
            # Prepare validation data
            X_val = val_data.drop('target', axis=1).values
            y_val = val_data['target'].values
            
            # Train model
            self.model.train(
                X_train, y_train,
                optimize=self.config['training']['optimize_hyperparameters']
            )
            
            # Get validation predictions
            val_predictions = self.model.predict(X_val)
            
            # Calculate validation metrics
            val_metrics = {
                'val_mse': mean_squared_error(y_val, val_predictions),
                'val_mae': mean_absolute_error(y_val, val_predictions),
                'val_r2': r2_score(y_val, val_predictions)
            }
            
            return {
                'validation_metrics': val_metrics,
                'feature_importance': self.model.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def _evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        try:
            X_test = test_data.drop('target', axis=1).values
            y_test = test_data['target'].values
            
            # Generate predictions
            predictions = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'test_mse': mean_squared_error(y_test, predictions),
                'test_mae': mean_absolute_error(y_test, predictions),
                'test_r2': r2_score(y_test, predictions)
            }
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(y_test, predictions)
            metrics.update(trading_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def _calculate_trading_metrics(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict:
        """Calculate trading-specific metrics"""
        try:
            # Calculate returns
            actual_returns = np.diff(y_true)
            predicted_returns = np.diff(y_pred)
            
            # Calculate directional accuracy
            directional_accuracy = np.mean(np.sign(actual_returns) == np.sign(predicted_returns))
            
            # Calculate Sharpe ratio (assuming daily data)
            strategy_returns = predicted_returns * actual_returns
            sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumsum(strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns)
            
            return {
                'directional_accuracy': directional_accuracy,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {str(e)}")
            raise
    
    def _save_model_artifacts(self, run_id: str):
        """Save model artifacts and metadata"""
        try:
            # Create save directory
            save_path = Path(self.config['model_save_path']) / run_id
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save_model(str(save_path / 'model'))
            
            # Save feature importance
            feature_importance_path = save_path / 'feature_importance.json'
            with open(feature_importance_path, 'w') as f:
                json.dump(self.model.feature_importance, f, indent=2)
            
            # Save configuration
            config_path = save_path / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Log artifacts to MLflow
            mlflow.log_artifacts(str(save_path))
            
            logger.info(f"Model artifacts saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {str(e)}")
            raise
