import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Concatenate, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import joblib
import logging
import shap
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class AdvancedEnsembleModel:
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.feature_importance = {}
        self.model_weights = None
        self.scaler = None
        self.best_params = {}
        
    def _get_default_config(self) -> Dict:
        return {
            'lstm': {
                'units': [128, 64, 32],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'cnn': {
                'filters': [64, 32, 16],
                'kernel_size': 3,
                'pool_size': 2,
                'dense_units': [64, 32],
                'dropout': 0.2
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 7,
                'learning_rate': 0.1
            },
            'lightgbm': {
                'n_estimators': 200,
                'num_leaves': 31,
                'learning_rate': 0.1
            },
            'sequence_length': 20,
            'train_test_split': 0.8,
            'optimization_trials': 50
        }
    
    def build_deep_learning_model(self, input_shape: Tuple) -> Model:
        """Build hybrid CNN-LSTM model with attention"""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # CNN branch
        conv = Conv1D(filters=self.config['cnn']['filters'][0],
                     kernel_size=self.config['cnn']['kernel_size'],
                     activation='relu')(inputs)
        conv = LayerNormalization()(conv)
        conv = MaxPooling1D(pool_size=self.config['cnn']['pool_size'])(conv)
        
        # LSTM branch with attention
        lstm = LSTM(self.config['lstm']['units'][0],
                   return_sequences=True)(inputs)
        lstm = LayerNormalization()(lstm)
        
        # Attention mechanism
        attention = Attention()([lstm, lstm])
        
        # Merge CNN and LSTM branches
        merged = Concatenate()([conv, attention])
        
        # Additional LSTM layers
        for units in self.config['lstm']['units'][1:]:
            merged = LSTM(units, return_sequences=True)(merged)
            merged = LayerNormalization()(merged)
            merged = Dropout(self.config['lstm']['dropout'])(merged)
        
        # Dense layers
        merged = LSTM(self.config['lstm']['units'][-1])(merged)
        for units in self.config['cnn']['dense_units']:
            merged = Dense(units, activation='relu')(merged)
            merged = Dropout(self.config['cnn']['dropout'])(merged)
        
        # Output layer
        outputs = Dense(1)(merged)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.config['lstm']['learning_rate']),
                     loss='mse')
        
        return model
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            # Deep learning params
            lstm_units = trial.suggest_int('lstm_units', 32, 256)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            
            # Traditional ML params
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            
            # Update config
            temp_config = self.config.copy()
            temp_config['lstm']['units'] = [lstm_units, lstm_units//2, lstm_units//4]
            temp_config['lstm']['dropout'] = dropout
            temp_config['lstm']['learning_rate'] = learning_rate
            temp_config['random_forest']['n_estimators'] = n_estimators
            temp_config['random_forest']['max_depth'] = max_depth
            
            # Create and train models
            self.config = temp_config
            self._create_models(X_train.shape[1:])
            return self._train_evaluate_fold(X_train, y_train)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config['optimization_trials'])
        
        return study.best_params
    
    def _create_models(self, input_shape: Tuple):
        """Create all models in the ensemble"""
        self.models = {
            'deep_learning': self.build_deep_learning_model(input_shape),
            'random_forest': RandomForestRegressor(**self.config['random_forest']),
            'xgboost': xgb.XGBRegressor(**self.config['xgboost']),
            'lightgbm': lgb.LGBMRegressor(**self.config['lightgbm'])
        }
    
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Prepare sequences for deep learning models"""
        sequences = []
        for i in range(len(X) - self.config['sequence_length']):
            sequences.append(X[i:i + self.config['sequence_length']])
        return np.array(sequences)
    
    def train(self, X: np.ndarray, y: np.ndarray, optimize: bool = True):
        """Train the ensemble model"""
        try:
            # Optimize hyperparameters if requested
            if optimize:
                logger.info("Optimizing hyperparameters...")
                self.best_params = self.optimize_hyperparameters(X, y)
                self.config.update(self.best_params)
            
            # Create models with current config
            self._create_models(X.shape[1:])
            
            # Prepare sequences for deep learning
            X_seq = self._prepare_sequences(X)
            y_seq = y[self.config['sequence_length']:]
            
            # Train deep learning model
            self.models['deep_learning'].fit(
                X_seq, y_seq,
                batch_size=self.config['lstm']['batch_size'],
                epochs=self.config['lstm']['epochs'],
                validation_split=0.2,
                verbose=1
            )
            
            # Train traditional ML models
            for name, model in self.models.items():
                if name != 'deep_learning':
                    model.fit(X, y)
            
            # Calculate feature importance
            self._calculate_feature_importance(X, y)
            
            # Calculate model weights based on validation performance
            self._calculate_model_weights(X, y)
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions"""
        try:
            predictions = {}
            
            # Deep learning prediction
            X_seq = self._prepare_sequences(X)
            predictions['deep_learning'] = self.models['deep_learning'].predict(X_seq)
            
            # Traditional ML predictions
            for name, model in self.models.items():
                if name != 'deep_learning':
                    predictions[name] = model.predict(X)
            
            # Weighted ensemble prediction
            final_prediction = np.zeros_like(predictions['deep_learning'])
            for name, pred in predictions.items():
                if name == 'deep_learning':
                    final_prediction += self.model_weights[name] * pred
                else:
                    final_prediction += self.model_weights[name] * pred[-len(final_prediction):]
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance for each model"""
        try:
            for name, model in self.models.items():
                if name == 'deep_learning':
                    # Use SHAP for deep learning model
                    explainer = shap.DeepExplainer(self.models['deep_learning'], 
                                                 self._prepare_sequences(X[:100]))
                    shap_values = explainer.shap_values(self._prepare_sequences(X[:100]))
                    self.feature_importance['deep_learning'] = np.abs(shap_values).mean(0)
                else:
                    # Use built-in feature importance for tree-based models
                    self.feature_importance[name] = model.feature_importances_
                    
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
    
    def _calculate_model_weights(self, X: np.ndarray, y: np.ndarray):
        """Calculate model weights based on validation performance"""
        try:
            scores = {}
            tscv = TimeSeriesSplit(n_splits=5)
            
            for name, model in self.models.items():
                fold_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    if name == 'deep_learning':
                        X_train_seq = self._prepare_sequences(X_train)
                        X_val_seq = self._prepare_sequences(X_val)
                        y_train_seq = y_train[self.config['sequence_length']:]
                        y_val_seq = y_val[self.config['sequence_length']:]
                        
                        model.fit(X_train_seq, y_train_seq, 
                                batch_size=self.config['lstm']['batch_size'],
                                epochs=self.config['lstm']['epochs'],
                                verbose=0)
                        pred = model.predict(X_val_seq)
                        score = r2_score(y_val_seq, pred)
                    else:
                        model.fit(X_train, y_train)
                        pred = model.predict(X_val)
                        score = r2_score(y_val, pred)
                    
                    fold_scores.append(score)
                
                scores[name] = np.mean(fold_scores)
            
            # Calculate weights using softmax
            weights = np.array(list(scores.values()))
            weights = np.exp(weights) / np.sum(np.exp(weights))
            self.model_weights = dict(zip(scores.keys(), weights))
            
        except Exception as e:
            logger.error(f"Error calculating model weights: {str(e)}")
    
    def save_model(self, path: str):
        """Save the ensemble model"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save deep learning model
            self.models['deep_learning'].save(save_path / 'deep_learning_model')
            
            # Save traditional ML models
            for name, model in self.models.items():
                if name != 'deep_learning':
                    joblib.dump(model, save_path / f'{name}_model.joblib')
            
            # Save configuration and weights
            config_data = {
                'config': self.config,
                'model_weights': self.model_weights,
                'feature_importance': self.feature_importance,
                'best_params': self.best_params
            }
            joblib.dump(config_data, save_path / 'model_config.joblib')
            
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load the ensemble model"""
        try:
            load_path = Path(path)
            
            # Load deep learning model
            self.models['deep_learning'] = tf.keras.models.load_model(load_path / 'deep_learning_model')
            
            # Load traditional ML models
            for name in ['random_forest', 'xgboost', 'lightgbm']:
                self.models[name] = joblib.load(load_path / f'{name}_model.joblib')
            
            # Load configuration and weights
            config_data = joblib.load(load_path / 'model_config.joblib')
            self.config = config_data['config']
            self.model_weights = config_data['model_weights']
            self.feature_importance = config_data['feature_importance']
            self.best_params = config_data['best_params']
            
            logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
