import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib
from pathlib import Path
import json
import optuna
from sklearn.model_selection import TimeSeriesSplit
import shap

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.lstm_model = None
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.cnn_model = None
        
        # Model weights (will be optimized)
        self.model_weights = {
            'lstm': 0.25,
            'rf': 0.20,
            'xgb': 0.25,
            'lgb': 0.15,
            'cnn': 0.15
        }
        
        # Performance tracking
        self.performance_history = []
        self.model_versions = {}
        
        self._load_or_create_models()

    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create LSTM model with attention"""
        model = Sequential([
            # LSTM layers
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            
            # Attention layer
            Attention(),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae', 'mape'])
        
        return model

    def _create_cnn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create CNN model for time series"""
        model = Sequential([
            # CNN layers
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # Flatten and Dense layers
            tf.keras.layers.Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae', 'mape'])
        
        return model

    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            # LSTM
            lstm_path = self.models_dir / "lstm_model.h5"
            if lstm_path.exists():
                self.lstm_model = load_model(lstm_path)
                logger.info("Loaded existing LSTM model")
            
            # Random Forest
            rf_path = self.models_dir / "rf_model.joblib"
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                logger.info("Loaded existing Random Forest model")
            else:
                self.rf_model = RandomForestRegressor(n_estimators=200,
                                                    max_depth=20,
                                                    min_samples_split=10,
                                                    random_state=42,
                                                    n_jobs=-1)
            
            # XGBoost
            xgb_path = self.models_dir / "xgb_model.json"
            if xgb_path.exists():
                self.xgb_model = xgb.XGBRegressor()
                self.xgb_model.load_model(xgb_path)
                logger.info("Loaded existing XGBoost model")
            else:
                self.xgb_model = xgb.XGBRegressor(n_estimators=200,
                                                max_depth=10,
                                                learning_rate=0.1,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                random_state=42)
            
            # LightGBM
            lgb_path = self.models_dir / "lgb_model.txt"
            if lgb_path.exists():
                self.lgb_model = lgb.Booster(model_file=str(lgb_path))
                logger.info("Loaded existing LightGBM model")
            else:
                self.lgb_model = lgb.LGBMRegressor(n_estimators=200,
                                                 max_depth=10,
                                                 learning_rate=0.1,
                                                 subsample=0.8,
                                                 colsample_bytree=0.8,
                                                 random_state=42)
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _save_models(self):
        """Save all models"""
        try:
            # LSTM
            if self.lstm_model:
                self.lstm_model.save(self.models_dir / "lstm_model.h5")
            
            # Random Forest
            if self.rf_model:
                joblib.dump(self.rf_model, self.models_dir / "rf_model.joblib")
            
            # XGBoost
            if self.xgb_model:
                self.xgb_model.save_model(self.models_dir / "xgb_model.json")
            
            # LightGBM
            if self.lgb_model:
                self.lgb_model.save_model(str(self.models_dir / "lgb_model.txt"))
            
            # Save model versions
            with open(self.models_dir / "model_versions.json", 'w') as f:
                json.dump(self.model_versions, f)
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Optimize model hyperparameters using Optuna"""
        def objective(trial):
            # LSTM hyperparameters
            lstm_units = trial.suggest_int('lstm_units', 32, 256)
            lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            
            # Tree model hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            
            # Create and train models with suggested hyperparameters
            # ... (implementation details)
            
            # Return validation score
            return val_score
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params

    def prepare_sequence_data(self, X: np.ndarray, y: np.ndarray, 
                            sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              sequence_length: int = 10) -> Dict[str, float]:
        """Train all models in the ensemble"""
        try:
            # Prepare sequential data for LSTM and CNN
            X_seq_train, y_seq_train = self.prepare_sequence_data(X_train, y_train, sequence_length)
            X_seq_val, y_seq_val = self.prepare_sequence_data(X_val, y_val, sequence_length)
            
            # Train LSTM
            if self.lstm_model is None:
                self.lstm_model = self._create_lstm_model((sequence_length, X_train.shape[1]))
            
            lstm_history = self.lstm_model.fit(
                X_seq_train, y_seq_train,
                validation_data=(X_seq_val, y_seq_val),
                epochs=100,
                batch_size=32,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ModelCheckpoint(self.models_dir / "lstm_best.h5",
                                  save_best_only=True)
                ],
                verbose=0
            )
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            
            # Train XGBoost
            self.xgb_model.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             early_stopping_rounds=10,
                             verbose=False)
            
            # Train LightGBM
            self.lgb_model.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             early_stopping_rounds=10,
                             verbose=False)
            
            # Train CNN
            if self.cnn_model is None:
                self.cnn_model = self._create_cnn_model((sequence_length, X_train.shape[1]))
            
            cnn_history = self.cnn_model.fit(
                X_seq_train, y_seq_train,
                validation_data=(X_seq_val, y_seq_val),
                epochs=100,
                batch_size=32,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ModelCheckpoint(self.models_dir / "cnn_best.h5",
                                  save_best_only=True)
                ],
                verbose=0
            )
            
            # Update model versions
            self.model_versions = {
                'lstm': {'version': datetime.now().isoformat(),
                        'val_loss': min(lstm_history.history['val_loss'])},
                'rf': {'version': datetime.now().isoformat()},
                'xgb': {'version': datetime.now().isoformat(),
                       'best_iteration': self.xgb_model.best_iteration},
                'lgb': {'version': datetime.now().isoformat(),
                       'best_iteration': self.lgb_model.best_iteration},
                'cnn': {'version': datetime.now().isoformat(),
                       'val_loss': min(cnn_history.history['val_loss'])}
            }
            
            # Save models
            self._save_models()
            
            # Calculate and return training metrics
            return self.evaluate(X_val, y_val, sequence_length)
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def predict(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """Generate ensemble predictions"""
        try:
            # Prepare sequential data
            X_seq = np.array([X[i:i + sequence_length] for i in range(len(X) - sequence_length)])
            
            # Get predictions from each model
            lstm_pred = self.lstm_model.predict(X_seq, verbose=0)
            rf_pred = self.rf_model.predict(X)
            xgb_pred = self.xgb_model.predict(X)
            lgb_pred = self.lgb_model.predict(X)
            cnn_pred = self.cnn_model.predict(X_seq, verbose=0)
            
            # Adjust predictions to match lengths
            rf_pred = rf_pred[sequence_length:]
            xgb_pred = xgb_pred[sequence_length:]
            lgb_pred = lgb_pred[sequence_length:]
            
            # Weighted ensemble
            ensemble_pred = (
                self.model_weights['lstm'] * lstm_pred +
                self.model_weights['rf'] * rf_pred.reshape(-1, 1) +
                self.model_weights['xgb'] * xgb_pred.reshape(-1, 1) +
                self.model_weights['lgb'] * lgb_pred.reshape(-1, 1) +
                self.model_weights['cnn'] * cnn_pred
            )
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                sequence_length: int = 10) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Generate predictions
            y_pred = self.predict(X_test, sequence_length)
            y_test = y_test[sequence_length:]  # Adjust for sequence length
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate directional accuracy
            direction_pred = np.sign(y_pred[1:] - y_pred[:-1])
            direction_true = np.sign(y_test[1:] - y_test[:-1])
            directional_accuracy = np.mean(direction_pred == direction_true)
            
            metrics = {
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy
            }
            
            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            raise

    def explain_predictions(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """Generate model explanations using SHAP"""
        try:
            explanations = {}
            
            # SHAP values for tree models
            for model_name, model in [('rf', self.rf_model),
                                    ('xgb', self.xgb_model),
                                    ('lgb', self.lgb_model)]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Calculate feature importance
                feature_importance = np.abs(shap_values).mean(0)
                explanations[model_name] = dict(zip(feature_names,
                                                  feature_importance))
            
            # For LSTM and CNN, use integrated gradients
            # (Implementation depends on specific requirements)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            raise

    def update_model_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                           sequence_length: int = 10):
        """Update ensemble weights based on recent performance"""
        try:
            # Prepare sequential data
            X_seq = np.array([X_val[i:i + sequence_length] for i in range(len(X_val) - sequence_length)])
            y_val = y_val[sequence_length:]
            
            # Get predictions from each model
            predictions = {
                'lstm': self.lstm_model.predict(X_seq, verbose=0),
                'rf': self.rf_model.predict(X_val[sequence_length:]).reshape(-1, 1),
                'xgb': self.xgb_model.predict(X_val[sequence_length:]).reshape(-1, 1),
                'lgb': self.lgb_model.predict(X_val[sequence_length:]).reshape(-1, 1),
                'cnn': self.cnn_model.predict(X_seq, verbose=0)
            }
            
            # Calculate individual model performance
            performance = {}
            for model_name, pred in predictions.items():
                mse = mean_squared_error(y_val, pred)
                performance[model_name] = 1 / mse  # Use inverse MSE as weight
            
            # Normalize weights
            total_performance = sum(performance.values())
            self.model_weights = {
                model: score / total_performance
                for model, score in performance.items()
            }
            
            logger.info(f"Updated model weights: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")
            raise
