import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_path / f"{name}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ModelLogger:
    """Logger specifically for model training and prediction"""
    
    def __init__(self, model_name: str, log_dir: str = "logs/models"):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"model_{model_name}")
        
    def log_training_start(self, config: dict):
        """Log training start with configuration"""
        self.logger.info(f"Starting training for model: {self.model_name}")
        self._save_event("training_start", {
            "model_name": self.model_name,
            "config": config,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_training_end(self, metrics: dict):
        """Log training completion with metrics"""
        self.logger.info(f"Training completed for model: {self.model_name}")
        self._save_event("training_end", {
            "model_name": self.model_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_prediction(self, prediction_data: dict):
        """Log prediction details"""
        self.logger.info(f"Making prediction with model: {self.model_name}")
        self._save_event("prediction", {
            "model_name": self.model_name,
            "prediction": prediction_data,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_error(self, error_type: str, error_msg: str, stack_trace: str = None):
        """Log error details"""
        self.logger.error(f"Error in {self.model_name}: {error_msg}")
        self._save_event("error", {
            "model_name": self.model_name,
            "error_type": error_type,
            "error_message": error_msg,
            "stack_trace": stack_trace,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_model_metrics(self, metrics: dict):
        """Log model performance metrics"""
        self.logger.info(f"Model metrics for {self.model_name}: {metrics}")
        self._save_event("metrics", {
            "model_name": self.model_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
    def _save_event(self, event_type: str, data: dict):
        """Save event data to JSON file"""
        try:
            file_path = self.log_dir / f"{self.model_name}_{event_type}.json"
            
            # Load existing events
            events = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    events = json.load(f)
            
            # Append new event
            events.append(data)
            
            # Save updated events
            with open(file_path, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving event data: {str(e)}")

class DataLogger:
    """Logger specifically for data processing and validation"""
    
    def __init__(self, log_dir: str = "logs/data"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("data_processing")
        
    def log_data_validation(self, data_source: str, validation_results: dict):
        """Log data validation results"""
        self.logger.info(f"Data validation results for {data_source}")
        self._save_event("validation", {
            "data_source": data_source,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_data_processing(self, data_source: str, processing_details: dict):
        """Log data processing details"""
        self.logger.info(f"Processing data from {data_source}")
        self._save_event("processing", {
            "data_source": data_source,
            "processing_details": processing_details,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_data_error(self, data_source: str, error_msg: str):
        """Log data-related errors"""
        self.logger.error(f"Error processing data from {data_source}: {error_msg}")
        self._save_event("error", {
            "data_source": data_source,
            "error_message": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        
    def _save_event(self, event_type: str, data: dict):
        """Save event data to JSON file"""
        try:
            file_path = self.log_dir / f"data_{event_type}.json"
            
            # Load existing events
            events = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    events = json.load(f)
            
            # Append new event
            events.append(data)
            
            # Save updated events
            with open(file_path, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving event data: {str(e)}")

# Create global loggers
model_logger = ModelLogger("ensemble")
data_logger = DataLogger()
