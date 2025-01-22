"""Documentation for database schema and data models."""

SCHEMA_DOCUMENTATION = {
    "stock_data": {
        "description": """
        Stores historical stock price and volume data.
        
        ## Schema Design
        
        The table uses a compound index on (symbol, timestamp) for efficient querying
        of historical data ranges. All price fields use DECIMAL type for accuracy
        in financial calculations.
        
        ## Data Quality
        
        * Prices are validated for consistency (high >= low, etc.)
        * Volume must be non-negative
        * Timestamps are stored in UTC
        * Missing values are handled according to data quality rules
        """,
        "fields": {
            "id": "Primary key",
            "symbol": "Stock symbol (e.g., AAPL)",
            "timestamp": "UTC timestamp of the data point",
            "open_price": "Opening price",
            "high_price": "Highest price during period",
            "low_price": "Lowest price during period",
            "close_price": "Closing price",
            "volume": "Trading volume",
            "created_at": "Record creation timestamp",
            "updated_at": "Last update timestamp"
        },
        "indexes": [
            "PRIMARY KEY (id)",
            "INDEX idx_symbol_timestamp (symbol, timestamp)",
            "INDEX idx_timestamp (timestamp)"
        ]
    },
    
    "features": {
        "description": """
        Stores feature definitions and metadata.
        
        ## Feature Categories
        
        Features are organized into categories:
        * Technical indicators
        * Market indicators
        * Sentiment indicators
        * Custom features
        
        ## Parameters
        
        Feature parameters are stored as JSON to allow flexible configuration
        while maintaining schema stability.
        """,
        "fields": {
            "id": "Primary key",
            "name": "Feature name",
            "description": "Feature description",
            "category": "Feature category",
            "parameters": "JSON configuration",
            "created_at": "Creation timestamp"
        },
        "indexes": [
            "PRIMARY KEY (id)",
            "UNIQUE INDEX idx_name (name)"
        ]
    },
    
    "feature_values": {
        "description": """
        Stores computed feature values.
        
        ## Storage Strategy
        
        Values are stored with their computation timestamp to track data freshness.
        The table is partitioned by symbol and timestamp for query performance.
        """,
        "fields": {
            "id": "Primary key",
            "feature_id": "Reference to features table",
            "symbol": "Stock symbol",
            "timestamp": "UTC timestamp",
            "value": "Computed feature value",
            "created_at": "Computation timestamp"
        },
        "indexes": [
            "PRIMARY KEY (id)",
            "INDEX idx_feature_symbol_timestamp (feature_id, symbol, timestamp)",
            "FOREIGN KEY (feature_id) REFERENCES features(id)"
        ]
    },
    
    "models": {
        "description": """
        Stores model metadata and configuration.
        
        ## Version Control
        
        Models are versioned with status tracking:
        * active: currently in use
        * archived: previous versions
        * failed: failed models
        
        ## Configuration
        
        Model parameters and metrics are stored as JSON for flexibility.
        """,
        "fields": {
            "id": "Primary key",
            "name": "Model name",
            "version": "Model version",
            "type": "Model type (ensemble, lstm, etc.)",
            "parameters": "JSON configuration",
            "metrics": "Performance metrics",
            "status": "Model status",
            "created_at": "Creation timestamp"
        },
        "indexes": [
            "PRIMARY KEY (id)",
            "INDEX idx_status (status)"
        ]
    },
    
    "predictions": {
        "description": """
        Stores model predictions.
        
        ## Tracking
        
        Each prediction is linked to its generating model and includes:
        * Confidence score
        * Features used
        * Timestamp for backtesting
        """,
        "fields": {
            "id": "Primary key",
            "model_id": "Reference to models table",
            "symbol": "Stock symbol",
            "timestamp": "Prediction timestamp",
            "prediction": "Predicted value",
            "confidence": "Confidence score",
            "features_used": "JSON array of features",
            "created_at": "Creation timestamp"
        },
        "indexes": [
            "PRIMARY KEY (id)",
            "INDEX idx_model_symbol_timestamp (model_id, symbol, timestamp)",
            "FOREIGN KEY (model_id) REFERENCES models(id)"
        ]
    },
    
    "model_performance": {
        "description": """
        Tracks model performance metrics over time.
        
        ## Metrics
        
        Multiple metrics are tracked:
        * Accuracy
        * Precision
        * Recall
        * Custom metrics
        
        ## Time Windows
        
        Metrics are computed over different windows:
        * Daily
        * Weekly
        * Monthly
        """,
        "fields": {
            "id": "Primary key",
            "model_id": "Reference to models table",
            "timestamp": "Metric timestamp",
            "metric_name": "Name of the metric",
            "metric_value": "Metric value",
            "symbol": "Stock symbol",
            "window": "Time window",
            "created_at": "Creation timestamp"
        },
        "indexes": [
            "PRIMARY KEY (id)",
            "INDEX idx_model_timestamp (model_id, timestamp)",
            "FOREIGN KEY (model_id) REFERENCES models(id)"
        ]
    }
}
