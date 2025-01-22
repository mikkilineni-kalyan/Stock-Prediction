"""Documentation for ML models and algorithms."""

MODEL_DOCUMENTATION = {
    "ensemble_model": {
        "name": "Advanced Ensemble Model",
        "description": """
        A sophisticated ensemble model combining multiple ML algorithms for stock price prediction.
        
        ## Architecture
        
        The model uses a hierarchical ensemble approach:
        
        1. Base Models:
           * LSTM for sequence modeling
           * CNN for pattern recognition
           * Random Forest for feature importance
           * XGBoost for non-linear relationships
           * LightGBM for gradient boosting
        
        2. Meta-Learning Layer:
           * Weighted averaging of base models
           * Dynamic weight adjustment
           * Confidence score calculation
        
        ## Features Used
        
        1. Technical Indicators:
           * Moving averages (SMA, EMA)
           * Momentum indicators (RSI, MACD)
           * Volatility indicators (Bollinger Bands)
           * Volume indicators (OBV, Volume Profile)
        
        2. Market Data:
           * Price action patterns
           * Volume analysis
           * Market regime indicators
           * Cross-asset correlations
        
        3. Sentiment Features:
           * News sentiment scores
           * Social media sentiment
           * Market sentiment indicators
           * Trading activity sentiment
        
        ## Training Process
        
        1. Data Preparation:
           * Quality checks
           * Normalization
           * Feature engineering
           * Sequence preparation
        
        2. Model Training:
           * Individual model training
           * Cross-validation
           * Hyperparameter optimization
           * Ensemble weight optimization
        
        3. Validation:
           * Out-of-sample testing
           * Backtesting
           * Performance metrics calculation
        
        ## Performance Metrics
        
        The model is evaluated on:
        * Directional accuracy
        * Mean absolute error (MAE)
        * Root mean square error (RMSE)
        * Sharpe ratio
        * Maximum drawdown
        
        ## Deployment
        
        The model supports:
        * Online learning
        * Real-time predictions
        * Model versioning
        * Performance monitoring
        """,
        "parameters": {
            "learning_rate": {
                "description": "Learning rate for model updates",
                "default": 0.001,
                "range": [0.0001, 0.01]
            },
            "batch_size": {
                "description": "Batch size for training",
                "default": 32,
                "range": [16, 128]
            },
            "sequence_length": {
                "description": "Length of input sequences",
                "default": 60,
                "range": [20, 120]
            },
            "ensemble_weights": {
                "description": "Initial weights for ensemble models",
                "default": {
                    "lstm": 0.3,
                    "cnn": 0.2,
                    "random_forest": 0.2,
                    "xgboost": 0.15,
                    "lightgbm": 0.15
                }
            }
        },
        "usage": """
        ```python
        from backend.ml_models import AdvancedEnsembleModel
        
        # Initialize model
        model = AdvancedEnsembleModel()
        
        # Train model
        model.train(X_train, y_train, validation_data=(X_val, y_val))
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Get prediction confidence
        confidence = model.get_confidence_scores(X_test)
        
        # Update model with new data
        model.update(X_new, y_new)
        ```
        """
    },
    
    "feature_engineering": {
        "name": "Feature Engineering Pipeline",
        "description": """
        Comprehensive feature engineering pipeline for stock price prediction.
        
        ## Feature Categories
        
        1. Technical Features:
           * Price-based indicators
           * Volume indicators
           * Volatility measures
           * Momentum indicators
        
        2. Market Features:
           * Market regime indicators
           * Cross-asset correlations
           * Market microstructure
           * Liquidity measures
        
        3. Sentiment Features:
           * News sentiment
           * Social media sentiment
           * Market sentiment
           * Options sentiment
        
        ## Implementation
        
        The pipeline uses:
        * Efficient computation methods
        * Caching for performance
        * Quality checks
        * Feature selection
        """,
        "usage": """
        ```python
        from backend.data_processing import FeatureEngineer
        
        # Initialize engineer
        engineer = FeatureEngineer()
        
        # Generate features
        features = engineer.generate_features(stock_data)
        
        # Select important features
        selected_features = engineer.select_features(features)
        ```
        """
    }
}
