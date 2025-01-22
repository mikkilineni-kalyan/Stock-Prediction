"""API endpoint documentation."""

# Stock Data Endpoints
STOCK_DATA_DOCS = {
    "get_historical_data": {
        "summary": "Get historical stock data",
        "description": """
        Retrieve historical stock price data for a given symbol and date range.
        
        The data includes:
        * Open, high, low, close prices
        * Trading volume
        * Adjusted close price
        * Additional market data if available
        
        Data is sourced from multiple providers and validated for accuracy.
        """,
        "response_description": "Historical stock data points"
    },
    
    "get_latest_price": {
        "summary": "Get latest stock price",
        "description": """
        Retrieve the most recent stock price data.
        
        Includes real-time or delayed quotes depending on your subscription level.
        """,
        "response_description": "Latest stock price data"
    }
}

# Prediction Endpoints
PREDICTION_DOCS = {
    "get_price_prediction": {
        "summary": "Get stock price prediction",
        "description": """
        Generate price predictions using our ensemble ML model.
        
        The prediction includes:
        * Predicted price
        * Confidence score
        * Prediction horizon
        * Contributing factors
        
        Multiple timeframes are supported: 1d, 7d, 30d, 90d
        """,
        "response_description": "Price prediction with confidence metrics"
    },
    
    "get_prediction_accuracy": {
        "summary": "Get prediction accuracy metrics",
        "description": """
        Retrieve accuracy metrics for previous predictions.
        
        Metrics include:
        * Mean Absolute Error (MAE)
        * Root Mean Square Error (RMSE)
        * Directional Accuracy
        * Sharpe Ratio
        """,
        "response_description": "Historical prediction accuracy metrics"
    }
}

# Model Management Endpoints
MODEL_DOCS = {
    "get_model_status": {
        "summary": "Get model status",
        "description": """
        Retrieve current status and performance metrics of the prediction model.
        
        Includes:
        * Model version
        * Training status
        * Performance metrics
        * Last update time
        * Feature importance
        """,
        "response_description": "Model status and metrics"
    },
    
    "update_model": {
        "summary": "Update prediction model",
        "description": """
        Trigger model update with new training data.
        
        The update process:
        1. Validates new data
        2. Retrains model
        3. Evaluates performance
        4. Updates if performance improves
        """,
        "response_description": "Model update status"
    }
}

# Trading Signal Endpoints
SIGNAL_DOCS = {
    "get_trading_signals": {
        "summary": "Get trading signals",
        "description": """
        Generate trading signals based on multiple factors:
        * Technical analysis
        * Price predictions
        * Market sentiment
        * News analysis
        
        Signals include confidence scores and contributing factors.
        """,
        "response_description": "Trading signals with analysis"
    },
    
    "get_signal_performance": {
        "summary": "Get signal performance metrics",
        "description": """
        Retrieve performance metrics for historical trading signals.
        
        Metrics include:
        * Signal accuracy
        * Profit/Loss ratio
        * Risk metrics
        * Timing accuracy
        """,
        "response_description": "Signal performance metrics"
    }
}

# Analysis Endpoints
ANALYSIS_DOCS = {
    "get_technical_analysis": {
        "summary": "Get technical analysis",
        "description": """
        Perform comprehensive technical analysis including:
        * Moving averages
        * Momentum indicators
        * Volatility analysis
        * Volume analysis
        * Pattern recognition
        """,
        "response_description": "Technical analysis results"
    },
    
    "get_sentiment_analysis": {
        "summary": "Get market sentiment analysis",
        "description": """
        Analyze market sentiment from multiple sources:
        * News articles
        * Social media
        * Market indicators
        * Trading activity
        
        Includes sentiment scores and confidence metrics.
        """,
        "response_description": "Sentiment analysis results"
    }
}
