from fastapi.openapi.utils import get_openapi
from typing import Dict

def custom_openapi() -> Dict:
    """Generate custom OpenAPI documentation"""
    return {
        "openapi": "3.0.2",
        "info": {
            "title": "Stock Prediction API",
            "description": """
            # Stock Prediction System API
            
            This API provides endpoints for stock price prediction and analysis using advanced machine learning techniques.
            
            ## Key Features
            
            * Historical stock data retrieval and analysis
            * Technical indicator calculation
            * Price prediction using ensemble ML models
            * Trading signal generation
            * Model performance monitoring
            
            ## Authentication
            
            All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:
            ```
            Authorization: Bearer <your_token>
            ```
            
            ## Rate Limiting
            
            API calls are rate-limited to:
            * 100 requests per minute for standard users
            * 1000 requests per minute for premium users
            
            ## Error Handling
            
            The API uses standard HTTP status codes:
            * 200: Success
            * 400: Bad Request
            * 401: Unauthorized
            * 403: Forbidden
            * 404: Not Found
            * 429: Too Many Requests
            * 500: Internal Server Error
            
            Error responses include detailed error messages and error codes.
            """,
            "version": "1.0.0",
            "contact": {
                "name": "Stock Prediction Team",
                "email": "support@stockprediction.com",
                "url": "https://stockprediction.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "tags": [
            {
                "name": "Stock Data",
                "description": "Operations for retrieving and managing stock price data"
            },
            {
                "name": "Predictions",
                "description": "Stock price prediction endpoints"
            },
            {
                "name": "Models",
                "description": "Model management and monitoring"
            },
            {
                "name": "Trading Signals",
                "description": "Trading signal generation and analysis"
            },
            {
                "name": "Analysis",
                "description": "Technical and fundamental analysis"
            }
        ],
        "components": {
            "schemas": {
                "StockData": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "example": "AAPL"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "open": {"type": "number", "format": "float"},
                        "high": {"type": "number", "format": "float"},
                        "low": {"type": "number", "format": "float"},
                        "close": {"type": "number", "format": "float"},
                        "volume": {"type": "integer"}
                    },
                    "required": ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
                },
                "Prediction": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "predicted_price": {"type": "number", "format": "float"},
                        "confidence": {"type": "number", "format": "float"},
                        "prediction_horizon": {"type": "string"},
                        "features_used": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "TradingSignal": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "signal_type": {"type": "string", "enum": ["buy", "sell", "hold"]},
                        "confidence": {"type": "number", "format": "float"},
                        "factors": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    }
                },
                "ModelMetrics": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string"},
                        "accuracy": {"type": "number", "format": "float"},
                        "precision": {"type": "number", "format": "float"},
                        "recall": {"type": "number", "format": "float"},
                        "f1_score": {"type": "number", "format": "float"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"}
                    }
                }
            },
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
        },
        "security": [
            {"bearerAuth": []}
        ]
    }
