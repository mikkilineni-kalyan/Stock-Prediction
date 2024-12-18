from typing import Dict, Any
from datetime import datetime
import numpy as np

class ConfidenceAdjuster:
    @staticmethod
    def adjust_confidence(prediction: Dict[str, Any], 
                         days_ahead: int,
                         market_volatility: float,
                         news_impact: float) -> float:
        """
        Adjust prediction confidence based on multiple factors
        """
        base_confidence = prediction['confidence']
        
        # Time decay factor (confidence decreases with prediction distance)
        time_decay = 1 / (1 + (days_ahead * 0.15))
        
        # Volatility impact (higher volatility = lower confidence)
        volatility_factor = 1 - min(market_volatility, 0.5)
        
        # News impact (stronger news sentiment = higher confidence)
        news_factor = 1 + (abs(news_impact) * 0.2)
        
        # Market hours factor (lower confidence outside market hours)
        market_hours_factor = 0.9 if not is_market_open() else 1.0
        
        # Combine all factors
        adjusted_confidence = (
            base_confidence *
            time_decay *
            volatility_factor *
            news_factor *
            market_hours_factor
        )
        
        return min(adjusted_confidence, 1.0)
    
    @staticmethod
    def calculate_prediction_intervals(
        prediction: float,
        confidence: float,
        volatility: float,
        days_ahead: int
    ) -> Dict[str, float]:
        """
        Calculate prediction intervals based on confidence and volatility
        """
        # Base interval width based on confidence
        interval_width = (1 - confidence) * volatility
        
        # Widen interval based on prediction distance
        time_factor = 1 + (days_ahead * 0.1)
        interval_width *= time_factor
        
        return {
            'lower': prediction * (1 - interval_width),
            'upper': prediction * (1 + interval_width),
            'interval_width': interval_width
        } 