import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

class MarketDataAnalyzer:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)  # Cache data for 5 minutes

    def get_stock_data(self, ticker: str, period: str = "1d", interval: str = "1h") -> pd.DataFrame:
        """Fetch real-time and historical stock data"""
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data

    def get_real_time_analysis(self, ticker: str) -> Dict[str, Any]:
        """Get real-time market analysis including volume, price movements"""
        data = self.get_stock_data(ticker, period="5d", interval="1h")
        
        if data.empty:
            return {"error": "No data available"}

        current_price = data['Close'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        
        # Calculate key metrics
        price_change = ((current_price - prev_close) / prev_close) * 100
        volume_avg = data['Volume'].mean()
        volume_ratio = volume / volume_avg

        return {
            "current_price": round(current_price, 2),
            "price_change_percent": round(price_change, 2),
            "volume": volume,
            "volume_ratio": round(volume_ratio, 2),
            "timestamp": datetime.now().isoformat(),
            "analysis": self._generate_market_analysis(price_change, volume_ratio)
        }

    def _generate_market_analysis(self, price_change: float, volume_ratio: float) -> str:
        """Generate market analysis based on price and volume"""
        analysis = []
        
        # Price analysis
        if abs(price_change) > 2:
            analysis.append(
                f"Significant {'upward' if price_change > 0 else 'downward'} price movement"
            )
        
        # Volume analysis
        if volume_ratio > 1.5:
            analysis.append("Above average trading volume")
        elif volume_ratio < 0.5:
            analysis.append("Below average trading volume")

        return " | ".join(analysis) if analysis else "Normal market conditions"

    def combine_with_news(self, ticker: str, news_data: Dict) -> Dict[str, Any]:
        """Combine market data with news analysis for comprehensive prediction"""
        market_data = self.get_real_time_analysis(ticker)
        
        # Get historical patterns
        hist_data = self.get_stock_data(ticker, period="3mo", interval="1d")
        
        prediction_score = self._calculate_prediction_score(
            market_data,
            news_data,
            hist_data
        )

        return {
            "ticker": ticker,
            "market_data": market_data,
            "news_analysis": news_data,
            "prediction": {
                "direction": "positive" if prediction_score > 0 else "negative",
                "score": abs(prediction_score),  # 1-5 scale
                "confidence": min(abs(prediction_score) / 5 * 100, 100),  # Percentage
                "timestamp": datetime.now().isoformat()
            }
        }

    def _calculate_prediction_score(
        self,
        market_data: Dict,
        news_data: Dict,
        hist_data: pd.DataFrame
    ) -> float:
        """Calculate final prediction score combining all factors"""
        # Market technical factors (30% weight)
        price_change = market_data['price_change_percent']
        volume_factor = market_data['volume_ratio'] - 1
        
        # News sentiment (40% weight)
        news_sentiment = news_data.get('confidence', 0) * (
            1 if news_data.get('prediction') == 'positive' else -1
        )
        
        # Historical patterns (30% weight)
        hist_momentum = self._calculate_momentum(hist_data)
        
        # Combine factors with weights
        final_score = (
            0.3 * (price_change / 2) +  # Normalize price change
            0.4 * news_sentiment +
            0.3 * hist_momentum
        )
        
        # Convert to 1-5 scale
        return max(min(final_score * 2.5, 5), -5)

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum from historical data"""
        if len(data) < 2:
            return 0
            
        returns = data['Close'].pct_change()
        return returns.mean() * 100  # Convert to percentage 