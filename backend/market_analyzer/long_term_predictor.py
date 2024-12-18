from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from .market_data import MarketDataAnalyzer

class LongTermPredictor:
    def __init__(self):
        self.market_analyzer = MarketDataAnalyzer()
        
    def get_weekly_prediction(self, ticker: str, news_data: List[Dict]) -> Dict:
        """Generate weekly prediction based on news and market data"""
        # Get 3 months of historical data for pattern analysis
        hist_data = self.market_analyzer.get_stock_data(ticker, period="3mo", interval="1d")
        
        # Analyze patterns
        pattern_score = self._analyze_patterns(hist_data)
        
        # Analyze news trends
        news_trend = self._analyze_news_trend(news_data)
        
        # Combine analyses
        prediction = self._combine_weekly_analysis(pattern_score, news_trend)
        
        return {
            "ticker": ticker,
            "timeframe": "weekly",
            "prediction": prediction,
            "analysis": self._generate_weekly_analysis(prediction, pattern_score, news_trend),
            "confidence": prediction['confidence'],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_monthly_prediction(self, ticker: str, news_data: List[Dict]) -> Dict:
        """Generate monthly prediction based on extended analysis"""
        # Get 1 year of historical data for long-term patterns
        hist_data = self.market_analyzer.get_stock_data(ticker, period="1y", interval="1d")
        
        # Analyze long-term patterns
        pattern_score = self._analyze_long_term_patterns(hist_data)
        
        # Analyze news and SEC filings
        news_trend = self._analyze_news_trend(news_data, long_term=True)
        
        # Combine analyses
        prediction = self._combine_monthly_analysis(pattern_score, news_trend)
        
        return {
            "ticker": ticker,
            "timeframe": "monthly",
            "prediction": prediction,
            "analysis": self._generate_monthly_analysis(prediction, pattern_score, news_trend),
            "confidence": prediction['confidence'],
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze price patterns for weekly prediction"""
        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(5)  # Weekly volatility
        momentum = returns.rolling(5).mean().iloc[-1]
        
        return {
            "momentum": momentum,
            "volatility": volatility,
            "trend": "up" if momentum > 0 else "down",
            "strength": abs(momentum) / volatility
        }

    def _analyze_long_term_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze price patterns for monthly prediction"""
        returns = data['Close'].pct_change()
        monthly_returns = returns.rolling(21).mean()
        trend_strength = monthly_returns.iloc[-1] / returns.std()
        
        return {
            "trend": "up" if monthly_returns.iloc[-1] > 0 else "down",
            "strength": abs(trend_strength),
            "volatility": returns.std() * np.sqrt(21)  # Monthly volatility
        }

    def _analyze_news_trend(self, news_data: List[Dict], long_term: bool = False) -> Dict:
        """Analyze news sentiment trend"""
        if not news_data:
            return {"sentiment": "neutral", "strength": 0}
            
        # Filter by date for relevant timeframe
        cutoff = datetime.now() - timedelta(days=30 if long_term else 7)
        recent_news = [n for n in news_data if n['published'] >= cutoff]
        
        if not recent_news:
            return {"sentiment": "neutral", "strength": 0}
            
        # Calculate average sentiment
        avg_sentiment = sum(n['sentiment_score'] for n in recent_news) / len(recent_news)
        
        return {
            "sentiment": "positive" if avg_sentiment > 0 else "negative",
            "strength": abs(avg_sentiment)
        }

    def _combine_weekly_analysis(self, pattern_score: Dict, news_trend: Dict) -> Dict:
        """Combine different analyses for weekly prediction"""
        # Weight: 60% patterns, 40% news for weekly
        pattern_weight = 0.6
        news_weight = 0.4
        
        combined_score = (
            pattern_weight * pattern_score['strength'] * (1 if pattern_score['trend'] == 'up' else -1) +
            news_weight * news_trend['strength'] * (1 if news_trend['sentiment'] == 'positive' else -1)
        )
        
        return {
            "direction": "positive" if combined_score > 0 else "negative",
            "score": min(abs(combined_score) * 5, 5),  # Convert to 1-5 scale
            "confidence": min(abs(combined_score) * 100, 100)  # Convert to percentage
        }

    def _combine_monthly_analysis(self, pattern_score: Dict, news_trend: Dict) -> Dict:
        """Combine different analyses for monthly prediction"""
        # Weight: 70% patterns, 30% news for monthly
        pattern_weight = 0.7
        news_weight = 0.3
        
        combined_score = (
            pattern_weight * pattern_score['strength'] * (1 if pattern_score['trend'] == 'up' else -1) +
            news_weight * news_trend['strength'] * (1 if news_trend['sentiment'] == 'positive' else -1)
        )
        
        return {
            "direction": "positive" if combined_score > 0 else "negative",
            "score": min(abs(combined_score) * 5, 5),
            "confidence": min(abs(combined_score) * 100, 100)
        }

    def _generate_weekly_analysis(self, prediction: Dict, pattern_score: Dict, news_trend: Dict) -> str:
        """Generate detailed analysis explanation for weekly prediction"""
        analysis = []
        
        # Pattern analysis
        analysis.append(f"Technical Analysis: {pattern_score['trend'].title()} trend with "
                       f"{'high' if pattern_score['strength'] > 0.5 else 'moderate'} strength")
        
        # News analysis
        if news_trend['strength'] > 0:
            analysis.append(f"News Sentiment: {news_trend['sentiment'].title()} with "
                          f"{'strong' if news_trend['strength'] > 0.5 else 'moderate'} impact")
        
        # Confidence explanation
        analysis.append(f"Prediction Confidence: {prediction['confidence']:.1f}%")
        
        return " | ".join(analysis)

    def _generate_monthly_analysis(self, prediction: Dict, pattern_score: Dict, news_trend: Dict) -> str:
        """Generate detailed analysis explanation for monthly prediction"""
        analysis = []
        
        # Long-term trend analysis
        analysis.append(f"Long-term Trend: {pattern_score['trend'].title()} trend with "
                       f"{'high' if pattern_score['strength'] > 0.5 else 'moderate'} strength")
        
        # Volatility assessment
        volatility_level = 'high' if pattern_score['volatility'] > 0.02 else 'moderate'
        analysis.append(f"Market Volatility: {volatility_level}")
        
        # News trend
        if news_trend['strength'] > 0:
            analysis.append(f"News Trend: {news_trend['sentiment'].title()} with "
                          f"{'strong' if news_trend['strength'] > 0.5 else 'moderate'} impact")
        
        # Confidence explanation
        analysis.append(f"Prediction Confidence: {prediction['confidence']:.1f}%")
        
        return " | ".join(analysis) 