from .news_sources import NewsAnalyzer, create_news_sources, NEWS_SOURCES
import nltk
from typing import Dict, List

class EnhancedNewsAnalyzer(NewsAnalyzer):
    def __init__(self):
        super().__init__()
        self.sources = NEWS_SOURCES
        
    def analyze_news(self, ticker: str) -> Dict:
        """Enhanced news analysis with sentiment and impact scoring"""
        news_items = self.fetch_all_sources(ticker)
        
        if not news_items:
            return {
                'sentiment_score': 0,
                'impact': 'neutral',
                'confidence': 0.5,
                'sources': 0,
                'summary': 'No recent news found'
            }
        
        # Calculate aggregate sentiment and impact
        total_sentiment = 0
        total_impact = 0
        
        for item in news_items:
            total_sentiment += item['sentiment_score']
            total_impact += item['impact_score']
            
        avg_sentiment = total_sentiment / len(news_items)
        avg_impact = total_impact / len(news_items)
        
        return {
            'sentiment_score': avg_sentiment,
            'impact': 'positive' if avg_sentiment > 0 else 'negative',
            'confidence': min(len(news_items) / 10, 1.0),
            'sources': len(news_items),
            'summary': self._generate_summary(news_items[:3])
        }
    
    def _generate_summary(self, news_items: List[Dict]) -> str:
        """Generate a summary of the most important news"""
        if not news_items:
            return "No recent news available"
            
        summaries = []
        for item in news_items:
            sentiment = "positive" if item['sentiment_score'] > 0 else "negative"
            summaries.append(f"{item['title']} ({sentiment} impact)")
            
        return " | ".join(summaries) 