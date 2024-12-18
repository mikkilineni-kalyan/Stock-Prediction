import requests
from datetime import datetime, timedelta

class NewsPredictor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def predict(self, symbol):
        try:
            # Get news articles for the past week
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            params = {
                'q': f'{symbol} stock',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key
            }
            
            if not self.api_key:
                # Return mock data if no API key
                return {
                    'prediction': 'neutral',
                    'confidence': 0.6,
                    'news_sentiment': 'neutral',
                    'articles': []
                }
            
            response = requests.get(self.base_url, params=params)
            articles = response.json().get('articles', [])
            
            # Simple sentiment analysis based on titles
            positive_words = ['surge', 'jump', 'rise', 'gain', 'up', 'high', 'growth', 'profit']
            negative_words = ['fall', 'drop', 'decline', 'loss', 'down', 'low', 'crash', 'risk']
            
            sentiment_score = 0
            for article in articles:
                title = article['title'].lower()
                sentiment_score += sum(1 for word in positive_words if word in title)
                sentiment_score -= sum(1 for word in negative_words if word in title)
            
            # Normalize sentiment score
            if articles:
                sentiment_score = sentiment_score / len(articles)
            
            # Generate prediction
            if sentiment_score > 0.2:
                prediction = 'bullish'
                confidence = min(0.5 + abs(sentiment_score) * 0.3, 0.9)
            elif sentiment_score < -0.2:
                prediction = 'bearish'
                confidence = min(0.5 + abs(sentiment_score) * 0.3, 0.9)
            else:
                prediction = 'neutral'
                confidence = 0.6
            
            # Get sentiment label
            if sentiment_score > 0.1:
                sentiment = 'positive'
            elif sentiment_score < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 2),
                'news_sentiment': sentiment,
                'articles': articles[:5]  # Return top 5 articles
            }
            
        except Exception as e:
            print(f"Error in news prediction: {e}")
            return {
                'prediction': 'neutral',
                'confidence': 0.5,
                'news_sentiment': 'neutral',
                'articles': []
            }
