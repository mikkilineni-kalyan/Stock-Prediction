from transformers import pipeline
from newsapi import NewsApiClient
import pandas as pd

class StockSentimentAnalyzer:
    def __init__(self, news_api_key):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.news_api = NewsApiClient(api_key=news_api_key)
        
    def get_news_sentiment(self, symbol, days=7):
        # Fetch news articles
        news = self.news_api.get_everything(
            q=symbol,
            language='en',
            sort_by='publishedAt',
            from_param=pd.Timestamp.now() - pd.Timedelta(days=days)
        )
        
        sentiments = []
        for article in news['articles']:
            sentiment = self.sentiment_analyzer(article['title'])[0]
            sentiments.append({
                'date': article['publishedAt'],
                'title': article['title'],
                'sentiment': sentiment['label'],
                'score': sentiment['score']
            })
            
        return pd.DataFrame(sentiments)
    
    def calculate_sentiment_score(self, sentiments_df):
        # Convert sentiment labels to numeric scores
        sentiment_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        sentiments_df['numeric_sentiment'] = sentiments_df['sentiment'].map(sentiment_map)
        weighted_sentiment = (sentiments_df['numeric_sentiment'] * 
                            sentiments_df['score']).mean()
        
        return weighted_sentiment 