import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from textblob import TextBlob
import finnhub
import praw
import tweepy
from dotenv import load_dotenv

load_dotenv()

class NewsService:
    def __init__(self):
        # Initialize API clients
        self.finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
        self.setup_reddit_client()
        self.setup_twitter_client()
    
    def setup_reddit_client(self):
        """Initialize Reddit client"""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent='StockPrediction/1.0'
            )
        except Exception as e:
            print(f"Error setting up Reddit client: {str(e)}")
            self.reddit = None

    def setup_twitter_client(self):
        """Initialize Twitter client"""
        try:
            auth = tweepy.OAuthHandler(
                os.getenv('TWITTER_API_KEY'),
                os.getenv('TWITTER_API_SECRET')
            )
            self.twitter = tweepy.API(auth)
        except Exception as e:
            print(f"Error setting up Twitter client: {str(e)}")
            self.twitter = None

    def get_news_alpha_vantage(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': os.getenv('ALPHA_VANTAGE_API_KEY')
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            news_items = []
            for item in data.get('feed', []):
                news_items.append({
                    'source': 'Alpha Vantage',
                    'title': item.get('title'),
                    'summary': item.get('summary'),
                    'url': item.get('url'),
                    'sentiment': float(item.get('overall_sentiment_score', 0)),
                    'published_at': datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S')
                })
            return news_items
        except Exception as e:
            print(f"Error fetching Alpha Vantage news: {str(e)}")
            return []

    def get_news_finnhub(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from Finnhub"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            news = self.finnhub_client.company_news(
                ticker,
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            news_items = []
            for item in news:
                sentiment = self._analyze_sentiment(f"{item.get('headline', '')} {item.get('summary', '')}")
                news_items.append({
                    'source': 'Finnhub',
                    'title': item.get('headline'),
                    'summary': item.get('summary'),
                    'url': item.get('url'),
                    'sentiment': sentiment,
                    'published_at': datetime.fromtimestamp(item.get('datetime', 0))
                })
            return news_items
        except Exception as e:
            print(f"Error fetching Finnhub news: {str(e)}")
            return []

    def get_news_yahoo(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            news_items = []
            for item in news:
                sentiment = self._analyze_sentiment(f"{item.get('title', '')} {item.get('summary', '')}")
                news_items.append({
                    'source': 'Yahoo Finance',
                    'title': item.get('title'),
                    'summary': item.get('summary'),
                    'url': item.get('link'),
                    'sentiment': sentiment,
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0))
                })
            return news_items
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {str(e)}")
            return []

    def get_reddit_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get Reddit sentiment for a ticker"""
        try:
            if not self.reddit:
                return None
                
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            posts = []
            
            for subreddit in subreddits:
                for post in self.reddit.subreddit(subreddit).search(ticker, limit=50, time_filter='week'):
                    sentiment = self._analyze_sentiment(f"{post.title} {post.selftext}")
                    posts.append({
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'sentiment': sentiment,
                        'created_at': datetime.fromtimestamp(post.created_utc)
                    })
            
            if not posts:
                return None
                
            avg_sentiment = sum(post['sentiment'] for post in posts) / len(posts)
            avg_score = sum(post['score'] for post in posts) / len(posts)
            
            return {
                'average_sentiment': avg_sentiment,
                'average_score': avg_score,
                'post_count': len(posts),
                'latest_posts': sorted(posts, key=lambda x: x['score'], reverse=True)[:5]
            }
        except Exception as e:
            print(f"Error getting Reddit sentiment: {str(e)}")
            return None

    def get_twitter_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get Twitter sentiment for a ticker"""
        try:
            if not self.twitter:
                return None
                
            search_query = f"${ticker} -filter:retweets"
            tweets = []
            
            for tweet in tweepy.Cursor(self.twitter.search_tweets, q=search_query, lang="en", tweet_mode="extended").items(100):
                sentiment = self._analyze_sentiment(tweet.full_text)
                tweets.append({
                    'text': tweet.full_text,
                    'sentiment': sentiment,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'created_at': tweet.created_at
                })
            
            if not tweets:
                return None
                
            avg_sentiment = sum(tweet['sentiment'] for tweet in tweets) / len(tweets)
            avg_engagement = sum(tweet['retweet_count'] + tweet['favorite_count'] for tweet in tweets) / len(tweets)
            
            return {
                'average_sentiment': avg_sentiment,
                'average_engagement': avg_engagement,
                'tweet_count': len(tweets),
                'latest_tweets': sorted(tweets, key=lambda x: x['retweet_count'] + x['favorite_count'], reverse=True)[:5]
            }
        except Exception as e:
            print(f"Error getting Twitter sentiment: {str(e)}")
            return None

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            analysis = TextBlob(text)
            # Convert polarity (-1 to 1) to our scale (1 to 5)
            return (analysis.sentiment.polarity + 1) * 2 + 1
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return 3.0  # Neutral sentiment as fallback

    def get_all_news(self, ticker: str) -> Dict[str, Any]:
        """Get news and sentiment from all sources"""
        # Fetch news from all sources
        alpha_vantage_news = self.get_news_alpha_vantage(ticker)
        finnhub_news = self.get_news_finnhub(ticker)
        yahoo_news = self.get_news_yahoo(ticker)
        
        # Get social media sentiment
        reddit_sentiment = self.get_reddit_sentiment(ticker)
        twitter_sentiment = self.get_twitter_sentiment(ticker)
        
        # Combine all news
        all_news = alpha_vantage_news + finnhub_news + yahoo_news
        
        # Sort by publish date
        all_news.sort(key=lambda x: x['published_at'], reverse=True)
        
        # Calculate overall sentiment
        if all_news:
            news_sentiment = sum(item['sentiment'] for item in all_news) / len(all_news)
        else:
            news_sentiment = 3.0  # Neutral if no news
            
        # Calculate confidence based on amount and consistency of data
        news_count = len(all_news)
        sentiment_std = (
            float(np.std([item['sentiment'] for item in all_news]))
            if all_news else 1.0
        )
        
        confidence = min((
            0.5 * min(news_count / 20, 1.0) +  # News volume factor
            0.3 * (1 - min(sentiment_std / 2, 0.5)) +  # Sentiment consistency factor
            0.2 * (1 if reddit_sentiment and twitter_sentiment else 0.5)  # Social media factor
        ), 1.0)
        
        return {
            'news': all_news[:20],  # Latest 20 news items
            'sentiment': {
                'overall_score': news_sentiment,
                'confidence': confidence,
                'news_count': news_count,
                'reddit': reddit_sentiment,
                'twitter': twitter_sentiment
            }
        }
