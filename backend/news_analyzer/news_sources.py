import requests
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import feedparser
import numpy as np

def create_news_sources():
    return {
        'reuters': 'https://www.reuters.com/markets/companies',
        'bloomberg': 'https://www.bloomberg.com/markets',
        'yahoo_finance': 'https://finance.yahoo.com/news',
        'seeking_alpha': 'https://seekingalpha.com/market-news/all',
        'marketwatch': 'https://www.marketwatch.com/markets',
        'financial_times': 'https://www.ft.com/markets',
        'sec_filings': 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=&company=&dateb=&owner=include&start=0&count=40&output=atom',
        'nasdaq_news': 'https://www.nasdaq.com/feed/rssoutbound?category=Markets'
    }

NEWS_SOURCES = create_news_sources()

class NewsAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        self.news_sources = NEWS_SOURCES
        
        # Additional sources for long-term analysis
        self.long_term_sources = {
            'earnings_calendar': 'https://finance.yahoo.com/calendar/earnings',
            'economic_calendar': 'https://www.investing.com/economic-calendar/',
            'sec_filings_direct': 'https://www.sec.gov/edgar/searchedgar/companysearch.html'
        }

    def fetch_all_sources(self, ticker: str) -> List[Dict]:
        """Fetch news from all sources"""
        news_items = []
        
        # Yahoo Finance API
        news_items.extend(self.fetch_stock_news(ticker))
        
        # RSS Feeds
        news_items.extend(self.fetch_rss_news(ticker))
        
        # SEC Filings
        news_items.extend(self.fetch_sec_filings(ticker))
        
        return self.deduplicate_news(news_items)

    def fetch_rss_news(self, ticker: str) -> List[Dict]:
        """Fetch news from RSS feeds"""
        news_items = []
        
        for source, url in self.news_sources.items():
            if url.endswith('atom') or url.endswith('rss'):
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries:
                        if ticker.lower() in entry.title.lower() or ticker.lower() in entry.summary.lower():
                            sentiment_score = self.analyze_sentiment(entry.title)
                            impact_score = self.calculate_impact_score(sentiment_score, {'source': source})
                            
                            news_items.append({
                                'title': entry.title,
                                'source': source,
                                'published': datetime(*entry.published_parsed[:6]),
                                'url': entry.link,
                                'sentiment_score': sentiment_score,
                                'impact_score': impact_score,
                                'prediction': 'positive' if sentiment_score > 0 else 'negative'
                            })
                except Exception as e:
                    print(f"Error fetching RSS from {source}: {str(e)}")
                    
        return news_items

    def fetch_sec_filings(self, ticker: str) -> List[Dict]:
        """Fetch SEC filings"""
        news_items = []
        try:
            # Use SEC API to fetch recent filings
            url = f"https://data.sec.gov/submissions/CIK{ticker.zfill(10)}.json"
            headers = {
                'User-Agent': 'Your Company Name admin@company.com'
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                filings = response.json().get('filings', {}).get('recent', [])
                for filing in filings:
                    news_items.append({
                        'title': f"SEC Filing: {filing.get('form')}",
                        'source': 'SEC',
                        'published': datetime.strptime(filing.get('filingDate'), '%Y-%m-%d'),
                        'url': f"https://www.sec.gov/Archives/edgar/data/{filing.get('accessionNumber')}",
                        'sentiment_score': 0,  # Neutral for filings
                        'impact_score': 4,  # SEC filings usually important
                        'prediction': 'neutral'
                    })
        except Exception as e:
            print(f"Error fetching SEC filings: {str(e)}")
            
        return news_items

    def deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """Remove duplicate news items based on title or URL"""
        seen = set()
        unique_news = []
        
        for item in news_items:
            key = item.get('title', '') + item.get('url', '')
            if key not in seen:
                seen.add(key)
                unique_news.append(item)
        
        return unique_news

    def fetch_stock_news(self, ticker: str) -> List[Dict]:
        """Fetch news for a specific stock ticker"""
        news_items = []
        
        # Get news from Yahoo Finance API
        stock = yf.Ticker(ticker)
        yahoo_news = stock.news
        
        for article in yahoo_news:
            sentiment_score = self.analyze_sentiment(article['title'])
            impact_score = self.calculate_impact_score(sentiment_score, article)
            
            news_items.append({
                'title': article['title'],
                'source': article['source'],
                'published': datetime.fromtimestamp(article['providerPublishTime']),
                'url': article['link'],
                'sentiment_score': sentiment_score,
                'impact_score': impact_score,  # 1-5 scale as per original approach
                'prediction': 'positive' if sentiment_score > 0 else 'negative'
            })
        
        return news_items

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of news headline"""
        sentiment = self.sia.polarity_scores(text)
        return sentiment['compound']

    def calculate_impact_score(self, sentiment_score: float, article: Dict) -> int:
        """Calculate impact score (1-5) based on sentiment and other factors"""
        # Convert sentiment (-1 to 1) to 1-5 scale
        base_score = abs(sentiment_score) * 2.5 + 2.5
        
        # Adjust based on source reliability
        source_weight = 1.2 if article['source'] in ['Reuters', 'Bloomberg'] else 1.0
        
        # Final score calculation
        final_score = min(5, max(1, round(base_score * source_weight)))
        return final_score

    def get_hourly_predictions(self, ticker: str) -> Dict:
        """Get hourly predictions based on news analysis"""
        news = self.fetch_stock_news(ticker)
        if not news:
            return {
                "ticker": ticker,
                "prediction": "neutral",
                "confidence": 0,
                "impact_score": 0,
                "analysis": "Insufficient news data"
            }

        # Filter significant news (impact score >= 3)
        significant_news = [n for n in news if n['impact_score'] >= 3]
        
        if significant_news:
            # Calculate average sentiment and impact
            avg_sentiment = sum(n['sentiment_score'] for n in significant_news) / len(significant_news)
            avg_impact = sum(n['impact_score'] for n in significant_news) / len(significant_news)
            
            return {
                "ticker": ticker,
                "prediction": "positive" if avg_sentiment > 0 else "negative",
                "confidence": abs(avg_sentiment),
                "impact_score": round(avg_impact, 1),
                "analysis": self._generate_analysis(significant_news),
                "news_items": significant_news
            }
        
        return {"ticker": ticker, "prediction": "neutral", "impact_score": 0}

    def _generate_analysis(self, news_items: List[Dict]) -> str:
        """Generate analysis explanation based on news items"""
        if not news_items:
            return "No significant news found"
            
        most_impactful = max(news_items, key=lambda x: x['impact_score'])
        return f"Primary driver: {most_impactful['title']} (Impact: {most_impactful['impact_score']}/5)" 

    def predict_impact(self, ticker: str) -> Dict:
        """Predict the impact of news on stock price"""
        try:
            news_items = self.fetch_all_sources(ticker)
            if not news_items:
                return {
                    'prediction': 'neutral',
                    'confidence': 0,
                    'sentiment_score': 0,
                    'news_count': 0
                }
            
            total_sentiment = 0
            for item in news_items:
                text = f"{item.get('title', '')} {item.get('description', '')}"
                sentiment = self.sia.polarity_scores(text)
                total_sentiment += sentiment['compound']
            
            avg_sentiment = total_sentiment / len(news_items)
            
            if avg_sentiment > 0.2:
                prediction = 'positive'
            elif avg_sentiment < -0.2:
                prediction = 'negative'
            else:
                prediction = 'neutral'
            
            return {
                'prediction': prediction,
                'confidence': min(abs(avg_sentiment) * 100, 100),
                'sentiment_score': avg_sentiment,
                'news_count': len(news_items)
            }
        except Exception as e:
            print(f"Error in predict_impact: {str(e)}")
            return {
                'prediction': 'error',
                'confidence': 0,
                'sentiment_score': 0,
                'news_count': 0,
                'error': str(e)
            }

    def analyze_historical_correlation(self, ticker: str) -> Dict:
        """Analyze historical correlation between news sentiment and price movements"""
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period='1mo')
            if hist_data.empty:
                return {'correlation': 0, 'data_points': 0}
            
            price_changes = hist_data['Close'].pct_change().dropna()
            return {
                'correlation': 0.5,  # Placeholder correlation
                'data_points': len(price_changes),
                'period': '1 month'
            }
        except Exception as e:
            print(f"Error in historical correlation: {str(e)}")
            return {
                'correlation': 0,
                'data_points': 0,
                'error': str(e)
            }