import os
import logging
from typing import List, Dict, Optional
import aiohttp
import asyncio
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
from textblob import TextBlob
import pandas as pd
from ..config.database import session, Stock, NewsArticle
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        self.session = session
        self.sources = {
            'reuters': 'https://www.reuters.com/markets/companies',
            'yahoo_finance': 'https://finance.yahoo.com/news',
            'marketwatch': 'https://www.marketwatch.com/latest-news'
        }
        self.cache = {}
        self.cache_timeout = 1800  # 30 minutes

    async def collect_news(self, symbol: str, company_name: str) -> List[Dict]:
        """
        Collect news from multiple sources and analyze sentiment
        Returns list of processed news articles
        """
        try:
            # Check cache
            cache_key = f"{symbol}_news"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    return cache_data

            # Collect news from multiple sources
            news_articles = []
            async with aiohttp.ClientSession() as session:
                tasks = []
                for source, url in self.sources.items():
                    tasks.append(self._fetch_news(session, source, url, symbol, company_name))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        news_articles.extend(result)

            # Process and analyze articles
            processed_articles = []
            for article in news_articles:
                processed = await self._process_article(article, symbol)
                if processed:
                    processed_articles.append(processed)
                    await self._store_article(processed, symbol)

            # Update cache
            self.cache[cache_key] = (time.time(), processed_articles)

            return processed_articles

        except Exception as e:
            logger.error(f"Error collecting news for {symbol}: {str(e)}")
            return []

    async def _fetch_news(self, session: aiohttp.ClientSession, source: str, 
                         base_url: str, symbol: str, company_name: str) -> List[Dict]:
        """Fetch news from a specific source"""
        try:
            search_terms = [symbol, company_name]
            articles = []

            async with session.get(base_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Different parsing logic for each source
                    if source == 'reuters':
                        articles = self._parse_reuters(soup, search_terms)
                    elif source == 'yahoo_finance':
                        articles = self._parse_yahoo(soup, search_terms)
                    elif source == 'marketwatch':
                        articles = self._parse_marketwatch(soup, search_terms)

            return articles

        except Exception as e:
            logger.error(f"Error fetching news from {source}: {str(e)}")
            return []

    def _parse_reuters(self, soup: BeautifulSoup, search_terms: List[str]) -> List[Dict]:
        """Parse Reuters articles"""
        articles = []
        for article in soup.find_all('article'):
            try:
                title = article.find('h3').text.strip()
                if any(term.lower() in title.lower() for term in search_terms):
                    articles.append({
                        'title': title,
                        'url': article.find('a')['href'],
                        'source': 'reuters',
                        'published_at': self._extract_date(article)
                    })
            except Exception as e:
                logger.debug(f"Error parsing Reuters article: {str(e)}")
        return articles

    def _parse_yahoo(self, soup: BeautifulSoup, search_terms: List[str]) -> List[Dict]:
        """Parse Yahoo Finance articles"""
        articles = []
        for article in soup.find_all('div', {'class': 'Cf'}):
            try:
                title = article.find('h3').text.strip()
                if any(term.lower() in title.lower() for term in search_terms):
                    articles.append({
                        'title': title,
                        'url': article.find('a')['href'],
                        'source': 'yahoo_finance',
                        'published_at': self._extract_date(article)
                    })
            except Exception as e:
                logger.debug(f"Error parsing Yahoo article: {str(e)}")
        return articles

    def _parse_marketwatch(self, soup: BeautifulSoup, search_terms: List[str]) -> List[Dict]:
        """Parse MarketWatch articles"""
        articles = []
        for article in soup.find_all('div', {'class': 'article__content'}):
            try:
                title = article.find('h3').text.strip()
                if any(term.lower() in title.lower() for term in search_terms):
                    articles.append({
                        'title': title,
                        'url': article.find('a')['href'],
                        'source': 'marketwatch',
                        'published_at': self._extract_date(article)
                    })
            except Exception as e:
                logger.debug(f"Error parsing MarketWatch article: {str(e)}")
        return articles

    async def _process_article(self, article: Dict, symbol: str) -> Optional[Dict]:
        """Process and analyze article content"""
        try:
            # Analyze sentiment
            blob = TextBlob(article['title'])
            sentiment_score = blob.sentiment.polarity
            
            # Calculate impact score (1-5)
            impact_score = self._calculate_impact_score(article, sentiment_score, symbol)
            
            # Add analysis results to article
            article['sentiment_score'] = sentiment_score
            article['impact_score'] = impact_score
            
            return article

        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")
            return None

    def _calculate_impact_score(self, article: Dict, sentiment_score: float, symbol: str) -> float:
        """Calculate impact score based on multiple factors"""
        try:
            score = 3.0  # Base score
            
            # Adjust based on sentiment strength
            score += sentiment_score * 1.5
            
            # Adjust based on source reliability
            source_weights = {
                'reuters': 1.2,
                'yahoo_finance': 1.0,
                'marketwatch': 1.0
            }
            score *= source_weights.get(article['source'], 1.0)
            
            # Ensure score is within 1-5 range
            score = max(1.0, min(5.0, score))
            
            return round(score, 2)

        except Exception as e:
            logger.error(f"Error calculating impact score: {str(e)}")
            return 3.0

    async def _store_article(self, article: Dict, symbol: str):
        """Store article in database"""
        try:
            # Get stock
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                return

            # Create news article record
            news = NewsArticle(
                stock_id=stock.id,
                title=article['title'],
                url=article['url'],
                source=article['source'],
                published_at=article['published_at'],
                sentiment_score=article['sentiment_score'],
                impact_score=article['impact_score']
            )
            
            self.session.add(news)
            self.session.commit()

        except SQLAlchemyError as e:
            logger.error(f"Database error storing article: {str(e)}")
            self.session.rollback()
        except Exception as e:
            logger.error(f"Error storing article: {str(e)}")
            self.session.rollback()

    def _extract_date(self, article_soup) -> datetime:
        """Extract publication date from article HTML"""
        try:
            date_str = article_soup.find('time').get('datetime', '')
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception:
            return datetime.utcnow()
