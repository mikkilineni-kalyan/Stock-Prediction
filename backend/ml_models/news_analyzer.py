import os

import finnhub

import newsapi

from transformers import pipeline

from datetime import datetime, timedelta

import logging

import pandas as pd

import numpy as np



logger = logging.getLogger(__name__)



class AdvancedNewsAnalyzer:

    def __init__(self):

        self.news_api = newsapi.NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

        self.finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

        self.sentiment_model = pipeline("sentiment-analysis")

        self.source_weights = {

            'reuters.com': 1.5,

            'bloomberg.com': 1.5,

            'wsj.com': 1.4,

            'ft.com': 1.4,

            'cnbc.com': 1.2

        }

        

    async def analyze_impact(self, symbol):

        try:

            # Get news from multiple sources

            news_data = await self._fetch_all_sources(symbol)

            

            # Analyze sentiment

            sentiment_scores = self._analyze_sentiment(news_data)

            

            # Analyze volume and frequency

            volume_analysis = self._analyze_news_volume(news_data)

            

            # Get sector news impact

            sector_impact = await self._analyze_sector_impact(symbol)

            

            # Combine all analyses

            return {

                'sentiment': sentiment_scores,

                'volume': volume_analysis,

                'sector_impact': sector_impact,

                'overall_score': self._calculate_overall_score(

                    sentiment_scores,

                    volume_analysis,

                    sector_impact

                )

            }

        except Exception as e:

            logger.error(f"News analysis error: {str(e)}")

            return None

        

    async def _fetch_all_sources(self, symbol):

        news_api_data = await self._get_news_api_data(symbol)

        finnhub_news = await self._get_finnhub_news(symbol)

        return self._combine_news_sources([news_api_data, finnhub_news]) 

    async def _get_news_api_data(self, symbol):
        try:
            news = self.news_api.get_everything(
                q=symbol,
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            return news['articles']
        except Exception as e:
            logger.error(f"News API error: {str(e)}")
            return []

    async def _get_finnhub_news(self, symbol):
        try:
            end = datetime.now()
            start = end - timedelta(days=7)
            news = self.finnhub_client.company_news(
                symbol, 
                _from=start.strftime('%Y-%m-%d'),
                to=end.strftime('%Y-%m-%d')
            )
            return news
        except Exception as e:
            logger.error(f"Finnhub error: {str(e)}")
            return []

    def _score_articles(self, articles):
        scored = []
        for article in articles:
            sentiment = self.sentiment_model(article['title'])[0]
            score = self._calculate_article_score(sentiment, article)
            scored.append({
                'article': article,
                'score': score
            })
        return scored

    def _calculate_article_score(self, sentiment, article):
        base_score = sentiment['score']
        source_weight = self._get_source_weight(article.get('url', ''))
        time_weight = self._calculate_time_weight(article.get('publishedAt'))
        return base_score * source_weight * time_weight
