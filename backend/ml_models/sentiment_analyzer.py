from transformers import pipeline

import newsapi

import finnhub

import pandas as pd

import numpy as np

from datetime import datetime, timedelta

import logging

import os


import aiohttp
from typing import List, Dict, Optional



logger = logging.getLogger(__name__)



class SentimentAnalyzer:

    def __init__(self):

        self.news_api = newsapi.NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

        self.finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

        self.sentiment_model = pipeline("sentiment-analysis")

        self.source_weights = {

            'reuters.com': 1.5,

            'bloomberg.com': 1.5,

            'wsj.com': 1.4,

            'ft.com': 1.4,

            'cnbc.com': 1.2,

            'seekingalpha.com': 1.2,

            'fool.com': 1.0,

            'marketwatch.com': 1.1

        }

        

    async def analyze_stock_sentiment(self, symbol, company_name):

        try:

            # Fetch news from multiple sources

            news_data = await self._fetch_all_news_sources(symbol, company_name)

            

            # Analyze sentiment for each article

            analyzed_news = self._analyze_articles(news_data)

            

            # Calculate impact scores

            impact_scores = self._calculate_impact_scores(analyzed_news)

            

            # Get historical correlation

            historical_impact = self._analyze_historical_correlation(symbol, impact_scores)

            

            return {

                'overall_sentiment': self._calculate_weighted_sentiment(analyzed_news),

                'sentiment_breakdown': self._get_sentiment_breakdown(analyzed_news),

                'impact_prediction': self._predict_price_impact(historical_impact),

                'key_articles': self._get_key_articles(analyzed_news),

                'source_reliability': self._calculate_source_reliability(analyzed_news)

            }

            

        except Exception as e:

            logger.error(f"Sentiment analysis error for {symbol}: {str(e)}")

            return None

            

    async def _fetch_all_news_sources(self, symbol, company_name):

        news_articles = []

        

        # News API

        try:

            news_api_articles = await self._get_news_api_articles(symbol, company_name)

            news_articles.extend(news_api_articles)

        except Exception as e:

            logger.error(f"News API error: {str(e)}")

            

        # Finnhub

        try:

            finnhub_articles = await self._get_finnhub_articles(symbol)

            news_articles.extend(finnhub_articles)

        except Exception as e:

            logger.error(f"Finnhub error: {str(e)}")

            

        return news_articles

        

    def _analyze_articles(self, articles):

        analyzed = []

        for article in articles:

            try:

                # Get base sentiment

                sentiment = self._get_sentiment(article['title'] + " " + article['description'])

                

                # Calculate source reliability

                source_weight = self._get_source_weight(article['url'])

                

                # Calculate time decay

                time_weight = self._calculate_time_weight(article['publishedAt'])

                

                # Final weighted score

                impact_score = sentiment['score'] * source_weight * time_weight

                

                analyzed.append({

                    'article': article,

                    'sentiment': sentiment,

                    'impact_score': impact_score,

                    'source_weight': source_weight,

                    'time_weight': time_weight

                })

                

            except Exception as e:

                logger.error(f"Article analysis error: {str(e)}")

                continue

                

        return analyzed

        

    def _get_source_weight(self, url):

        for domain, weight in self.source_weights.items():

            if domain in url:

                return weight

        return 1.0

        

    def _calculate_time_weight(self, published_at):

        if not published_at:

            return 0.5

            

        pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))

        age_hours = (datetime.now() - pub_time).total_seconds() / 3600

        

        # Exponential decay with 24-hour half-life

        return np.exp(-age_hours / 24)

        

    def _calculate_weighted_sentiment(self, analyzed_news):

        if not analyzed_news:

            return 0.5

            

        weights = [article['impact_score'] for article in analyzed_news]

        sentiments = [article['sentiment']['score'] for article in analyzed_news]

        

        return np.average(sentiments, weights=weights)

        

    def _predict_price_impact(self, sentiment_score):

        # Convert sentiment to price impact

        # Typically, extreme sentiments have larger impact

        impact = (sentiment_score - 0.5) * 2  # Scale to [-1, 1]

        return {

            'direction': 'positive' if impact > 0 else 'negative',

            'magnitude': abs(impact),

            'confidence': min(abs(impact) * 2, 1)  # Higher for extreme sentiments

        }














