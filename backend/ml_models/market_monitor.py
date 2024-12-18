import finnhub

import yfinance as yf

from datetime import datetime, timedelta

import pandas as pd

import numpy as np

import logging



logger = logging.getLogger(__name__)



class MarketMonitor:

    def __init__(self):

        self.finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

        

    def get_market_status(self):

        return {

            'market_hours': self._check_market_hours(),

            'market_sentiment': self._get_market_sentiment(),

            'sector_performance': self._get_sector_performance(),

            'volatility_index': self._get_vix_data()

        }

            

    def _check_market_hours(self):

        now = datetime.now()

        market_open = datetime.now().replace(hour=9, minute=30)

        market_close = datetime.now().replace(hour=16, minute=0)

        return {

            'is_market_open': market_open <= now <= market_close,

            'next_open': self._get_next_market_open(),

            'next_close': self._get_next_market_close()

        } 

    def _get_market_sentiment(self):

        try:

            vix = yf.Ticker('^VIX')

            vix_data = vix.history(period='1d')

            return {

                'vix': float(vix_data['Close'].iloc[-1]),

                'fear_greed': self._calculate_fear_greed_index()

            }

        except Exception as e:

            logger.error(f"Market sentiment error: {str(e)}")

            return None

    def _get_sector_performance(self):

        try:

            sectors = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE']

            performance = {}

            for sector in sectors:

                ticker = yf.Ticker(sector)

                hist = ticker.history(period='5d')

                performance[sector] = float(hist['Close'].pct_change().mean() * 100)

            return performance

        except Exception as e:

            logger.error(f"Sector performance error: {str(e)}")

            return None

    def _analyze_market_sentiment(self):
        try:
            # Get VIX data
            vix = self._get_vix_data()
            
            # Get market breadth
            breadth = self._calculate_market_breadth()
            
            # Get sector rotation
            sector_rotation = self._analyze_sector_rotation()
            
            # Get options sentiment
            options_sentiment = self._analyze_options_sentiment()
            
            return {
                'vix_analysis': vix,
                'market_breadth': breadth,
                'sector_rotation': sector_rotation,
                'options_sentiment': options_sentiment,
                'overall_sentiment': self._calculate_overall_sentiment(
                    vix, breadth, sector_rotation, options_sentiment
                )
            }
        except Exception as e:
            logger.error(f"Market sentiment analysis error: {str(e)}")
            return None

    def _calculate_market_breadth(self):
        try:
            # Get S&P 500 components
            sp500 = self._get_sp500_components()
            advancing = 0
            declining = 0
            
            for symbol in sp500:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    if hist['Close'].iloc[-1] > hist['Open'].iloc[-1]:
                        advancing += 1
                    else:
                        declining += 1
                        
            return {
                'advancing': advancing,
                'declining': declining,
                'ratio': advancing / (advancing + declining) if advancing + declining > 0 else 0.5
            }
        except Exception as e:
            logger.error(f"Market breadth calculation error: {str(e)}")
            return None

    def _analyze_sector_rotation(self):
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial'
        }
        
        performance = {}
        for symbol, sector in sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                performance[sector] = float(hist['Close'].pct_change().mean() * 100)
            except Exception as e:
                logger.error(f"Sector analysis error for {sector}: {str(e)}")
                
        return performance

    def _analyze_options_sentiment(self):
        try:
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(period='1d')
            put_call_ratio = self._get_put_call_ratio()
            
            return {
                'vix_level': float(vix_data['Close'].iloc[-1]),
                'put_call_ratio': put_call_ratio,
                'implied_volatility': self._get_implied_volatility()
            }
        except Exception as e:
            logger.error(f"Options sentiment analysis error: {str(e)}")
            return None

    def _get_put_call_ratio(self):
        try:
            # Implement put/call ratio calculation using options data
            return 1.0  # Placeholder
        except Exception as e:
            logger.error(f"Put/call ratio calculation error: {str(e)}")
            return None
