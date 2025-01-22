import requests

import pandas as pd

from datetime import datetime

import logging

import os
from typing import Dict, Any, Optional  # Add type hints



logger = logging.getLogger(__name__)



class AlphaVantageClient:

    def __init__(self, api_key='G60ECSGKXY97SY11'):

        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY', 'G60ECSGKXY97SY11')

        self.base_url = 'https://www.alphavantage.co/query'

        

    def get_stock_data(self, symbol):

        try:

            # Get real-time data

            intraday = self._get_intraday(symbol)

            

            # Get technical indicators

            technical = self._get_technical_indicators(symbol)

            

            # Get fundamental data

            fundamental = self._get_fundamental_data(symbol)

            

            return {

                'intraday': intraday,

                'technical': technical,

                'fundamental': fundamental

            }

            

        except Exception as e:

            logger.error(f"Error fetching Alpha Vantage data: {str(e)}")

            return None

            

    def _get_intraday(self, symbol):

        params = {

            'function': 'TIME_SERIES_INTRADAY',

            'symbol': symbol,

            'interval': '5min',

            'apikey': self.api_key

        }

        return self._make_request(params)

        

    def _get_technical_indicators(self, symbol):

        indicators = {}

        

        # RSI

        indicators['RSI'] = self._make_request({

            'function': 'RSI',

            'symbol': symbol,

            'interval': '5min',

            'time_period': '14',

            'series_type': 'close',

            'apikey': self.api_key

        })

        

        # MACD

        indicators['MACD'] = self._make_request({

            'function': 'MACD',

            'symbol': symbol,

            'interval': '5min',

            'series_type': 'close',

            'apikey': self.api_key

        })

        

        # VWAP

        indicators['VWAP'] = self._make_request({

            'function': 'VWAP',

            'symbol': symbol,

            'interval': '5min',

            'apikey': self.api_key

        })

        

        return indicators

        

    def _get_fundamental_data(self, symbol):

        return self._make_request({

            'function': 'OVERVIEW',

            'symbol': symbol,

            'apikey': self.api_key

        })

        

    def _make_request(self, params):

        try:

            response = requests.get(self.base_url, params=params)

            response.raise_for_status()

            return response.json()

        except Exception as e:

            logger.error(f"API request failed: {str(e)}")

            return None 
