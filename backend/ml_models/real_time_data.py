import yfinance as yf
import finnhub
import requests
from datetime import datetime, timedelta
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

class RealTimeDataService:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
        self.update_interval = 5  # seconds
        self.cache = {}
        self.last_update = {}
        
    async def get_realtime_data(self, symbol):
        current_time = datetime.now().timestamp()
        
        if (symbol not in self.last_update or 
            current_time - self.last_update.get(symbol, 0) > self.update_interval):
            
            # Parallel data fetching
            tasks = [
                self._get_yahoo_data(symbol),
                self._get_alpha_vantage_data(symbol),
                self._get_finnhub_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks)
            yahoo_data, alpha_vantage_data, finnhub_data = results
            
            self.cache[symbol] = self._combine_data_sources(
                yahoo_data,
                alpha_vantage_data,
                finnhub_data
            )
            self.last_update[symbol] = current_time
            
        return self.cache.get(symbol) 