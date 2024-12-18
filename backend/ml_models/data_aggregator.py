import logging
from typing import Dict, Any, Optional
from .alpha_vantage_client import AlphaVantageClient
from .news_analyzer import AdvancedNewsAnalyzer
import yfinance as yf
import finnhub

logger = logging.getLogger(__name__)

class DataAggregator:
    def __init__(self):
        self.sources = {
            'alpha_vantage': AlphaVantageClient(),
            'finnhub': finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY')),
            'news_api': AdvancedNewsAnalyzer(),
            'yahoo': None  # Initialize when needed
        }
        
    async def get_complete_data(self, symbol: str) -> Dict[str, Any]:
        try:
            # Get data from all sources in parallel
            alpha_vantage_data = self.sources['alpha_vantage'].get_stock_data(symbol)
            finnhub_data = await self._get_finnhub_data(symbol)
            news_data = await self.sources['news_api'].analyze_impact(symbol)
            yahoo_data = self._get_yahoo_data(symbol)
            
            return self._combine_all_data(
                alpha_vantage_data,
                finnhub_data,
                news_data,
                yahoo_data
            )
            
        except Exception as e:
            logger.error(f"Data aggregation error: {str(e)}")
            return None 