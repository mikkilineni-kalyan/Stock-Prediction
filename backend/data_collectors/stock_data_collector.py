import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from ..config.database import session, Stock, StockPrice, NewsArticle
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class StockDataCollector:
    def __init__(self):
        self.session = session
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    async def collect_stock_data(self, symbol: str, period: str = "1d") -> Optional[Dict]:
        """
        Collect stock data from multiple sources and store in database
        Returns the collected data if successful, None otherwise
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    return cache_data

            # Get stock data from yfinance
            stock_data = await self._get_yfinance_data(symbol, period)
            if stock_data is None:
                return None

            # Calculate technical indicators
            stock_data = self._calculate_technical_indicators(stock_data)

            # Store in database
            await self._store_stock_data(symbol, stock_data)

            # Update cache
            self.cache[cache_key] = (time.time(), stock_data)

            return stock_data

        except Exception as e:
            logger.error(f"Error collecting stock data for {symbol}: {str(e)}")
            return None

    async def _get_yfinance_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get stock data from yfinance with retry mechanism"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                
                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                    
                return data

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"All attempts failed for {symbol}")
                    return None

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the stock data"""
        try:
            # Simple Moving Averages
            data['sma_20'] = data['Close'].rolling(window=20).mean()
            data['sma_50'] = data['Close'].rolling(window=50).mean()
            data['sma_200'] = data['Close'].rolling(window=200).mean()

            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2

            return data

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return data

    async def _store_stock_data(self, symbol: str, data: pd.DataFrame):
        """Store stock data in the database"""
        try:
            # Get or create stock record
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                stock = Stock(symbol=symbol)
                self.session.add(stock)
                self.session.commit()

            # Store price data
            for index, row in data.iterrows():
                price = StockPrice(
                    stock_id=stock.id,
                    timestamp=index,
                    open_price=row['Open'],
                    high_price=row['High'],
                    low_price=row['Low'],
                    close_price=row['Close'],
                    adjusted_close=row['Close'],
                    volume=row['Volume'],
                    sma_20=row.get('sma_20'),
                    sma_50=row.get('sma_50'),
                    sma_200=row.get('sma_200'),
                    rsi_14=row.get('rsi_14'),
                    macd=row.get('macd')
                )
                self.session.add(price)

            self.session.commit()

        except SQLAlchemyError as e:
            logger.error(f"Database error storing stock data: {str(e)}")
            self.session.rollback()
        except Exception as e:
            logger.error(f"Error storing stock data: {str(e)}")
            self.session.rollback()

    async def collect_multiple_stocks(self, symbols: List[str], period: str = "1d"):
        """Collect data for multiple stocks concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.collect_stock_data(symbol, period) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful = 0
            failed = 0
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to collect data for {symbol}: {str(result)}")
                    failed += 1
                elif result is None:
                    logger.warning(f"No data collected for {symbol}")
                    failed += 1
                else:
                    successful += 1
                    
            logger.info(f"Data collection completed. Successful: {successful}, Failed: {failed}")
            
            return successful, failed
