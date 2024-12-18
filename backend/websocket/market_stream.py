import asyncio
import websockets
import json
from typing import Dict, Set
from ..market_analyzer.market_data import MarketDataAnalyzer
from ..market_analyzer.pattern_recognition import AdvancedPatternRecognition
from ..news_analyzer.news_sources import NewsAnalyzer
from flask_socketio import SocketIO, emit
import yfinance as yf
from threading import Thread
import time

class MarketStream:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.active_tickers = set()
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._stream_data)
            self.thread.daemon = True  # Make thread daemon so it exits when main thread exits
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def add_ticker(self, ticker: str):
        self.active_tickers.add(ticker.upper())
        print(f"Added ticker: {ticker}")

    def remove_ticker(self, ticker: str):
        self.active_tickers.discard(ticker.upper())
        print(f"Removed ticker: {ticker}")

    def _stream_data(self):
        while self.running:
            for ticker in list(self.active_tickers):
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period='1d', interval='1m').iloc[-1]
                    
                    update = {
                        'ticker': ticker,
                        'price': float(data['Close']),
                        'volume': int(data['Volume']),
                        'timestamp': data.name.isoformat()
                    }
                    
                    self.socketio.emit('market_update', update)
                    print(f"Sent update for {ticker}: {update}")
                except Exception as e:
                    print(f"Error streaming {ticker}: {str(e)}")
            
            time.sleep(60)  # Update every minute

def run_websocket_server():
    server = MarketStream()
    asyncio.run(server.serve()) 