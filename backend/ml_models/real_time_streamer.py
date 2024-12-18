import asyncio

import websockets

import json

import logging

import os

from datetime import datetime

from typing import Dict, Set, Any



logger = logging.getLogger(__name__)



class RealTimeStreamer:

    def __init__(self):

        self.connections: Set[websockets.WebSocketServerProtocol] = set()

        self.subscribed_symbols: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}

        self.finnhub_ws = None

        self.alpha_vantage_client = None

        

    async def start(self):

        await self.connect_to_providers()

        await self.start_server()

        

    async def connect_to_providers(self):

        # Connect to Finnhub websocket

        uri = f"wss://ws.finnhub.io?token={os.getenv('FINNHUB_API_KEY')}"

        self.finnhub_ws = await websockets.connect(uri)

        

        # Start processing messages

        asyncio.create_task(self._process_finnhub_messages())

        

    async def _process_finnhub_messages(self):

        try:

            while True:

                message = await self.finnhub_ws.recv()

                data = json.loads(message)

                await self._broadcast_to_subscribers(data)

        except Exception as e:

            logger.error(f"Finnhub websocket error: {str(e)}")

            await self.reconnect()
