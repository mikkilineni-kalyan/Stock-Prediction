from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RealTimeUpdater:
    def __init__(self):
        self.data_aggregator = DataAggregator()
        self.update_interval = 5  # seconds
        self.cache = {}
        self.last_update = {}
        
    async def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        current_time = datetime.now().timestamp()
        
        if (symbol not in self.last_update or 
            current_time - self.last_update.get(symbol, 0) > self.update_interval):
            
            try:
                latest_data = await self.data_aggregator.get_complete_data(symbol)
                self.cache[symbol] = latest_data
                self.last_update[symbol] = current_time
            except Exception as e:
                logger.error(f"Real-time update error: {str(e)}")
                
        return self.cache.get(symbol) 