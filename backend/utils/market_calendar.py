from datetime import datetime, time, timedelta
from typing import List, Tuple
import pandas_market_calendars as mcal
import pytz

class MarketCalendar:
    def __init__(self):
        self.nyse = mcal.get_calendar('NYSE')
        self.est_tz = pytz.timezone('US/Eastern')
    
    def is_market_open(self, dt: datetime = None) -> bool:
        """Check if market is open at given datetime"""
        if dt is None:
            dt = datetime.now(self.est_tz)
        elif dt.tzinfo is None:
            dt = self.est_tz.localize(dt)
        
        # Check if it's a trading day
        schedule = self.nyse.schedule(
            start_date=dt.date(),
            end_date=dt.date()
        )
        
        if schedule.empty:
            return False
        
        # Check trading hours (9:30 AM - 4:00 PM EST)
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = dt.time()
        
        return (
            not schedule.empty and
            market_open <= current_time <= market_close
        )
    
    def get_next_trading_day(self, dt: datetime = None) -> datetime:
        """Get next trading day"""
        if dt is None:
            dt = datetime.now(self.est_tz)
        elif dt.tzinfo is None:
            dt = self.est_tz.localize(dt)
        
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def get_next_n_trading_days(self, n: int, start_date: datetime = None) -> List[datetime]:
        """Get next n trading days"""
        if start_date is None:
            start_date = datetime.now(self.est_tz)
        elif start_date.tzinfo is None:
            start_date = self.est_tz.localize(start_date)
        
        # Get calendar for next month to be safe
        end_date = start_date + timedelta(days=n*2)
        schedule = self.nyse.schedule(
            start_date=start_date.date(),
            end_date=end_date.date()
        )
        
        trading_days = []
        for date in schedule.index:
            if len(trading_days) >= n:
                break
            trading_days.append(date.to_pydatetime())
        
        return trading_days
    
    def is_trading_day(self, dt: datetime) -> bool:
        """Check if given date is a trading day"""
        if dt.tzinfo is None:
            dt = self.est_tz.localize(dt)
        
        schedule = self.nyse.schedule(
            start_date=dt.date(),
            end_date=dt.date()
        )
        return not schedule.empty
    
    def get_trading_hours(self, dt: datetime) -> Tuple[datetime, datetime]:
        """Get market open and close times for a given day"""
        if dt.tzinfo is None:
            dt = self.est_tz.localize(dt)
        
        schedule = self.nyse.schedule(
            start_date=dt.date(),
            end_date=dt.date()
        )
        
        if schedule.empty:
            return None
        
        market_open = schedule.iloc[0]['market_open'].to_pydatetime()
        market_close = schedule.iloc[0]['market_close'].to_pydatetime()
        
        return market_open, market_close 