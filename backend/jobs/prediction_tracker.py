from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionTracker:
    def __init__(self, historical_tracker, data_fetcher):
        self.historical_tracker = historical_tracker
        self.data_fetcher = data_fetcher
        self.scheduler = BackgroundScheduler()
        self.setup_jobs()
    
    def setup_jobs(self):
        """Setup scheduled jobs"""
        # Update predictions after market close
        self.scheduler.add_job(
            self.update_predictions,
            CronTrigger(hour=16, minute=30),  # 4:30 PM EST
            id='update_predictions'
        )
        
        # Clean old predictions weekly
        self.scheduler.add_job(
            self.clean_old_predictions,
            CronTrigger(day_of_week='sun', hour=0),
            id='clean_predictions'
        )
    
    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        logger.info("Prediction tracker scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("Prediction tracker scheduler stopped")
    
    def update_predictions(self):
        """Update all predictions from today with actual prices"""
        try:
            today = datetime.now().date()
            predictions = self.historical_tracker.get_todays_predictions()
            
            for pred in predictions:
                try:
                    # Get closing price
                    stock_data = self.data_fetcher.get_stock_data(
                        pred.ticker,
                        period='1d'
                    )
                    if not stock_data.empty:
                        closing_price = stock_data['Close'][-1]
                        
                        # Update prediction with actual price
                        self.historical_tracker.update_actual_price(
                            pred.ticker,
                            pred.timestamp,
                            closing_price
                        )
                        logger.info(f"Updated prediction for {pred.ticker}")
                except Exception as e:
                    logger.error(f"Error updating prediction for {pred.ticker}: {str(e)}")
            
            logger.info(f"Successfully updated {len(predictions)} predictions")
        except Exception as e:
            logger.error(f"Error in update_predictions: {str(e)}")
    
    def clean_old_predictions(self):
        """Remove predictions older than 90 days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=90)
            self.historical_tracker.remove_old_predictions(cutoff_date)
            logger.info("Cleaned old predictions")
        except Exception as e:
            logger.error(f"Error cleaning old predictions: {str(e)}") 