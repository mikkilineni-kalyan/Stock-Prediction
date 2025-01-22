from datetime import datetime, timedelta

def validate_prediction_dates(start_date, end_date, include_today=True):
    current_date = datetime.now().date()
    current_time = datetime.now().time()
    market_open_time = datetime.strptime('09:30', '%H:%M').time()
    market_close_time = datetime.strptime('16:00', '%H:%M').time()
    
    if include_today:
        # For same-day predictions, ensure we have enough data
        if end_date == current_date:
            # Check if market is open
            if market_open_time <= current_time <= market_close_time:
                return {
                    "valid": True,
                    "prediction_type": "intraday",
                    "message": "Providing real-time predictions based on current market data and news"
                }
            else:
                return {
                    "valid": True,
                    "prediction_type": "next_day",
                    "message": "Market is closed. Providing prediction for next market open"
                }
    
        # For future dates
        if end_date > current_date:
            return {
                "valid": True,
                "prediction_type": "future",
                "message": "Providing predictions based on historical patterns and current news analysis"
            }
    
    return {"valid": True, "prediction_type": "historical"} 