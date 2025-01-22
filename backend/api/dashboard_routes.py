from flask import Blueprint, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any

dashboard_bp = Blueprint('dashboard', __name__)

def calculate_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic technical indicators"""
    indicators = {}
    
    # Moving averages
    indicators['SMA_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
    indicators['SMA_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = (100 - (100 / (1 + rs))).iloc[-1]
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    indicators['MACD'] = macd.iloc[-1]
    indicators['Signal_Line'] = signal.iloc[-1]
    
    # Volatility
    indicators['Volatility'] = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized
    
    # Round all values
    return {k: round(float(v), 2) for k, v in indicators.items()}

@dashboard_bp.route('/api/dashboard/<ticker>', methods=['GET'])
def get_dashboard_data(ticker: str) -> Dict[str, Any]:
    """Get comprehensive dashboard data for a stock"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return jsonify({
                'error': 'No data available for this stock'
            }), 404
        
        # Calculate indicators
        indicators = calculate_indicators(data)
        
        # Get company info
        info = stock.info
        
        # Prepare response
        dashboard_data = {
            'ticker': ticker,
            'company_info': {
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            },
            'market_data': {
                'prices': data['Close'].round(2).tolist(),
                'volumes': data['Volume'].tolist(),
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'current_price': round(data['Close'].iloc[-1], 2),
                'price_change': round(data['Close'].iloc[-1] - data['Close'].iloc[-2], 2),
                'price_change_percent': round((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100, 2)
            },
            'technical_indicators': indicators
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        print(f"Error fetching dashboard data for {ticker}: {str(e)}")
        return jsonify({
            'error': 'Failed to fetch dashboard data',
            'details': str(e)
        }), 500