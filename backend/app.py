from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
import json
import logging
from ml_models.news_predictor import NewsPredictor
from ml_models.advanced_predictor import AdvancedPredictor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting application initialization...")

app = Flask(__name__)

# Configure CORS to accept requests from any frontend port
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Register blueprints
from api.stock_routes import stock_bp
app.register_blueprint(stock_bp, url_prefix='/api/stocks')
logger.info("Registered stock routes blueprint")

# Initialize predictors
try:
    news_predictor = NewsPredictor()
    advanced_predictor = AdvancedPredictor()
    logger.info("Successfully initialized predictors")
except Exception as e:
    logger.error(f"Error initializing predictors: {str(e)}", exc_info=True)
    news_predictor = None
    advanced_predictor = None

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory: {current_dir}")

# Load NASDAQ listings
try:
    nasdaq_file = os.path.join(current_dir, 'api', 'nasdaq-listed.csv')
    logger.info(f"Attempting to load NASDAQ data from: {nasdaq_file}")
    nasdaq_df = pd.read_csv(nasdaq_file)
    # Convert column names to match frontend expectations
    nasdaq_df = nasdaq_df.rename(columns={'Symbol': 'Symbol', 'Name': 'Name'})
    logger.info(f"Successfully loaded NASDAQ data with {len(nasdaq_df)} rows")
except FileNotFoundError as e:
    logger.error(f"Error loading NASDAQ stocks: {str(e)}")
    nasdaq_df = pd.DataFrame(columns=['Symbol', 'Name'])
    logger.info("Created empty DataFrame as fallback")

# Register blueprints and set predictors
from api.stock_routes import set_predictors
set_predictors(news_predictor, advanced_predictor)

@app.before_request
def log_request_info():
    logger.debug('Headers: %s', request.headers)
    logger.debug('Body: %s', request.get_data())

@app.after_request
def after_request(response):
    logger.debug('Response: %s', response.get_data())
    return response

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical indicators without ta library."""
    try:
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Initialize all fields with None arrays
        indicators = ['MA5', 'MA20', 'MA50', 'BB_middle', 'BB_upper', 'BB_lower', 'RSI', 'Volume_SMA', 'Volume_ratio']
        for indicator in indicators:
            df[indicator] = [None] * len(df)
        
        # Moving Averages
        if len(df) >= 5:
            df['MA5'] = df['Close'].rolling(window=5).mean()
        if len(df) >= 20:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        if len(df) >= 50:
            df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands (requires 20 periods)
        if len(df) >= 20:
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * std
            df['BB_lower'] = df['BB_middle'] - 2 * std
        
        # RSI (requires 14 periods)
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume indicators (requires 20 periods)
        if len(df) >= 20:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        # Return the DataFrame with initialized indicators even if calculation fails
        return df

def prepare_prediction_data(data: pd.DataFrame) -> tuple:
    """Prepare data for prediction model."""
    try:
        # Create features
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Returns_vol'] = df['Returns'].rolling(window=5).std()
        df['Price_vol'] = df['Close'].rolling(window=5).std()
        
        # Technical indicator features
        for col in ['RSI']:
            if col in df.columns:
                df[f'{col}_diff'] = df[col].diff()
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)
        
        # Create target (next day's return)
        df['Target'] = df['Returns'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features for prediction
        feature_columns = [col for col in df.columns if col not in ['Target', 'Date', 'Returns']]
        X = df[feature_columns].values
        y = df['Target'].values
        
        return X, y, df
    except Exception as e:
        logger.error(f"Error preparing prediction data: {str(e)}")
        return None, None, None

def predict_next_day(data: pd.DataFrame) -> dict:
    """Enhanced prediction model using Random Forest."""
    try:
        # Prepare data
        X, y, df = prepare_prediction_data(data)
        if X is None or y is None:
            raise ValueError("Failed to prepare prediction data")
        
        # Train model on recent data (last 252 trading days if available)
        train_size = min(len(X), 252)
        X_train = X[-train_size:-1]
        y_train = y[-train_size:-1]
        
        # Initialize and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict next day's return
        next_day_features = X[-1:]
        predicted_return = model.predict(next_day_features)[0]
        
        # Calculate prediction
        last_price = float(data['Close'].iloc[-1])
        next_day_price = last_price * (1 + predicted_return)
        
        # Calculate confidence based on feature importance
        confidence = min(0.8, 0.5 + model.score(X_train, y_train))
        
        # Determine trend
        trend = 'neutral'
        if predicted_return > 0.01:
            trend = 'bullish'
        elif predicted_return < -0.01:
            trend = 'bearish'
        
        return {
            'next_day': round(float(next_day_price), 2),
            'confidence': round(float(confidence), 2),
            'trend': trend,
            'predicted_return': round(float(predicted_return * 100), 2)
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {
            'next_day': float(data['Close'].iloc[-1]),
            'confidence': 0.5,
            'trend': 'neutral',
            'predicted_return': 0.0
        }

def prepare_stock_data(ticker: str, period: str = '1y') -> dict:
    """Fetch and prepare stock data for analysis and prediction."""
    try:
        logger.info(f"Fetching data for {ticker} with period {period}")
        
        # Initialize stock
        stock = yf.Ticker(ticker)
        
        # First attempt with primary intervals
        if period == '1d':
            hist = stock.history(period='1d', interval='5m')
            if len(hist) < 5:  # If not enough data, try 2m interval
                hist = stock.history(period='1d', interval='2m')
            if len(hist) < 5:  # If still not enough, try 1m as last resort
                hist = stock.history(period='1d', interval='1m')
        elif period == '5d':
            hist = stock.history(period='5d', interval='15m')
            if len(hist) < 5:  # If not enough data, try 5m interval
                hist = stock.history(period='5d', interval='5m')
        else:
            # For other periods, use standard intervals
            period_intervals = {
                '1mo': ('1mo', '1h'),
                '3mo': ('3mo', '1h'),
                '6mo': ('6mo', '1d'),
                '1y': ('1y', '1d'),
                '2y': ('2y', '1d'),
                '5y': ('5y', '1d'),
                'max': ('max', '1d')
            }
            p, interval = period_intervals.get(period, ('1y', '1d'))
            hist = stock.history(period=p, interval=interval)
        
        if hist.empty:
            logger.error(f"No data found for {ticker} with period {period}")
            # Fallback to daily data for very short periods
            if period in ['1d', '5d']:
                fallback_days = 5 if period == '5d' else 2
                hist = stock.history(period=f'{fallback_days}d', interval='1d')
                if hist.empty:
                    raise ValueError(f"No data found for ticker {ticker} for period {period}")
            else:
                raise ValueError(f"No data found for ticker {ticker} for period {period}")
            
        logger.info(f"Fetched {len(hist)} data points")
        
        try:
            # Calculate technical indicators
            data = calculate_technical_indicators(hist)
            logger.info("Successfully calculated technical indicators")
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            # If technical indicators fail, use raw historical data
            data = hist.copy()
            # Initialize indicator fields with None arrays
            indicators = ['MA5', 'MA20', 'MA50', 'BB_middle', 'BB_upper', 'BB_lower', 'RSI', 'Volume_SMA', 'Volume_ratio']
            for indicator in indicators:
                data[indicator] = [None] * len(data)
        
        # Prepare prediction
        prediction = {
            'next_day': float(hist['Close'].iloc[-1]),
            'confidence': 0.5,
            'trend': 'neutral',
            'predicted_return': 0.0
        }
        
        # Convert NaN values to None for JSON serialization
        def safe_convert(x):
            return None if pd.isna(x) else float(x)
        
        # Format datetime index based on period
        if period in ['1d', '5d']:
            date_format = '%Y-%m-%d %H:%M'
        else:
            date_format = '%Y-%m-%d'
            
        # Prepare response data
        response_data = {
            'dates': data.index.strftime(date_format).tolist(),
            'prices': [safe_convert(x) for x in data['Close']],
            'volumes': [int(x) for x in data['Volume']],
            'prediction': prediction,
            'indicators': {
                'ma5': [safe_convert(x) for x in data['MA5']],
                'ma20': [safe_convert(x) for x in data['MA20']],
                'ma50': [safe_convert(x) for x in data['MA50']],
                'rsi': [safe_convert(x) for x in data['RSI']],
                'bollinger': {
                    'upper': [safe_convert(x) for x in data['BB_upper']],
                    'middle': [safe_convert(x) for x in data['BB_middle']],
                    'lower': [safe_convert(x) for x in data['BB_lower']]
                },
                'volume': {
                    'raw': [int(x) for x in data['Volume']],
                    'sma': [safe_convert(x) for x in data['Volume_SMA']],
                    'ratio': [safe_convert(x) for x in data['Volume_ratio']]
                }
            }
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error preparing stock data: {str(e)}")
        raise Exception(f"Error preparing stock data: {str(e)}")

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "Stock Prediction API"})

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/search/stocks')
def search_stocks():
    logger.info("Handling request to /api/search/stocks")
    query = request.args.get('q', '').strip()
    logger.info(f"Search query: {query}")
    
    if not query:
        logger.info("Empty query, returning empty list")
        return jsonify([])
    
    try:
        # Convert query to uppercase for case-insensitive search
        query_upper = query.upper()
        
        # First try exact symbol match
        exact_matches = nasdaq_df[nasdaq_df['Symbol'] == query_upper]
        
        if not exact_matches.empty:
            results = exact_matches[['Symbol', 'Name']].to_dict('records')
            logger.info(f"Found exact match for query: {query}")
            return jsonify(results)
        
        # Then try partial matches
        symbol_matches = nasdaq_df[nasdaq_df['Symbol'].str.contains(query_upper, case=False, na=False)]
        name_matches = nasdaq_df[nasdaq_df['Name'].str.contains(query, case=False, na=False)]
        
        # Combine matches and remove duplicates
        matches = pd.concat([symbol_matches, name_matches]).drop_duplicates(subset=['Symbol'])
        
        # Sort by symbol length and then alphabetically
        matches['SymLen'] = matches['Symbol'].str.len()
        matches = matches.sort_values(['SymLen', 'Symbol']).head(10)
        
        # Convert to list of dictionaries
        results = matches[['Symbol', 'Name']].to_dict('records')
        logger.info(f"Found {len(results)} matches for query: {query}")
        
        if not results:
            logger.info("No matches found, returning empty list")
            return jsonify([])
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in search_stocks: {str(e)}")
        return jsonify({"error": "Failed to search stocks"}), 500

@app.route('/api/stocks/news/<ticker>')
def get_stock_news(ticker):
    """Get news analysis and predictions for a stock"""
    try:
        # Get impact prediction
        impact = news_predictor.predict_impact(ticker)
        
        # Get historical correlation
        correlation = news_predictor.analyze_historical_correlation(ticker)
        
        return jsonify({
            'status': 'success',
            'data': {
                'impact_prediction': impact,
                'historical_correlation': correlation
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/stocks/data/<ticker>')
def get_stock_data(ticker):
    """Get stock data and predictions."""
    try:
        period = request.args.get('period', '1y')
        
        # Get technical analysis data
        data = prepare_stock_data(ticker, period)
        
        # Get news impact prediction
        impact = news_predictor.predict_impact(ticker)
        
        # Combine technical and news analysis for final prediction
        confidence = (data['prediction']['confidence'] + impact['confidence']) / 2
        
        # Adjust prediction based on news sentiment
        if impact['score'] >= 3:
            if impact['impact'] == 'positive':
                predicted_return = data['prediction']['predicted_return'] * (1 + impact['score']/10)
            else:
                predicted_return = data['prediction']['predicted_return'] * (1 - impact['score']/10)
        else:
            predicted_return = data['prediction']['predicted_return']
        
        # Update prediction with news analysis
        data['prediction']['confidence'] = confidence
        data['prediction']['predicted_return'] = predicted_return
        data['prediction']['news_impact'] = impact
        
        return jsonify({
            'status': 'success',
            'data': data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000, threaded=True)
