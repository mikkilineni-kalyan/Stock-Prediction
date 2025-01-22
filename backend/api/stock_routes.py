from flask import Blueprint, jsonify, request
import yfinance as yf
from typing import List, Dict
import pandas as pd
from flask_cors import cross_origin
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

stock_bp = Blueprint('stock', __name__)

# Default response structure
default_response = {
    'data': {
        'news_sentiment': {
            'score': 2.5,
            'impact': 'NEUTRAL',
            'confidence': 0.5,
            'sources': 0,
            'summary': 'No recent news'
        },
        'prediction': {
            'next_day': 0.0,
            'confidence': 50.0,
            'trend': 'neutral',
            'predicted_return': 0.0
        },
        'indicators': {
            'rsi': [],
            'macd': [],
            'signal': [],
            'sma': [],
            'ema': []
        }
    }
}

def error_response(message: str):
    """Helper function to return error response"""
    logger.error(f"Error response: {message}")
    return jsonify({
        'status': 'error',
        'error': message,
        'data': None
    }), 500

def success_response(data: Dict):
    """Helper function to return success response"""
    return jsonify({
        'status': 'success',
        'error': None,
        'data': data
    })

# Initialize predictor variables
news_predictor = None
advanced_predictor = None

def set_predictors(news_pred, advanced_pred):
    """Set the predictor instances for use in routes"""
    global news_predictor, advanced_predictor
    news_predictor = news_pred
    advanced_predictor = advanced_pred
    logger.info("Predictors initialized successfully")

# Load NASDAQ listed stocks from CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), 'nasdaq-listed.csv')
print(f"Loading NASDAQ data from: {CSV_PATH}")
STOCK_DICT = {}

def load_nasdaq_stocks():
    """Load NASDAQ listed stocks from CSV file"""
    global STOCK_DICT
    try:
        print(f"Loading NASDAQ data from: {CSV_PATH}")
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            STOCK_DICT = dict(zip(df['Symbol'].astype(str), df['Name'].astype(str)))
            print(f"Successfully loaded {len(STOCK_DICT)} stocks")
        else:
            print(f"CSV file not found at {CSV_PATH}, using default stock dictionary")
            # Default stock dictionary for testing
            STOCK_DICT = {
                'AAPL': 'Apple Inc.',
                'GOOGL': 'Alphabet Inc.',
                'MSFT': 'Microsoft Corporation',
                'AMZN': 'Amazon.com Inc.',
                'TSLA': 'Tesla, Inc.',
                'META': 'Meta Platforms Inc.',
                'NVDA': 'NVIDIA Corporation',
                'NFLX': 'Netflix Inc.',
                'CSCO': 'Cisco Systems Inc.',
                'INTC': 'Intel Corporation'
            }
    except Exception as e:
        print(f"Error loading NASDAQ data: {str(e)}")
        # Use default stock dictionary on error
        STOCK_DICT = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla, Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'CSCO': 'Cisco Systems Inc.',
            'INTC': 'Intel Corporation'
        }

# Load stocks when module is imported
load_nasdaq_stocks()

@stock_bp.route('/search', methods=['GET'])
def search_stocks():
    """Search for stocks by symbol or name"""
    try:
        query = request.args.get('q', '').strip()
        logger.info(f"Received search query: {query}")
        
        if not query:
            logger.info("Empty query, returning empty list")
            return jsonify([])
        
        # Debug log the current state of STOCK_DICT
        logger.debug(f"STOCK_DICT contains {len(STOCK_DICT)} items")
        if len(STOCK_DICT) == 0:
            # Try reloading the stock dictionary
            logger.info("STOCK_DICT is empty, attempting to reload")
            load_nasdaq_stocks()
            if len(STOCK_DICT) == 0:
                logger.error("Failed to load stock data")
                return jsonify([])
        
        # Convert query to uppercase for case-insensitive symbol search
        query_upper = query.upper()
        # Convert query to lowercase for case-insensitive name search
        query_lower = query.lower()
        
        logger.info(f"Searching for matches with query: {query}")
        
        matches = []
        # First try exact symbol match
        if query_upper in STOCK_DICT:
            logger.info(f"Found exact symbol match: {query_upper}")
            matches.append({
                'Symbol': query_upper,
                'Name': STOCK_DICT[query_upper],
                'SentimentScore': 0,  # Default value
                'Prediction': 'Loading...'  # Default value
            })
        
        # Then try partial matches if we don't have enough results
        if len(matches) < 10:
            for symbol, name in STOCK_DICT.items():
                if len(matches) >= 10:
                    break
                    
                # Skip if already added
                if symbol in [m['Symbol'] for m in matches]:
                    continue
                    
                # Check for partial symbol or name match
                if (query_upper in symbol) or (query_lower in name.lower()):
                    matches.append({
                        'Symbol': symbol,
                        'Name': name,
                        'SentimentScore': 0,  # Default value
                        'Prediction': 'Loading...'  # Default value
                    })
        
        logger.info(f"Found {len(matches)} matches")
        return jsonify(matches)
        
    except Exception as e:
        logger.error(f"Error in search_stocks: {str(e)}", exc_info=True)
        return jsonify([])

@stock_bp.route('/validate/<ticker>', methods=['GET'])
@cross_origin()
def validate_stock(ticker: str):
    """Validate if a stock exists and get its basic info"""
    try:
        # First check if it's in our NASDAQ dictionary
        if ticker in STOCK_DICT:
            name = STOCK_DICT[ticker]
        else:
            # If not in dictionary, try to get it from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            name = info.get('longName', '')
            if not name:
                return jsonify({"error": "Invalid stock symbol"}), 404
        
        return jsonify({
            "symbol": ticker,
            "name": name,
            "valid": True
        })
        
    except Exception as e:
        print(f"Error validating stock {ticker}: {e}")
        return jsonify({
            "error": f"Error validating stock: {str(e)}",
            "valid": False
        }), 500

@stock_bp.route('/news-prediction/<symbol>', methods=['GET'])
@cross_origin()
def get_news_prediction(symbol):
    """Get stock prediction based on news analysis"""
    try:
        prediction = news_predictor.predict_from_news(symbol)
        return jsonify(prediction)
        
    except Exception as e:
        print(f"Error in news prediction for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@stock_bp.route('/predict/<symbol>', methods=['GET'])
@cross_origin()
def predict_stock(symbol):
    """Get comprehensive stock prediction using technical analysis"""
    try:
        prediction = advanced_predictor.predict_comprehensive(symbol)
        return jsonify(prediction)
        
    except Exception as e:
        print(f"Error in stock prediction for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@stock_bp.route('/test/news/<ticker>', methods=['GET'])
@cross_origin()
def test_news_prediction(ticker):
    """Test route to check news prediction data"""
    try:
        news_pred = news_predictor.predict_from_news(ticker)
        print("Raw news prediction:", news_pred)
        return jsonify({
            'raw_prediction': news_pred,
            'score_type': str(type(news_pred.get('score'))),
            'score_value': news_pred.get('score'),
        })
    except Exception as e:
        print(f"Error in test route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@stock_bp.route('/data/<ticker>')
@cross_origin(origins="*", methods=['GET'], allow_headers=['Content-Type'])
def get_stock_data(ticker):
    """Get stock data and predictions."""
    try:
        period = request.args.get('period', '1y')
        logger.info(f"Getting stock data for {ticker} with period {period}")

        # Validate ticker
        ticker = ticker.upper()
        if not validate_stock(ticker):
            logger.warning(f"Invalid ticker: {ticker}")
            return error_response('Invalid ticker symbol')

        # Get stock data
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return error_response('No historical data available')
                
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return error_response('Error fetching stock data')
            
        # Initialize with default values
        news_pred = {
            'score': 2.5,
            'impact': 'NEUTRAL',
            'confidence': 50.0,
            'sources': 0,
            'summary': 'No recent news',
            'analysis': ['No recent news']
        }
        
        tech_pred = {
            'prediction': 'neutral',
            'confidence': 50.0,
            'next_day': float(hist['Close'].iloc[-1]),
            'predicted_return': 0.0
        }

        # Get news prediction with error handling
        try:
            if news_predictor:
                logger.info(f"Getting news prediction for {ticker}")
                raw_pred = news_predictor.predict_from_news(ticker)
                logger.debug(f"Raw news prediction: {raw_pred}")

                if isinstance(raw_pred, dict):
                    sentiment = raw_pred.get('sentiment', {})
                    if isinstance(sentiment, dict):
                        news_pred = {
                            'score': float(sentiment.get('overall_score', 2.5)),
                            'impact': 'POSITIVE' if sentiment.get('overall_score', 2.5) > 3.0 
                                    else 'NEGATIVE' if sentiment.get('overall_score', 2.5) < 2.0 
                                    else 'NEUTRAL',
                            'confidence': float(sentiment.get('confidence', 0.5)) * 100,
                            'sources': int(sentiment.get('news_count', 0)),
                            'summary': str(raw_pred.get('news', [{}])[0].get('title', 'No recent news')),
                            'analysis': [f"Analyzed {sentiment.get('news_count', 0)} news articles"]
                        }
                        logger.info(f"Processed news prediction: {news_pred}")
                    else:
                        logger.warning(f"Invalid sentiment structure in prediction: {sentiment}")
                else:
                    logger.warning(f"Invalid prediction structure: {raw_pred}")
        except Exception as e:
            logger.error(f"Error in news prediction: {str(e)}", exc_info=True)
            # Keep using default news_pred values initialized earlier

        # Get technical prediction with error handling
        try:
            if advanced_predictor:
                logger.info(f"Getting technical prediction for {ticker}")
                raw_tech_pred = advanced_predictor.predict_comprehensive(ticker)
                logger.debug(f"Raw technical prediction: {raw_tech_pred}")

                if isinstance(raw_tech_pred, dict):
                    tech_pred = {
                        'next_day': float(raw_tech_pred.get('next_day', hist['Close'].iloc[-1])),
                        'confidence': float(raw_tech_pred.get('confidence', 50.0)),
                        'trend': str(raw_tech_pred.get('prediction', 'neutral')),
                        'predicted_return': float(raw_tech_pred.get('predicted_return', 0.0))
                    }
                    logger.info(f"Processed technical prediction: {tech_pred}")
                else:
                    logger.warning(f"Invalid technical prediction structure: {raw_tech_pred}")
        except Exception as e:
            logger.error(f"Error in technical prediction: {str(e)}", exc_info=True)
            # Keep using default tech_pred values initialized earlier

        # Build response with validated data
        try:
            response_data = {
                'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                'prices': [float(x) if pd.notnull(x) else 0.0 for x in hist['Close']],
                'volumes': [float(x) if pd.notnull(x) else 0.0 for x in hist['Volume']],
                'prediction': tech_pred,
                'news_sentiment': news_pred,
                'indicators': {
                    'ma5': hist['Close'].rolling(window=5).mean().fillna(0).tolist(),
                    'ma20': hist['Close'].rolling(window=20).mean().fillna(0).tolist(),
                    'ma50': hist['Close'].rolling(window=50).mean().fillna(0).tolist(),
                    'rsi': [],
                    'bollinger': {
                        'upper': [],
                        'middle': [],
                        'lower': []
                    },
                    'volume': {
                        'raw': [float(x) if pd.notnull(x) else 0.0 for x in hist['Volume']],
                        'sma': hist['Volume'].rolling(window=20).mean().fillna(0).tolist(),
                        'ratio': []
                    }
                }
            }
            
            logger.info(f"Successfully built response for {ticker}")
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error building response: {str(e)}", exc_info=True)
            return error_response('Error building response')

    except Exception as e:
        logger.error(f"Unexpected error in get_stock_data: {str(e)}", exc_info=True)
        return error_response(str(e))