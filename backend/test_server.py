from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
# Update CORS configuration to handle all responses including errors
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test data with mock sentiment scores
STOCKS = {
    'AAPL': {'name': 'Apple Inc.', 'sentiment_score': 4.2, 'prediction': 'positive'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sentiment_score': 3.8, 'prediction': 'positive'},
    'MSFT': {'name': 'Microsoft Corporation', 'sentiment_score': 4.5, 'prediction': 'positive'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sentiment_score': 3.9, 'prediction': 'neutral'},
    'META': {'name': 'Meta Platforms Inc.', 'sentiment_score': 3.5, 'prediction': 'positive'}
}

@app.route('/api/stocks/search', methods=['GET'])
def search_stocks():
    try:
        logger.debug("Received search request")
        query = request.args.get('q', '').upper()
        logger.debug(f"Search query: {query}")
        
        if not query:
            return jsonify([])
        
        matches = []
        for symbol, data in STOCKS.items():
            if query in symbol or query.lower() in data['name'].lower():
                matches.append({
                    'Symbol': symbol,
                    'Name': data['name'],
                    'SentimentScore': data['sentiment_score'],
                    'Prediction': data['prediction']
                })
        
        logger.debug(f"Matches found: {matches}")
        return jsonify(matches)
    except Exception as e:
        logger.error(f"Error in search_stocks: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks/data/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        logger.debug(f"Fetching data for symbol: {symbol}")
        symbol = symbol.upper()
        
        if symbol not in STOCKS:
            return jsonify({'error': 'Invalid stock symbol'}), 404
            
        try:
            # Get stock data from yfinance
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1y')
            
            if hist.empty:
                return jsonify({'error': 'No data available for this symbol'}), 404
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1]
            price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
            price_change_percent = (price_change / hist['Close'].iloc[-2]) * 100
            
            response_data = {
                'symbol': symbol,
                'name': STOCKS[symbol]['name'],
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_percent': round(price_change_percent, 2),
                'sentiment_score': STOCKS[symbol]['sentiment_score'],
                'prediction': STOCKS[symbol]['prediction'],
                'historical_data': hist['Close'].tail(30).to_list(),
                'last_updated': datetime.now().isoformat()
            }
            
            logger.debug(f"Returning data for {symbol}")
            return jsonify(response_data)
        except Exception as e:
            logger.error(f"Error fetching stock data from yfinance: {str(e)}")
            return jsonify({'error': 'Failed to fetch stock data'}), 500
            
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "message": "Stock Prediction API",
        "version": "1.0"
    })

if __name__ == '__main__':
    logger.info("Starting Flask server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
