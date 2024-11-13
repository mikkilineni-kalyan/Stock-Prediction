from flask import Flask, jsonify
from flask_cors import CORS
from backend.ml_models.model import StockPredictor
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

app = Flask(__name__)
CORS(app)

predictor = StockPredictor()

# Add this function to convert numpy types to Python native types
def convert_to_json_serializable(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj

@app.route('/api/predict/<symbol>')
def predict(symbol):
    try:
        # Get prediction data
        result = predictor.predict(symbol.upper())
        
        # Get historical metrics
        historical_metrics = predictor.get_historical_metrics(symbol.upper())
        
        # Combine the data
        result['historicalMetrics'] = historical_metrics
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in API: {str(e)}")  # Add this for debugging
        return jsonify({'message': str(e), 'status': 'error'}), 500

@app.route('/api/news/<symbol>')
def get_news(symbol):
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        formatted_news = [
            {
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'url': item.get('link', ''),
                'published': item.get('providerPublishTime', 0)
            } 
            for item in (news[:5] if news else [])
        ]
        return jsonify({
            'news': formatted_news
        })
    except Exception as e:
        return jsonify({'message': str(e), 'status': 'error'}), 500

@app.route('/api/compare/<symbol1>/<symbol2>')
def compare_stocks(symbol1, symbol2):
    try:
        # Get historical data for both stocks
        stock1 = yf.Ticker(symbol1)
        stock2 = yf.Ticker(symbol2)
        
        # Get last year's data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        hist1 = stock1.history(start=start_date, end=end_date)
        hist2 = stock2.history(start=start_date, end=end_date)
        
        # Normalize prices for comparison
        prices1 = (hist1['Close'] / hist1['Close'].iloc[0]) * 100
        prices2 = (hist2['Close'] / hist2['Close'].iloc[0]) * 100
        
        return jsonify({
            'dates': hist1.index.strftime('%Y-%m-%d').tolist(),
            'prices1': prices1.tolist(),
            'prices2': prices2.tolist(),
            'symbol1': symbol1,
            'symbol2': symbol2
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 