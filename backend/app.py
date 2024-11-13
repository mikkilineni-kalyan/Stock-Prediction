from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import logging
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_csv_file():
    # Get the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Script directory: {script_dir}")
    
    # Define possible CSV paths relative to the script directory
    possible_paths = [
        os.path.join(script_dir, 'nasdaq-listed.csv'),
        os.path.join(script_dir, 'api', 'stock_listings.csv')
    ]
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found CSV at: {path}")
            return path
            
    logger.error("CSV file not found in any of these locations:")
    for path in possible_paths:
        logger.error(f"- {path}")
    return None

# Load the NASDAQ listings
try:
    csv_path = find_csv_file()
    if csv_path is None:
        raise FileNotFoundError("Could not find CSV file in any location")
        
    logger.info(f"Attempting to load CSV from: {csv_path}")
    STOCK_DATA = pd.read_csv(csv_path)
    
    # Clean up any potential whitespace in column names
    STOCK_DATA.columns = STOCK_DATA.columns.str.strip()
    
    logger.info("CSV file loaded successfully")
    logger.info(f"CSV Columns: {STOCK_DATA.columns.tolist()}")
    logger.info(f"Number of rows: {len(STOCK_DATA)}")
    
except Exception as e:
    logger.error(f"Error loading stock listings: {str(e)}", exc_info=True)
    STOCK_DATA = pd.DataFrame()

@app.route('/api/search-stocks', methods=['GET'])
def search_stocks():
    query = request.args.get('q', '').strip()
    logger.info(f"Received search query: {query}")
    
    try:
        if STOCK_DATA.empty:
            logger.error("STOCK_DATA is empty!")
            return jsonify([])
        
        if not query:
            # Return first 10 stocks if no query
            results = STOCK_DATA.head(10)
        else:
            # Create a mapping of common company names to their symbols
            company_aliases = {
                'GOOGLE': ['GOOGL', 'GOOG'],
                'ALPHABET': ['GOOGL', 'GOOG'],
                'META': ['META', 'FB'],
                'FACEBOOK': ['META', 'FB'],
                'BERKSHIRE': ['BRK.B', 'BRK-B', 'BRK/B'],
                'MICROSOFT': ['MSFT'],
                'APPLE': ['AAPL'],
                'AMAZON': ['AMZN'],
                'TESLA': ['TSLA']
            }
            
            query_upper = query.upper()
            
            # Check if query matches any company alias
            matching_symbols = []
            for company, symbols in company_aliases.items():
                if query_upper in company or company in query_upper:
                    matching_symbols.extend(symbols)
            
            # Search in Symbol and Name columns, now case-insensitive
            mask = (
                STOCK_DATA['Symbol'].astype(str).str.contains(query, case=False, na=False) |
                STOCK_DATA['Name'].astype(str).str.contains(query, case=False, na=False) |
                STOCK_DATA['Symbol'].isin(matching_symbols)
            )
            results = STOCK_DATA[mask].head(10)
            logger.info(f"Found {len(results)} matches for query: {query}")

        # Convert to list of dictionaries with more detailed information
        suggestions = []
        for _, row in results.iterrows():
            suggestion = {
                'symbol': str(row['Symbol']),
                'name': str(row['Name']),
                'type': 'NASDAQ Stock',
                'lastSale': str(row['Last Sale']).replace('$', ''),
                'netChange': str(row['Net Change']),
                'percentChange': str(row['% Change']),
                'marketCap': str(row['Market Cap']),
                'sector': str(row['Sector']),
                'industry': str(row['Industry']),
                'country': str(row['Country']),
                'volume': str(row['Volume'])
            }
            suggestions.append(suggestion)
            logger.info(f"Added suggestion: {suggestion}")
        
        return jsonify(suggestions)
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify([])

@app.route('/api/predict/<symbol>', methods=['POST'])
def predict_stock(symbol):
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in prediction request")
            return jsonify({'error': 'No data provided'}), 400

        days = int(data.get('days', 7))
        start_datetime_str = data.get('startDateTime')
        
        if not start_datetime_str:
            logger.error("Start date/time is missing in prediction request")
            return jsonify({'error': 'Start date/time is required'}), 400

        logger.info(f"Predicting {symbol} for {days} days from {start_datetime_str}")

        # Parse start datetime
        try:
            start_datetime = datetime.fromisoformat(start_datetime_str.replace('Z', '+00:00'))
            est = pytz.timezone('US/Eastern')
            start_datetime = start_datetime.astimezone(est)
        except ValueError as e:
            logger.error(f"Invalid date/time format: {e}")
            return jsonify({'error': 'Invalid date/time format'}), 400

        # Fetch historical data
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1y')
            
            if hist.empty:
                logger.error(f"No data available for symbol {symbol}")
                return jsonify({'error': f'No data available for symbol {symbol}'}), 404

            # Calculate predictions
            current_price = float(hist['Close'].iloc[-1])
            logger.info(f"Current price for {symbol}: {current_price}")
            
            # Calculate daily returns and volatility
            returns = hist['Close'].pct_change().dropna()
            daily_volatility = returns.std()
            annual_volatility = daily_volatility * (252 ** 0.5)  # Annualized volatility
            
            # Generate predictions using historical volatility
            predictions = [current_price]
            dates = [start_datetime.strftime('%Y-%m-%d')]
            
            for i in range(days):
                last_price = predictions[-1]
                # Use historical mean return and add volatility
                avg_return = returns.mean()
                next_return = np.random.normal(avg_return, daily_volatility)
                next_price = last_price * (1 + next_return)
                predictions.append(float(next_price))
                
                next_date = start_datetime + timedelta(days=i+1)
                dates.append(next_date.strftime('%Y-%m-%d'))

            # Calculate confidence intervals
            confidence_interval = 1.96 * daily_volatility  # 95% confidence interval
            upper_bound = [price * (1 + confidence_interval) for price in predictions]
            lower_bound = [price * (1 - confidence_interval) for price in predictions]

            response_data = {
                'symbol': symbol,
                'currentPrice': current_price,
                'predictions': predictions,
                'dates': dates,
                'confidenceIntervals': {
                    'upper': upper_bound,
                    'lower': lower_bound
                },
                'metrics': {
                    'volatility': float(annual_volatility * 100),  # Convert to percentage
                    'predictedChange': float((predictions[-1]/current_price - 1) * 100),
                    'averageVolume': float(hist['Volume'].mean())
                }
            }

            logger.info(f"Prediction successful for {symbol}")
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return jsonify({'error': f'Error fetching data: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error predicting {symbol}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
