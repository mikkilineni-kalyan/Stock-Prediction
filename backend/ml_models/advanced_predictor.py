import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedPredictor:
    def __init__(self):
        """Initialize the predictor with basic technical analysis capabilities"""
        self.default_response = {
            'prediction': 'neutral',
            'confidence': 50.0,
            'next_day': None,
            'predicted_return': 0.0,
            'technical_indicators': {},
            'analysis': ['Insufficient data for prediction']
        }

    def predict_comprehensive(self, symbol: str) -> Dict[str, Any]:
        """
        Multi-stage comprehensive prediction using technical analysis
        Returns a dictionary with prediction details including next day price and confidence
        """
        try:
            logger.info(f"Starting comprehensive prediction for {symbol}")
            
            # Get historical data
            try:
                stock = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                logger.debug(f"Fetching data for {symbol} from {start_date} to {end_date}")
                data = stock.history(period='1y', interval='1d')
                
                if data.empty:
                    logger.warning(f"No historical data available for {symbol}")
                    return self.default_response
                    
                logger.info(f"Retrieved {len(data)} days of historical data for {symbol}")
                logger.debug(f"Data columns: {data.columns.tolist()}")
                logger.debug(f"First few rows:\n{data.head()}")
                logger.debug(f"Last few rows:\n{data.tail()}")
                
            except Exception as e:
                logger.error(f"Error retrieving data for {symbol}: {str(e)}")
                return self.default_response
            
            # Calculate technical indicators
            try:
                technical_indicators = self._calculate_technical_indicators(data)
                logger.debug(f"Calculated technical indicators: {technical_indicators}")
            except Exception as e:
                logger.error(f"Error calculating technical indicators: {str(e)}", exc_info=True)
                return self.default_response
            
            # Generate prediction
            try:
                prediction = self._generate_prediction(technical_indicators)
                logger.debug(f"Generated prediction: {prediction}")
            except Exception as e:
                logger.error(f"Error generating prediction: {str(e)}", exc_info=True)
                return self.default_response
            
            # Calculate next day price prediction
            try:
                next_day_price, predicted_return = self._predict_next_day_price(
                    data, 
                    prediction['direction'],
                    prediction['confidence']
                )
                logger.debug(f"Predicted next day price: {next_day_price}, return: {predicted_return}")
            except Exception as e:
                logger.error(f"Error predicting next day price: {str(e)}", exc_info=True)
                next_day_price = data['Close'].iloc[-1]
                predicted_return = 0.0
            
            result = {
                'prediction': prediction['direction'],
                'confidence': prediction['confidence'],
                'next_day': float(next_day_price),
                'predicted_return': float(predicted_return),
                'technical_indicators': technical_indicators,
                'analysis': prediction['analysis']
            }
            
            logger.info(f"Completed prediction for {symbol}: {prediction['direction']}, confidence: {prediction['confidence']}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive prediction for {symbol}: {str(e)}", exc_info=True)
            return self.default_response

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        indicators = {}
        
        try:
            # Ensure data is not empty
            if len(data) < 50:  # Need at least 50 days of data
                raise ValueError("Insufficient historical data")
            
            # Moving averages
            indicators['SMA_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['SMA_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = float((100 - (100 / (1 + rs))).iloc[-1])
            
            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = float(macd.iloc[-1])
            indicators['Signal_Line'] = float(signal.iloc[-1])
            
            # Volatility (annualized)
            indicators['Volatility'] = float(data['Close'].pct_change().std() * np.sqrt(252))
            
            # Price momentum
            indicators['Price_Change_1D'] = float((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100)
            indicators['Price_Change_5D'] = float((data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100)
            
            # Current Price
            indicators['Current_Price'] = float(data['Close'].iloc[-1])
            
            # Round all values
            return {k: round(float(v), 2) for k, v in indicators.items()}
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            raise

    def _generate_prediction(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Generate prediction based on technical indicators"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            analysis = []
            
            # RSI Analysis
            if indicators['RSI'] < 30:
                bullish_signals += 2  # Strong oversold signal
                analysis.append("RSI indicates oversold conditions (strongly bullish)")
            elif indicators['RSI'] < 40:
                bullish_signals += 1
                analysis.append("RSI indicates potential oversold conditions (bullish)")
            elif indicators['RSI'] > 70:
                bearish_signals += 2  # Strong overbought signal
                analysis.append("RSI indicates overbought conditions (strongly bearish)")
            elif indicators['RSI'] > 60:
                bearish_signals += 1
                analysis.append("RSI indicates potential overbought conditions (bearish)")
                
            # MACD Analysis
            if indicators['MACD'] > indicators['Signal_Line']:
                bullish_signals += 1
                analysis.append("MACD is above signal line (bullish)")
            else:
                bearish_signals += 1
                analysis.append("MACD is below signal line (bearish)")
                
            # Moving Average Analysis
            if indicators['SMA_20'] > indicators['SMA_50']:
                bullish_signals += 1
                analysis.append("Short-term MA above long-term MA (bullish)")
            else:
                bearish_signals += 1
                analysis.append("Short-term MA below long-term MA (bearish)")
                
            # Recent Price Movement
            if indicators['Price_Change_1D'] > 1.0:
                bullish_signals += 2
                analysis.append("Strong positive momentum (strongly bullish)")
            elif indicators['Price_Change_1D'] > 0:
                bullish_signals += 1
                analysis.append("Positive momentum (bullish)")
            elif indicators['Price_Change_1D'] < -1.0:
                bearish_signals += 2
                analysis.append("Strong negative momentum (strongly bearish)")
            else:
                bearish_signals += 1
                analysis.append("Negative momentum (bearish)")
                
            # Calculate prediction
            total_signals = bullish_signals + bearish_signals
            if total_signals == 0:
                direction = "neutral"
                confidence = 50.0
            else:
                if bullish_signals > bearish_signals:
                    direction = "bullish"
                    confidence = min(90.0, (bullish_signals / total_signals) * 100)
                elif bearish_signals > bullish_signals:
                    direction = "bearish"
                    confidence = min(90.0, (bearish_signals / total_signals) * 100)
                else:
                    direction = "neutral"
                    confidence = 50.0
                    
            return {
                'direction': direction,
                'confidence': round(confidence, 2),
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}", exc_info=True)
            raise

    def _predict_next_day_price(self, data: pd.DataFrame, direction: str, confidence: float) -> Tuple[float, float]:
        """Predict next day's price based on trend and confidence"""
        try:
            current_price = float(data['Close'].iloc[-1])
            volatility = float(data['Close'].pct_change().std())
            
            # Calculate predicted return based on direction and confidence
            if direction == "bullish":
                predicted_return = (confidence / 100.0) * volatility
            elif direction == "bearish":
                predicted_return = -(confidence / 100.0) * volatility
            else:
                predicted_return = 0.0
            
            # Calculate predicted price
            next_day_price = current_price * (1 + predicted_return)
            
            return round(next_day_price, 2), round(predicted_return * 100, 2)
            
        except Exception as e:
            logger.error(f"Error predicting next day price: {str(e)}", exc_info=True)
            raise
