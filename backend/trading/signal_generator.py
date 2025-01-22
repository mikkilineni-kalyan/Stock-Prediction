import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
from pathlib import Path

from ..config.database import session, Stock, StockPrice, Prediction, TradingSignal
from ..ml_models.ensemble_predictor import EnsemblePredictor
from ..ml_models.model_validator import ModelValidator

logger = logging.getLogger(__name__)

class SignalType(Enum):
    STRONG_BUY = 'STRONG_BUY'
    BUY = 'BUY'
    HOLD = 'HOLD'
    SELL = 'SELL'
    STRONG_SELL = 'STRONG_SELL'

class SignalGenerator:
    def __init__(self):
        self.session = session
        self.predictor = EnsemblePredictor()
        self.validator = ModelValidator()
        
        # Signal generation parameters
        self.params = {
            'min_confidence': 0.7,
            'strong_threshold': 0.05,  # 5% price change
            'moderate_threshold': 0.02,  # 2% price change
            'volume_threshold': 1.5,  # 50% above average
            'trend_weight': 0.4,
            'sentiment_weight': 0.3,
            'volume_weight': 0.3
        }

    async def generate_signals(self, symbol: str) -> Dict:
        """Generate trading signals based on predictions and market conditions"""
        try:
            # Get latest predictions
            predictions = await self.predictor.predict(symbol)
            if 'error' in predictions:
                raise ValueError(f"Error getting predictions: {predictions['error']}")

            # Get historical data for context
            historical_data = self._get_historical_data(symbol)
            if not historical_data:
                raise ValueError(f"No historical data found for {symbol}")

            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(historical_data)

            # Generate signals
            signals = self._generate_trading_signals(
                predictions['predictions'],
                predictions['confidence_scores'],
                market_conditions
            )

            # Validate signals
            validated_signals = self._validate_signals(signals, historical_data)

            # Store signals
            await self._store_signals(symbol, validated_signals)

            return {
                'status': 'success',
                'signals': validated_signals,
                'market_conditions': market_conditions
            }

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical price and volume data"""
        try:
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                return None

            # Get last 30 days of data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)

            prices = self.session.query(StockPrice).filter(
                StockPrice.stock_id == stock.id,
                StockPrice.timestamp >= start_date,
                StockPrice.timestamp <= end_date
            ).order_by(StockPrice.timestamp.asc()).all()

            if not prices:
                return None

            return pd.DataFrame([{
                'timestamp': p.timestamp,
                'open': p.open_price,
                'high': p.high_price,
                'low': p.low_price,
                'close': p.close_price,
                'volume': p.volume
            } for p in prices])

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None

    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Analyze current market conditions"""
        try:
            # Calculate trend indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Trend analysis
            trend = {
                'short_term': 'bullish' if latest['sma_20'] > latest['sma_50'] else 'bearish',
                'momentum': latest['close'] - prev['close'],
                'volatility': df['close'].pct_change().std()
            }

            # Volume analysis
            volume = {
                'current': latest['volume'],
                'avg_volume': df['volume'].mean(),
                'volume_trend': 'increasing' if latest['volume'] > df['volume'].mean() else 'decreasing'
            }

            # Support and resistance
            support = df['low'].rolling(window=20).min().iloc[-1]
            resistance = df['high'].rolling(window=20).max().iloc[-1]

            return {
                'trend': trend,
                'volume': volume,
                'support': support,
                'resistance': resistance,
                'risk_level': self._calculate_risk_level(df)
            }

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}

    def _calculate_risk_level(self, df: pd.DataFrame) -> float:
        """Calculate current market risk level"""
        try:
            # Volatility risk
            volatility = df['close'].pct_change().std()
            vol_risk = min(volatility * 100, 1)  # Cap at 1

            # Volume risk
            volume_ratio = df['volume'].iloc[-1] / df['volume'].mean()
            vol_risk = min(abs(1 - volume_ratio), 1)

            # Trend risk
            price_trend = df['close'].pct_change().mean()
            trend_risk = min(abs(price_trend * 100), 1)

            # Combine risks
            total_risk = (vol_risk + vol_risk + trend_risk) / 3
            return round(total_risk, 2)

        except Exception as e:
            logger.error(f"Error calculating risk level: {str(e)}")
            return 0.5

    def _generate_trading_signals(self, predictions: List[float],
                                confidence_scores: List[float],
                                market_conditions: Dict) -> List[Dict]:
        """Generate trading signals based on predictions and market conditions"""
        try:
            signals = []
            current_price = market_conditions.get('current_price', 0)
            
            for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                if conf < self.params['min_confidence']:
                    signal_type = SignalType.HOLD
                else:
                    # Calculate predicted return
                    pred_return = (pred - current_price) / current_price

                    # Determine signal type based on predicted return
                    if pred_return > self.params['strong_threshold']:
                        signal_type = SignalType.STRONG_BUY
                    elif pred_return > self.params['moderate_threshold']:
                        signal_type = SignalType.BUY
                    elif pred_return < -self.params['strong_threshold']:
                        signal_type = SignalType.STRONG_SELL
                    elif pred_return < -self.params['moderate_threshold']:
                        signal_type = SignalType.SELL
                    else:
                        signal_type = SignalType.HOLD

                # Adjust signal based on market conditions
                signal_type = self._adjust_signal(
                    signal_type,
                    market_conditions,
                    conf
                )

                signals.append({
                    'timestamp': datetime.utcnow() + timedelta(days=i+1),
                    'signal_type': signal_type.value,
                    'confidence': conf,
                    'predicted_price': pred,
                    'predicted_return': pred_return,
                    'market_conditions': market_conditions
                })

            return signals

        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return []

    def _adjust_signal(self, signal_type: SignalType,
                      market_conditions: Dict,
                      confidence: float) -> SignalType:
        """Adjust signal based on market conditions"""
        try:
            # Get market factors
            trend = market_conditions.get('trend', {})
            volume = market_conditions.get('volume', {})
            risk_level = market_conditions.get('risk_level', 0.5)

            # Calculate adjustment factors
            trend_factor = 1 if trend.get('short_term') == 'bullish' else -1
            volume_factor = volume.get('current', 0) / volume.get('avg_volume', 1)

            # Combine factors
            total_factor = (
                trend_factor * self.params['trend_weight'] +
                volume_factor * self.params['volume_weight']
            )

            # Adjust signal based on factors and risk
            if risk_level > 0.7:  # High risk
                if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                    return SignalType.BUY if signal_type == SignalType.STRONG_BUY else SignalType.SELL
            elif total_factor > 0.5 and confidence > 0.8:  # Strong positive factors
                if signal_type == SignalType.BUY:
                    return SignalType.STRONG_BUY
            elif total_factor < -0.5 and confidence > 0.8:  # Strong negative factors
                if signal_type == SignalType.SELL:
                    return SignalType.STRONG_SELL

            return signal_type

        except Exception as e:
            logger.error(f"Error adjusting signal: {str(e)}")
            return signal_type

    def _validate_signals(self, signals: List[Dict],
                         historical_data: pd.DataFrame) -> List[Dict]:
        """Validate generated signals"""
        try:
            validated_signals = []
            
            for signal in signals:
                # Skip low confidence signals
                if signal['confidence'] < self.params['min_confidence']:
                    continue

                # Validate against historical patterns
                if self._validate_against_history(signal, historical_data):
                    # Check for signal consistency
                    if self._check_signal_consistency(signal, validated_signals):
                        validated_signals.append(signal)

            return validated_signals

        except Exception as e:
            logger.error(f"Error validating signals: {str(e)}")
            return signals

    def _validate_against_history(self, signal: Dict,
                                historical_data: pd.DataFrame) -> bool:
        """Validate signal against historical patterns"""
        try:
            # Get similar historical patterns
            similar_patterns = self._find_similar_patterns(
                signal,
                historical_data
            )

            # If we find similar patterns, check their accuracy
            if similar_patterns:
                accuracy = self._calculate_pattern_accuracy(similar_patterns)
                return accuracy >= 0.6  # 60% accuracy threshold

            return True  # If no similar patterns found, allow the signal

        except Exception as e:
            logger.error(f"Error validating against history: {str(e)}")
            return True

    def _check_signal_consistency(self, new_signal: Dict,
                                existing_signals: List[Dict]) -> bool:
        """Check if new signal is consistent with existing signals"""
        try:
            if not existing_signals:
                return True

            # Check for conflicting signals in short timeframe
            recent_signals = [s for s in existing_signals 
                            if abs((s['timestamp'] - new_signal['timestamp']).days) <= 2]

            for signal in recent_signals:
                if self._signals_conflict(signal['signal_type'],
                                       new_signal['signal_type']):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking signal consistency: {str(e)}")
            return True

    def _signals_conflict(self, signal1: str, signal2: str) -> bool:
        """Check if two signals conflict with each other"""
        try:
            buy_signals = [SignalType.STRONG_BUY.value, SignalType.BUY.value]
            sell_signals = [SignalType.STRONG_SELL.value, SignalType.SELL.value]

            return (signal1 in buy_signals and signal2 in sell_signals) or \
                   (signal1 in sell_signals and signal2 in buy_signals)

        except Exception as e:
            logger.error(f"Error checking signal conflict: {str(e)}")
            return False

    async def _store_signals(self, symbol: str, signals: List[Dict]):
        """Store generated signals in database"""
        try:
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                return

            # Store each signal
            for signal in signals:
                trading_signal = TradingSignal(
                    stock_id=stock.id,
                    timestamp=signal['timestamp'],
                    signal_type=signal['signal_type'],
                    confidence_score=signal['confidence'],
                    predicted_price=signal['predicted_price'],
                    predicted_return=signal['predicted_return'],
                    market_conditions=json.dumps(signal['market_conditions'])
                )
                self.session.add(trading_signal)

            self.session.commit()

        except Exception as e:
            logger.error(f"Error storing signals: {str(e)}")
            self.session.rollback()

    def _find_similar_patterns(self, signal: Dict,
                             historical_data: pd.DataFrame) -> List[Dict]:
        """Find similar historical price patterns"""
        try:
            # Implementation of pattern matching logic
            # This could use techniques like DTW or correlation
            return []

        except Exception as e:
            logger.error(f"Error finding similar patterns: {str(e)}")
            return []

    def _calculate_pattern_accuracy(self, patterns: List[Dict]) -> float:
        """Calculate accuracy of similar historical patterns"""
        try:
            if not patterns:
                return 0.0

            # Implementation of pattern accuracy calculation
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating pattern accuracy: {str(e)}")
            return 0.0
