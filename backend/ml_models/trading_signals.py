class TradingSignalGenerator:
    def __init__(self):
        self.thresholds = {
            'price_change': {'buy': 0.02, 'sell': -0.02},
            'rsi': {'overbought': 70, 'oversold': 30},
            'volume': {'spike': 2.0},
            'macd': {'threshold': 0},
            'sentiment': {'positive': 0.7, 'negative': 0.3}
        }
        
    def generate_signals(self, data, technical_indicators, sentiment_score):
        signals = []
        
        # Price action signals
        price_signals = self._check_price_patterns(data)
        signals.extend(price_signals)
        
        # Technical indicator signals
        tech_signals = self._check_technical_signals(technical_indicators)
        signals.extend(tech_signals)
        
        # Sentiment-based signals
        sentiment_signals = self._check_sentiment_signals(sentiment_score)
        signals.extend(sentiment_signals)
        
        return self._prioritize_signals(signals) 