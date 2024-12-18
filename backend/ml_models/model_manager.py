from .sentiment_analyzer import NewsAnalyzer







from .indicators import FinancialAnalyzer







from .stock_predictor import StockPredictor







from .alpha_vantage_client import AlphaVantageClient







from .news_analyzer import AdvancedNewsAnalyzer







from .indicators import TechnicalAnalyzer







from .market_monitor import MarketMonitor







from .ensemble_model import EnsembleModel















import logging







import os















logger = logging.getLogger(__name__)















class ModelManager:







    def __init__(self):







        self.news_analyzer = AdvancedNewsAnalyzer()







        self.technical_analyzer = TechnicalAnalyzer()







        self.market_monitor = MarketMonitor()







        self.stock_predictor = StockPredictor()







        self.ensemble_model = EnsembleModel()







        self.alpha_vantage = AlphaVantageClient()















    async def get_complete_analysis(self, symbol, start_date, end_date):







        try:







            # Get all types of analysis







            news_impact = await self.news_analyzer.analyze_impact(symbol)







            technical_analysis = self.technical_analyzer.calculate_all(self._get_historical_data(symbol))







            market_status = self.market_monitor.get_market_status()







            alpha_vantage_data = self.alpha_vantage.get_stock_data(symbol)















            # Combine all features







            features = self._combine_features(







                news_impact,







                technical_analysis,







                market_status,







                alpha_vantage_data







            )
















            # Get ensemble prediction







            base_predictions = self.stock_predictor.predict(symbol, start_date, end_date)







            ensemble_predictions = self.ensemble_model.predict(features)















            # Calculate confidence scores







            confidence_scores = self._calculate_confidence_scores(







                news_impact,







                technical_analysis,







                market_status







            )
















            # Generate alerts







            alerts = self._generate_alerts(







                symbol,







                news_impact,







                technical_analysis,







                market_status







            )















            return {







                'predictions': self._weighted_combine_predictions(







                    base_predictions,







                    ensemble_predictions,







                    weights=[0.6, 0.4]







                ),







                'analysis': {







                    'news_sentiment': news_impact,







                    'technical': technical_analysis,







                    'market': market_status,







                    'alpha_vantage': alpha_vantage_data







                },







                'confidence_scores': confidence_scores,







                'alerts': alerts







            }















        except Exception as e:







            logger.error(f"Complete analysis error: {str(e)}")







            return None






























