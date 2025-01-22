import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..config.database import session, Stock, StockPrice, Prediction
from .ensemble_predictor import EnsemblePredictor

logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self):
        self.session = session
        self.predictor = EnsemblePredictor()
        self.validation_metrics = [
            'mse', 'rmse', 'mape', 'hit_rate', 'profit_factor'
        ]
        self.reports_dir = Path("reports/validation")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def validate_model(self, symbol: str, validation_period: int = 30) -> Dict:
        """
        Validate model performance using multiple metrics
        Returns validation results and generates validation report
        """
        try:
            # Get historical predictions and actual prices
            predictions = self._get_historical_predictions(symbol, validation_period)
            if not predictions:
                raise ValueError(f"No predictions found for {symbol}")

            # Calculate validation metrics
            metrics = self._calculate_metrics(predictions)

            # Generate validation report
            report = self._generate_validation_report(symbol, predictions, metrics)

            # Save report
            self._save_validation_report(symbol, report)

            return {
                'status': 'success',
                'metrics': metrics,
                'report_path': str(self.reports_dir / f"{symbol}_validation_report.json")
            }

        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def backtest_strategy(self, symbol: str, start_date: datetime,
                              end_date: datetime) -> Dict:
        """
        Perform backtesting of the prediction strategy
        """
        try:
            # Get historical data
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                raise ValueError(f"Stock {symbol} not found")

            prices = self.session.query(StockPrice).filter(
                StockPrice.stock_id == stock.id,
                StockPrice.timestamp >= start_date,
                StockPrice.timestamp <= end_date
            ).order_by(StockPrice.timestamp.asc()).all()

            if not prices:
                raise ValueError("Insufficient data for backtesting")

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': p.timestamp,
                'actual_price': p.close_price
            } for p in prices])

            # Perform walk-forward optimization
            results = self._walk_forward_test(df)

            # Calculate performance metrics
            metrics = self._calculate_backtest_metrics(results)

            # Generate backtest report
            report = self._generate_backtest_report(symbol, results, metrics)

            # Save report
            self._save_backtest_report(symbol, report)

            return {
                'status': 'success',
                'metrics': metrics,
                'report_path': str(self.reports_dir / f"{symbol}_backtest_report.json")
            }

        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _get_historical_predictions(self, symbol: str, days: int) -> List[Dict]:
        """Get historical predictions and actual prices"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                return []

            # Get predictions
            predictions = self.session.query(Prediction).filter(
                Prediction.stock_id == stock.id,
                Prediction.timestamp >= start_date,
                Prediction.timestamp <= end_date
            ).order_by(Prediction.timestamp.asc()).all()

            # Get actual prices
            prices = self.session.query(StockPrice).filter(
                StockPrice.stock_id == stock.id,
                StockPrice.timestamp >= start_date,
                StockPrice.timestamp <= end_date
            ).order_by(StockPrice.timestamp.asc()).all()

            # Match predictions with actual prices
            price_dict = {p.timestamp.date(): p.close_price for p in prices}
            
            results = []
            for pred in predictions:
                pred_date = pred.timestamp.date()
                if pred_date in price_dict:
                    results.append({
                        'date': pred_date,
                        'predicted': pred.predicted_price,
                        'actual': price_dict[pred_date],
                        'confidence': pred.confidence_score
                    })

            return results

        except Exception as e:
            logger.error(f"Error getting historical predictions: {str(e)}")
            return []

    def _calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate various validation metrics"""
        try:
            if not predictions:
                return {}

            y_true = [p['actual'] for p in predictions]
            y_pred = [p['predicted'] for p in predictions]
            confidence = [p['confidence'] for p in predictions]

            # Basic metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true, y_pred)

            # Direction accuracy (hit rate)
            actual_direction = np.sign(np.diff([p['actual'] for p in predictions]))
            pred_direction = np.sign(np.diff([p['predicted'] for p in predictions]))
            hit_rate = np.mean(actual_direction == pred_direction)

            # Profit factor
            gains = []
            losses = []
            for i in range(len(predictions)-1):
                if pred_direction[i] == actual_direction[i]:
                    gains.append(abs(y_true[i+1] - y_true[i]))
                else:
                    losses.append(abs(y_true[i+1] - y_true[i]))

            profit_factor = sum(gains) / sum(losses) if losses else float('inf')

            # Confidence-weighted accuracy
            weighted_accuracy = np.mean(
                [1 - abs(p['actual'] - p['predicted'])/p['actual'] * p['confidence']
                 for p in predictions]
            )

            return {
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'hit_rate': hit_rate,
                'profit_factor': profit_factor,
                'weighted_accuracy': weighted_accuracy
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _walk_forward_test(self, df: pd.DataFrame) -> List[Dict]:
        """Perform walk-forward optimization"""
        try:
            results = []
            tscv = TimeSeriesSplit(n_splits=5)

            for train_idx, test_idx in tscv.split(df):
                train_data = df.iloc[train_idx]
                test_data = df.iloc[test_idx]

                # Train model on training data
                # Make predictions on test data
                # Store results

                for idx in test_idx:
                    results.append({
                        'date': df.iloc[idx]['timestamp'],
                        'actual': df.iloc[idx]['actual_price'],
                        'predicted': 0,  # Replace with actual prediction
                        'train_size': len(train_idx),
                        'test_size': len(test_idx)
                    })

            return results

        except Exception as e:
            logger.error(f"Error in walk-forward testing: {str(e)}")
            return []

    def _calculate_backtest_metrics(self, results: List[Dict]) -> Dict:
        """Calculate backtest performance metrics"""
        try:
            metrics = {
                'total_trades': len(results),
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

            if not results:
                return metrics

            # Calculate returns
            returns = []
            for i in range(len(results)-1):
                if results[i]['predicted'] > results[i]['actual']:
                    returns.append(
                        (results[i+1]['actual'] - results[i]['actual']) / 
                        results[i]['actual']
                    )

            if returns:
                metrics.update({
                    'win_rate': len([r for r in returns if r > 0]) / len(returns),
                    'profit_factor': sum([r for r in returns if r > 0]) / 
                                   abs(sum([r for r in returns if r < 0])) 
                                   if sum([r for r in returns if r < 0]) != 0 else float('inf'),
                    'max_drawdown': self._calculate_max_drawdown(returns),
                    'sharpe_ratio': np.mean(returns) / np.std(returns) 
                                   if np.std(returns) != 0 else 0
                })

            return metrics

        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {str(e)}")
            return {}

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return abs(min(drawdown))

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _generate_validation_report(self, symbol: str, predictions: List[Dict],
                                  metrics: Dict) -> Dict:
        """Generate detailed validation report"""
        try:
            return {
                'symbol': symbol,
                'validation_date': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'predictions_sample': predictions[:10],
                'validation_period': len(predictions),
                'model_version': 'ensemble_v1'
            }

        except Exception as e:
            logger.error(f"Error generating validation report: {str(e)}")
            return {}

    def _generate_backtest_report(self, symbol: str, results: List[Dict],
                                metrics: Dict) -> Dict:
        """Generate detailed backtest report"""
        try:
            return {
                'symbol': symbol,
                'backtest_date': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'results_sample': results[:10],
                'total_trades': len(results),
                'model_version': 'ensemble_v1'
            }

        except Exception as e:
            logger.error(f"Error generating backtest report: {str(e)}")
            return {}

    def _save_validation_report(self, symbol: str, report: Dict):
        """Save validation report to file"""
        try:
            report_path = self.reports_dir / f"{symbol}_validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}")

    def _save_backtest_report(self, symbol: str, report: Dict):
        """Save backtest report to file"""
        try:
            report_path = self.reports_dir / f"{symbol}_backtest_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving backtest report: {str(e)}")

    def plot_validation_results(self, predictions: List[Dict], 
                              save_path: Optional[str] = None):
        """Generate validation plots"""
        try:
            dates = [p['date'] for p in predictions]
            actual = [p['actual'] for p in predictions]
            predicted = [p['predicted'] for p in predictions]

            plt.figure(figsize=(12, 6))
            plt.plot(dates, actual, label='Actual', color='blue')
            plt.plot(dates, predicted, label='Predicted', color='red', linestyle='--')
            plt.title('Prediction vs Actual Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting validation results: {str(e)}")
