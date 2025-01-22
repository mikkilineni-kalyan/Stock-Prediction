class Backtester:

    def __init__(self):

        self.strategies = {

            'momentum': self._momentum_strategy,

            'mean_reversion': self._mean_reversion_strategy,

            'ml_based': self._ml_based_strategy

        }

        self.performance_metrics = {}

        

    def run_backtest(self, strategy: str, symbol: str, start_date: str, end_date: str,

                    initial_capital: float = 100000.0) -> Dict[str, Any]:

        try:

            # Get historical data

            data = self._get_historical_data(symbol, start_date, end_date)

            

            # Run strategy

            trades = self.strategies[strategy](data)

            

            # Calculate performance

            performance = self._calculate_performance(trades, initial_capital)

            

            # Calculate metrics

            metrics = self._calculate_metrics(performance)

            

            return {

                'trades': trades,

                'performance': performance,

                'metrics': metrics

            }

        except Exception as e:

            logger.error(f"Backtest error: {str(e)}")

            return None 
