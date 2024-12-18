class PortfolioManager:

    def __init__(self):

        self.positions = {}

        self.cash = 0

        self.risk_limits = {

            'max_position_size': 0.2,  # 20% of portfolio

            'max_sector_exposure': 0.3,  # 30% of portfolio

            'stop_loss': 0.05  # 5% stop loss

        }

        self.performance_metrics = {}

        

    def analyze_portfolio(self):

        try:

            return {

                'total_value': self._calculate_total_value(),

                'positions': self._analyze_positions(),

                'risk_metrics': self._calculate_risk_metrics(),

                'performance': self._calculate_performance(),

                'recommendations': self._generate_recommendations()

            }

        except Exception as e:

            logger.error(f"Portfolio analysis error: {str(e)}")

            return None 

        

    def add_position(self, symbol: str, quantity: int, price: float):

        try:

            cost = quantity * price

            if self._check_risk_limits(symbol, cost):

                self.positions[symbol] = {

                    'quantity': quantity,

                    'avg_price': price,

                    'current_price': price,

                    'stop_loss': price * (1 - self.risk_limits['stop_loss'])

                }

                self.cash -= cost

                return True

            return False

        except Exception as e:

            logger.error(f"Error adding position: {str(e)}")

            return False

            

    def update_positions(self, market_data: Dict[str, float]):

        for symbol, data in market_data.items():

            if symbol in self.positions:

                self.positions[symbol]['current_price'] = data['price']

                self._check_stop_loss(symbol)
