# Stock Prediction Platform 📈

A modern web application that combines news sentiment analysis with machine learning to predict stock market movements. Built with React/TypeScript frontend and Python/Flask backend.

## 🚀 Features

- **Real-time Stock Search**: Quick search functionality with autocomplete for stock symbols
- **Sentiment Analysis**: Advanced sentiment scoring based on market news
- **Interactive Dashboard**: Comprehensive view of stock performance and predictions
- **Price Analytics**: Real-time price updates and historical data visualization
- **Prediction Engine**: Machine learning-based market movement predictions

## 🛠️ Tech Stack

### Frontend
- React.js
- TypeScript
- React Router
- Modern CSS with Flexbox/Grid
- Real-time data updates

### Backend
- Python
- Flask
- Flask-CORS
- yfinance for market data
- Logging system for debugging

## 🏗️ Project Structure

```
Stock-Prediction/
├── frontend/                # React/TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── StockSearch/
│   │   │   └── StockDashboard/
│   │   └── setupProxy.js   # API proxy configuration
│   └── package.json
│
└── backend/                # Python/Flask backend
    └── test_server.py     # API endpoints and business logic
```

## 🚦 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- Python 3.8+
- npm or yarn

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction
```

2. Install Frontend Dependencies
```bash
cd frontend
npm install
```

3. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Running the Application

1. Start the Backend Server
```bash
cd backend
python test_server.py
```

2. Start the Frontend Development Server
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## 🌟 Key Features in Detail

### Stock Search
- Real-time stock symbol search
- Company information display
- Sentiment score visualization
- Quick navigation to detailed analysis

### Stock Dashboard
- Current price and changes
- Historical price data
- Sentiment analysis results
- Prediction indicators
- Interactive charts (coming soon)

## 🔄 API Endpoints

- `GET /api/stocks/search?q={query}`: Search for stocks
- `GET /api/stocks/data/{symbol}`: Get detailed stock information

## 🚧 Current Development Status

The project is actively under development with the following features planned:

- [ ] Advanced ML prediction models
- [ ] Real-time news integration
- [ ] Enhanced visualization tools
- [ ] User authentication
- [ ] Portfolio tracking
- [ ] Email alerts for price movements

## Configuration

The application uses a hierarchical configuration system that loads settings from multiple sources:

1. Environment Variables (highest priority)
2. `.env` file
3. `config.json` file (lowest priority)

### Setting Up Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Update the `.env` file with your API keys and configuration:
- `ALPHA_VANTAGE_KEY`: API key for Alpha Vantage financial data
- `NEWS_API_KEY`: API key for News API
- `FINNHUB_API_KEY`: API key for Finnhub stock data
- `POLYGON_API_KEY`: API key for Polygon.io market data

### Database Configuration
- `DB_DIALECT`: Database dialect (default: sqlite)
- `DB_NAME`: Database name
- `DB_POOL_SIZE`: Connection pool size
- `DB_MAX_OVERFLOW`: Maximum pool overflow
- `DB_POOL_TIMEOUT`: Pool timeout in seconds
- `DB_POOL_RECYCLE`: Connection recycle time in seconds

### Model Configuration
- `MODEL_VERSION_PATH`: Path to model versions
- `MODEL_REGISTRY_PATH`: Path to model registry
- `MLFLOW_EXPERIMENT_NAME`: MLflow experiment name
- `MLFLOW_TRACKING_URI`: MLflow tracking URI

### API Configuration
- `API_RATE_LIMIT`: API rate limit per minute
- `JWT_SECRET`: Secret key for JWT tokens
- `JWT_ALGORITHM`: JWT algorithm (default: HS256)
- `TOKEN_EXPIRE_MINUTES`: JWT token expiration time

### Cache Configuration
- `REDIS_HOST`: Redis host
- `REDIS_PORT`: Redis port
- `REDIS_DB`: Redis database number

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details

## 🙏 Acknowledgments

- yfinance for providing market data
- React community for excellent documentation
- Flask for a robust backend framework

---
⚠️ Note: This project is in active development. Features and documentation may change frequently.
