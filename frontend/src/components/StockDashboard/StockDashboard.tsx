import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import './StockDashboard.css';

interface StockData {
    symbol: string;
    name: string;
    current_price: number;
    price_change: number;
    price_change_percent: number;
    sentiment_score: number;
    prediction: string;
    historical_data: number[];
    last_updated: string;
}

const StockDashboard: React.FC = () => {
    const { symbol } = useParams<{ symbol: string }>();
    const [data, setData] = useState<StockData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            if (!symbol) return;

            try {
                setLoading(true);
                setError(null);

                const response = await fetch(`/api/stocks/data/${symbol}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    throw new Error('Invalid response format: Expected JSON');
                }

                const stockData = await response.json();
                setData(stockData);
            } catch (err) {
                console.error('Error fetching data:', err);
                setError(err instanceof Error ? err.message : 'Failed to fetch stock data');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [symbol]);

    const getSentimentColor = (score: number) => {
        if (score >= 4.0) return 'green';
        if (score >= 3.0) return 'orange';
        return 'red';
    };

    if (loading) return <div className="loading">Loading...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!data) return <div className="error">No data available</div>;

    return (
        <div className="stock-dashboard">
            <header className="dashboard-header">
                <h1>{data.name} ({data.symbol})</h1>
                <div className="price-info">
                    <div className="current-price">${data.current_price.toFixed(2)}</div>
                    <div className={`price-change ${data.price_change >= 0 ? 'positive' : 'negative'}`}>
                        {data.price_change >= 0 ? '+' : ''}{data.price_change.toFixed(2)} 
                        ({data.price_change_percent.toFixed(2)}%)
                    </div>
                </div>
            </header>

            <div className="sentiment-section">
                <h2>Market Sentiment</h2>
                <div className="sentiment-info">
                    <div 
                        className="sentiment-score"
                        style={{ color: getSentimentColor(data.sentiment_score) }}
                    >
                        Score: {data.sentiment_score.toFixed(1)}
                    </div>
                    <div className="prediction">
                        Prediction: <span className={`prediction-${data.prediction.toLowerCase()}`}>
                            {data.prediction}
                        </span>
                    </div>
                </div>
            </div>

            <div className="historical-section">
                <h2>Historical Data</h2>
                <div className="chart">
                    {/* Add chart visualization here */}
                    Coming soon: Price chart visualization
                </div>
            </div>

            <footer className="dashboard-footer">
                Last updated: {new Date(data.last_updated).toLocaleString()}
            </footer>
        </div>
    );
};

export default StockDashboard;
