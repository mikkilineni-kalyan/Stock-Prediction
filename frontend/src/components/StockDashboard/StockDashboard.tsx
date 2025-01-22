import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import AdvancedCharts from '../AdvancedCharts/AdvancedCharts';
import { StockData } from '../../types/stock';
import './StockDashboard.css';

const StockDashboard: React.FC = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStockData = async () => {
      if (!symbol) return;
      
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch(`/api/stocks/data/${symbol}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setStockData(data);
      } catch (err) {
        console.error('Error fetching stock data:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch stock data');
      } finally {
        setLoading(false);
      }
    };

    fetchStockData();
  }, [symbol]);

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!stockData) {
    return <div className="error">No data available</div>;
  }

  const chartData = {
    dates: stockData.historicalData.map(d => d.date),
    prices: stockData.historicalData.map(d => d.close),
    predictions: stockData.historicalData.map(() => stockData.prediction.nextDay)
  };

  return (
    <div className="stock-dashboard">
      <h2>{symbol} Stock Analysis</h2>
      <div className="stock-info">
        <div className="price-info">
          <h3>Current Price: ${stockData.price.toFixed(2)}</h3>
          <p className={stockData.change >= 0 ? 'positive' : 'negative'}>
            Change: {stockData.change.toFixed(2)} ({stockData.changePercent.toFixed(2)}%)
          </p>
        </div>
        <div className="prediction-info">
          <h3>Prediction</h3>
          <p>Next Day: ${stockData.prediction.nextDay.toFixed(2)}</p>
          <p>Confidence: {(stockData.prediction.confidence * 100).toFixed(1)}%</p>
          <p>Trend: {stockData.prediction.trend}</p>
        </div>
      </div>
      <div className="chart-container">
        <AdvancedCharts data={chartData} />
      </div>
    </div>
  );
};

export default StockDashboard;
