import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { Layout, Data } from 'plotly.js';
import './StockPredictor.css';
import { StockSearch } from './StockSearch';

interface TimeFrame {
  value: string;
  label: string;
  days: number;
}

interface PredictionData {
  symbol: string;
  currentPrice: number;
  dates: string[];
  predictions: number[];
  startDateTime: string;
  timeFrame: string;
  confidenceIntervals?: {
    lower: number[];
    upper: number[];
  };
  metrics?: {
    volatility: number;
    trend: number;
    averageVolume: number;
    priceChange1d: number;
    priceChange7d: number;
    priceChange30d: number;
    predictedChange: number;
  };
}

const TIME_FRAMES: TimeFrame[] = [
  { value: '1d', label: '1 Day', days: 1 },
  { value: '1w', label: '1 Week', days: 7 },
  { value: '2w', label: '2 Weeks', days: 14 },
  { value: '1m', label: '1 Month', days: 30 },
  { value: '3m', label: '3 Months', days: 90 },
  { value: '6m', label: '6 Months', days: 180 },
];

// Add time validation helper
const isMarketHours = (time: string): boolean => {
  const [hours, minutes] = time.split(':').map(Number);
  const marketOpen = 9 * 60 + 30;  // 9:30 AM in minutes
  const marketClose = 16 * 60;     // 4:00 PM in minutes
  const currentTime = hours * 60 + minutes;
  return currentTime >= marketOpen && currentTime <= marketClose;
};

const API_BASE_URL = 'http://localhost:5000';

const StockPredictor: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('AAPL');
  const [timeFrame, setTimeFrame] = useState<string>('1w');
  const [startDate, setStartDate] = useState<string>(
    new Date().toISOString().split('T')[0]
  );
  const [startTime, setStartTime] = useState<string>('09:30');
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const selectedTimeFrame = TIME_FRAMES.find(tf => tf.value === timeFrame);
      const startDateTime = new Date(`${startDate}T${startTime}`);

      const requestData = {
        days: selectedTimeFrame?.days || 7,
        startDateTime: startDateTime.toISOString(),
      };

      try {
        const response = await fetch(`${API_BASE_URL}/api/predict/${symbol}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify(requestData),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setPredictionData(data);
      } catch (fetchError) {
        if (fetchError instanceof TypeError && fetchError.message === 'Failed to fetch') {
          setError('Unable to connect to the prediction server. Please ensure the server is running.');
        } else {
          throw fetchError;
        }
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction');
    } finally {
      setIsLoading(false);
    }
  };

  const renderPredictionResults = () => {
    if (!predictionData) return null;

    const plotData: Partial<Data>[] = [
      {
        x: predictionData.dates,
        y: predictionData.predictions,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Price',
        line: { color: '#17B897' },
        showlegend: true
      }
    ];

    if (predictionData.confidenceIntervals) {
      // Add upper bound
      plotData.push({
        x: predictionData.dates,
        y: predictionData.confidenceIntervals.upper,
        type: 'scatter',
        mode: 'lines',
        name: 'Confidence Interval',
        line: { color: 'rgba(23, 184, 151, 0.2)' },
        showlegend: true
      } as Partial<Data>);

      // Add lower bound with fill
      plotData.push({
        x: predictionData.dates,
        y: predictionData.confidenceIntervals.lower,
        type: 'scatter',
        mode: 'lines',
        name: 'Lower Bound',
        line: { color: 'rgba(23, 184, 151, 0.2)' },
        fillcolor: 'rgba(23, 184, 151, 0.1)',
        fill: 'tonexty',
        showlegend: false
      } as Partial<Data>);
    }

    const layout: Partial<Layout> = {
      title: `${predictionData.symbol} Price Prediction`,
      xaxis: {
        title: 'Date',
        showgrid: true,
        gridcolor: '#f0f0f0'
      },
      yaxis: {
        title: 'Price ($)',
        showgrid: true,
        gridcolor: '#f0f0f0',
        tickprefix: '$'
      },
      height: 500,
      margin: { l: 50, r: 50, t: 50, b: 50 },
      showlegend: true,
      legend: {
        x: 0,
        y: 1,
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        bordercolor: '#E2E8F0'
      },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      hovermode: 'x unified'
    };

    return (
      <div className="prediction-results">
        <Plot
          data={plotData}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false
          }}
        />

        <div className="metrics-grid">
          <div className="metric-card">
            <h3>Current Price</h3>
            <p>${predictionData.currentPrice.toFixed(2)}</p>
          </div>
          
          <div className="metric-card">
            <h3>Predicted Price</h3>
            <p>${predictionData.predictions[predictionData.predictions.length - 1].toFixed(2)}</p>
          </div>

          {predictionData.metrics && (
            <>
              <div className="metric-card">
                <h3>Predicted Change</h3>
                <p className={predictionData.metrics.predictedChange >= 0 ? 'positive' : 'negative'}>
                  {predictionData.metrics.predictedChange.toFixed(2)}%
                </p>
              </div>

              <div className="metric-card">
                <h3>Volatility</h3>
                <p>{(predictionData.metrics.volatility * 100).toFixed(2)}%</p>
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="stock-predictor">
      <h2>Stock Price Predictor</h2>
      
      <div className="input-controls">
        <div className="input-group">
          <label>Stock Search</label>
          <StockSearch 
            onSelect={(symbol) => setSymbol(symbol)}
            initialValue={symbol}
          />
        </div>

        <div className="input-group">
          <label>Time Frame</label>
          <select value={timeFrame} onChange={(e) => setTimeFrame(e.target.value)}>
            {TIME_FRAMES.map(tf => (
              <option key={tf.value} value={tf.value}>{tf.label}</option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label>Start Date</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            min={new Date().toISOString().split('T')[0]}
          />
        </div>

        <div className="input-group">
          <label>Start Time</label>
          <input
            type="time"
            value={startTime}
            onChange={(e) => setStartTime(e.target.value)}
            min="09:30"
            max="16:00"
          />
        </div>
      </div>

      <button 
        onClick={handlePredict}
        disabled={isLoading}
        className="predict-button"
      >
        {isLoading ? 'Predicting...' : 'Predict'}
      </button>

      <div className="popular-stocks">
        <span>Popular stocks:</span>
        {['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'].map((stock) => (
          <button
            key={stock}
            onClick={() => setSymbol(stock)}
            className={`stock-button ${symbol === stock ? 'selected' : ''}`}
          >
            {stock}
          </button>
        ))}
      </div>

      {error && (
        <div className="error-message">{error}</div>
      )}

      {isLoading ? (
        <div className="loading-spinner">Loading prediction...</div>
      ) : (
        renderPredictionResults()
      )}
    </div>
  );
};

export default StockPredictor;