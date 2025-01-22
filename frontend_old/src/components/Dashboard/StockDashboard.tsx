import React, { useEffect, useState } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ChartData,
    ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './StockDashboard.css';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend
);

interface DashboardProps {
    ticker: string;
}

interface StockData {
    dates: string[];
    prices: number[];
    volumes: number[];
    prediction: {
        next_day: number;
        confidence: number;
        trend: string;
        predicted_return: number;
    };
    news_sentiment: {
        score: number;
        impact: string;
        confidence: number;
        sources: number;
        summary: string;
    };
    indicators: {
        ma5: number[];
        ma20: number[];
        ma50: number[];
        rsi: number[];
        bollinger: {
            upper: number[];
            middle: number[];
            lower: number[];
        };
        volume: {
            raw: number[];
            sma: number[];
            ratio: number[];
        };
    };
}

type TimePeriod = '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y' | 'max';

const StockDashboard: React.FC<DashboardProps> = ({ ticker }) => {
    const [data, setStockData] = useState<StockData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [period, setPeriod] = useState<TimePeriod>('1y');
    const [selectedChart, setSelectedChart] = useState<'price' | 'volume' | 'indicators'>('price');

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);
                
                console.log('Fetching data for ticker:', ticker, 'period:', period);
                
                const response = await fetch(
                    `http://localhost:5000/api/stocks/data/${ticker}?period=${period}`,
                    {
                        method: 'GET',
                        headers: {
                            'Accept': 'application/json',
                            'Content-Type': 'application/json'
                        }
                    }
                );
                
                const jsonData = await response.json();
                console.log('Raw response:', jsonData);

                // Check for error in the response
                if (jsonData.status === 'error' || !response.ok) {
                    throw new Error(jsonData.error || 'Failed to fetch stock data');
                }

                // Extract the actual data from the response
                const responseData = jsonData.data || {};

                // Default data structure
                const defaultData = {
                    dates: [],
                    prices: [],
                    volumes: [],
                    prediction: {
                        next_day: 0,
                        confidence: 50.0,
                        trend: 'neutral',
                        predicted_return: 0
                    },
                    news_sentiment: {
                        score: 2.5,
                        impact: 'NEUTRAL',
                        confidence: 0.5,
                        sources: 0,
                        summary: 'No news data available'
                    },
                    indicators: {
                        ma5: [],
                        ma20: [],
                        ma50: [],
                        rsi: [],
                        bollinger: {
                            upper: [],
                            middle: [],
                            lower: []
                        },
                        volume: {
                            raw: [],
                            sma: [],
                            ratio: []
                        }
                    }
                };

                // Safe number conversion
                const toNumber = (value: any, defaultValue: number): number => {
                    if (typeof value === 'number') return value;
                    if (typeof value === 'string') {
                        const num = Number(value);
                        return isNaN(num) ? defaultValue : num;
                    }
                    return defaultValue;
                };

                // Safe array conversion
                const toNumberArray = (arr: any[] | null | undefined, defaultArr: number[]): number[] => {
                    if (!Array.isArray(arr)) return defaultArr;
                    return arr.map(val => toNumber(val, 0));
                };

                // Validate and merge received data with defaults
                const completeData = {
                    dates: Array.isArray(responseData.dates) ? responseData.dates : defaultData.dates,
                    prices: toNumberArray(responseData.prices, defaultData.prices),
                    volumes: toNumberArray(responseData.volumes, defaultData.volumes),
                    prediction: {
                        next_day: toNumber(responseData.prediction?.next_day, defaultData.prediction.next_day),
                        confidence: toNumber(responseData.prediction?.confidence, defaultData.prediction.confidence),
                        trend: String(responseData.prediction?.trend || defaultData.prediction.trend),
                        predicted_return: toNumber(responseData.prediction?.predicted_return, defaultData.prediction.predicted_return)
                    },
                    news_sentiment: {
                        score: toNumber(responseData.news_sentiment?.score, defaultData.news_sentiment.score),
                        impact: String(responseData.news_sentiment?.impact || defaultData.news_sentiment.impact),
                        confidence: toNumber(responseData.news_sentiment?.confidence, defaultData.news_sentiment.confidence),
                        sources: toNumber(responseData.news_sentiment?.sources, defaultData.news_sentiment.sources),
                        summary: String(responseData.news_sentiment?.summary || defaultData.news_sentiment.summary)
                    },
                    indicators: {
                        ma5: toNumberArray(responseData.indicators?.ma5, defaultData.indicators.ma5),
                        ma20: toNumberArray(responseData.indicators?.ma20, defaultData.indicators.ma20),
                        ma50: toNumberArray(responseData.indicators?.ma50, defaultData.indicators.ma50),
                        rsi: toNumberArray(responseData.indicators?.rsi, defaultData.indicators.rsi),
                        bollinger: {
                            upper: toNumberArray(responseData.indicators?.bollinger?.upper, defaultData.indicators.bollinger.upper),
                            middle: toNumberArray(responseData.indicators?.bollinger?.middle, defaultData.indicators.bollinger.middle),
                            lower: toNumberArray(responseData.indicators?.bollinger?.lower, defaultData.indicators.bollinger.lower)
                        },
                        volume: {
                            raw: toNumberArray(responseData.indicators?.volume?.raw, defaultData.indicators.volume.raw),
                            sma: toNumberArray(responseData.indicators?.volume?.sma, defaultData.indicators.volume.sma),
                            ratio: toNumberArray(responseData.indicators?.volume?.ratio, defaultData.indicators.volume.ratio)
                        }
                    }
                };

                console.log('Processed data:', completeData);
                setStockData(completeData);
                
            } catch (err) {
                console.error('Error fetching data:', err);
                setError(err instanceof Error ? err.message : 'Failed to fetch data');
            } finally {
                setLoading(false);
            }
        };

        if (ticker) {
            fetchData();
        }
    }, [ticker, period]);

    if (loading) return <div className="loading">Loading stock data...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!data) return <div className="error">No data available</div>;

    const chartData: ChartData<'line'> = {
        labels: data?.dates || [],
        datasets: [
            {
                label: 'Price',
                data: data?.prices || [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                yAxisID: 'y',
            },
            {
                label: 'MA5',
                data: data?.indicators.ma5 || [],
                borderColor: 'rgba(255, 99, 132, 0.8)',
                borderWidth: 1,
                tension: 0.1,
                yAxisID: 'y',
            },
            {
                label: 'MA20',
                data: data?.indicators.ma20 || [],
                borderColor: 'rgba(54, 162, 235, 0.8)',
                borderWidth: 1,
                tension: 0.1,
                yAxisID: 'y',
            },
            {
                label: 'MA50',
                data: data?.indicators.ma50 || [],
                borderColor: 'rgba(255, 206, 86, 0.8)',
                borderWidth: 1,
                tension: 0.1,
                yAxisID: 'y',
            },
            {
                label: 'Bollinger Upper',
                data: data?.indicators.bollinger.upper || [],
                borderColor: 'rgba(75, 192, 192, 0.8)',
                borderDash: [5, 5],
                borderWidth: 1,
                tension: 0.1,
                yAxisID: 'y',
            },
            {
                label: 'Bollinger Lower',
                data: data?.indicators.bollinger.lower || [],
                borderColor: 'rgba(75, 192, 192, 0.8)',
                borderDash: [5, 5],
                borderWidth: 1,
                tension: 0.1,
                yAxisID: 'y',
            }
        ],
    };

    const volumeData: ChartData<'line'> = {
        labels: data?.dates || [],
        datasets: [
            {
                label: 'Volume',
                data: data?.volumes || [],
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                yAxisID: 'y',
            },
            {
                label: 'Volume SMA',
                data: data?.indicators.volume.sma || [],
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 2,
                tension: 0.1,
                yAxisID: 'y',
            }
        ],
    };

    const indicatorData: ChartData<'line'> = {
        labels: data?.dates || [],
        datasets: [
            {
                label: 'RSI',
                data: data?.indicators.rsi || [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                yAxisID: 'y1',
            }
        ],
    };

    const chartOptions: ChartOptions<'line'> = {
        responsive: true,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: `${ticker} ${selectedChart === 'price' ? 'Stock Price' : 
                       selectedChart === 'volume' ? 'Volume Analysis' : 
                       'Technical Indicators'}`,
            },
        },
        scales: selectedChart === 'price' ? {
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: {
                    display: true,
                    text: 'Price ($)'
                }
            }
        } : selectedChart === 'volume' ? {
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: {
                    display: true,
                    text: 'Volume'
                }
            }
        } : {
            y1: {
                type: 'linear',
                display: true,
                position: 'left',
                title: {
                    display: true,
                    text: 'RSI'
                }
            }
        },
    };

    const getChartData = () => {
        switch (selectedChart) {
            case 'volume':
                return volumeData;
            case 'indicators':
                return indicatorData;
            default:
                return chartData;
        }
    };

    // Get the latest RSI value safely
    const latestRsi = data?.indicators.rsi[data?.indicators.rsi.length - 1];
    const getRsiClass = (rsi: number | null) => {
        if (rsi === null) return 'neutral';
        if (rsi > 70) return 'overbought';
        if (rsi < 30) return 'oversold';
        return 'neutral';
    };

    return (
        <div className="stock-dashboard">
            <div className="controls">
                <div className="time-selector">
                    <label>Time Period:</label>
                    <select value={period} onChange={(e) => setPeriod(e.target.value as TimePeriod)}>
                        <option value="1d">1 Day</option>
                        <option value="5d">5 Days</option>
                        <option value="1mo">1 Month</option>
                        <option value="3mo">3 Months</option>
                        <option value="6mo">6 Months</option>
                        <option value="1y">1 Year</option>
                        <option value="2y">2 Years</option>
                        <option value="5y">5 Years</option>
                        <option value="max">Max</option>
                    </select>
                </div>
                <div className="chart-selector">
                    <label>Chart Type:</label>
                    <select value={selectedChart} onChange={(e) => setSelectedChart(e.target.value as 'price' | 'volume' | 'indicators')}>
                        <option value="price">Price</option>
                        <option value="volume">Volume</option>
                        <option value="indicators">Technical Indicators</option>
                    </select>
                </div>
            </div>

            <div className="prediction-summary">
                <h3>Prediction Summary</h3>
                <div className="prediction-details">
                    <p>Next Day Prediction: <strong>${data?.prediction?.next_day?.toFixed(2) || 'N/A'}</strong></p>
                    <p>Expected Return: <strong>{data?.prediction?.predicted_return ? `${data.prediction.predicted_return}%` : 'N/A'}</strong></p>
                    <p>Confidence: <strong>{data?.prediction?.confidence ? `${(data.prediction.confidence * 100).toFixed(1)}%` : 'N/A'}</strong></p>
                    <p>Score: <strong>
                        {data?.news_sentiment?.score 
                            ? (typeof data.news_sentiment.score === 'string' 
                                ? parseFloat(data.news_sentiment.score).toFixed(1)
                                : data.news_sentiment.score.toFixed(1)) + '/5'
                            : 'N/A'}
                    </strong></p>
                    <p>Trend: <strong className={`trend-${data?.prediction?.trend?.toLowerCase() || 'neutral'}`}>
                        {data?.prediction?.trend?.toUpperCase() || 'NEUTRAL'}
                    </strong></p>
                </div>
            </div>
            
            <div className="chart-container">
                <Line data={getChartData()} options={chartOptions} />
            </div>
            
            <div className="technical-indicators">
                <h3>Technical Indicators</h3>
                <div className="indicator-grid">
                    <div className="indicator">
                        <label>RSI (14):</label>
                        <span className={getRsiClass(latestRsi)}>
                            {latestRsi?.toFixed(2) || 'N/A'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StockDashboard;