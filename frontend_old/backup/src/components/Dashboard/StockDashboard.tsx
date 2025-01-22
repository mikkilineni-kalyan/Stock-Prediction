import React, { useEffect, useState } from 'react';
import NewsPanel from '../NewsPanel/NewsPanel';
import PredictionSummary from '../PredictionSummary/PredictionSummary';
import PatternVisualizer from './components/PatternVisualizer';
import { DashboardProps, StockData } from '../../types/dashboard';
import './StockDashboard.css';

const StockDashboard: React.FC<DashboardProps> = ({
    ticker,
    startDate,
    endDate,
    companyName
}) => {
    const [data, setData] = useState<StockData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch(
                    `/api/stocks/data/${ticker}?start=${startDate}&end=${endDate}`
                );
                if (!response.ok) throw new Error('Failed to fetch stock data');
                const data = await response.json();
                setData(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'An error occurred');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [ticker, startDate, endDate]);

    if (loading) return <div className="loading">Loading...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!data) return <div className="error">No data available</div>;

    return (
        <div className="stock-dashboard">
            <h2>{companyName || ticker} Analysis</h2>
            <div className="dashboard-content">
                {data && (
                    <PatternVisualizer 
                        data={{
                            labels: data.dates,
                            datasets: [
                                {
                                    label: 'Price',
                                    data: data.prices,
                                    borderColor: '#2196F3',
                                    fill: false
                                },
                                {
                                    label: 'Volume',
                                    data: data.volumes,
                                    borderColor: '#4CAF50',
                                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                    fill: true,
                                    yAxisID: 'volume'
                                }
                            ]
                        }}
                        options={{
                            responsive: true,
                            scales: {
                                y: {
                                    type: 'linear',
                                    display: true,
                                    position: 'left'
                                },
                                volume: {
                                    type: 'linear',
                                    display: true,
                                    position: 'right',
                                    grid: {
                                        drawOnChartArea: false
                                    }
                                }
                            }
                        }}
                        patterns={data.patterns || []}
                    />
                )}
                <NewsPanel ticker={ticker} />
                <PredictionSummary ticker={ticker} />
            </div>
        </div>
    );
};

export default StockDashboard; 