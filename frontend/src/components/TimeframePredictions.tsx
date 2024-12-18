import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface FuturePrediction {
    target_price: number;
    price_range: {
        low: number;
        high: number;
    };
    confidence: number;
    direction: string;
    expected_change_percent: number;
    days_ahead: number;
}

interface TimeframePredictionsProps {
    predictions: {
        [date: string]: FuturePrediction;
    };
    currentPrice: number;
    historicalPrices: number[];
    dates: string[];
}

export const TimeframePredictions: React.FC<TimeframePredictionsProps> = ({
    predictions,
    currentPrice,
    historicalPrices,
    dates
}) => {
    // Sort predictions by date
    const sortedDates = Object.keys(predictions).sort();
    const predictionValues = sortedDates.map(date => predictions[date].target_price);
    
    const chartData = {
        labels: [...dates, ...sortedDates],
        datasets: [
            {
                label: 'Historical Price',
                data: [...historicalPrices, null],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            },
            {
                label: 'Predicted Price',
                data: [...Array(historicalPrices.length).fill(null), ...predictionValues],
                borderColor: 'rgb(255, 99, 132)',
                borderDash: [5, 5],
                tension: 0.1
            }
        ]
    };

    const options = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top' as const,
            },
            tooltip: {
                callbacks: {
                    label: function(context: any) {
                        const label = context.dataset.label || '';
                        const value = context.parsed.y;
                        const timeframe = context.label;
                        
                        if (timeframe === '1D' || timeframe === '1W') {
                            const prediction = timeframe === '1D' ? predictions.daily : predictions.weekly;
                            return [
                                `${label}: $${value?.toFixed(2)}`,
                                `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`,
                                `Expected Change: ${prediction.expected_change_percent.toFixed(2)}%`
                            ];
                        }
                        return `${label}: $${value?.toFixed(2)}`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                title: {
                    display: true,
                    text: 'Price ($)'
                }
            }
        }
    };

    return (
        <div className="timeframe-predictions">
            <h3>Future Price Predictions</h3>
            <div className="prediction-summary">
                {sortedDates.map(date => (
                    <div key={date} className="prediction-item">
                        <h4>{date}</h4>
                        <div className={`direction ${predictions[date].direction}`}>
                            Direction: {predictions[date].direction.toUpperCase()}
                        </div>
                        <div>Target: ${predictions[date].target_price.toFixed(2)}</div>
                        <div>Confidence: {(predictions[date].confidence * 100).toFixed(1)}%</div>
                        <div>Expected Change: {predictions[date].expected_change_percent.toFixed(2)}%</div>
                    </div>
                ))}
            </div>
            <div style={{ height: '400px' }}>
                <Line data={chartData} options={options} />
            </div>
        </div>
    );
}; 