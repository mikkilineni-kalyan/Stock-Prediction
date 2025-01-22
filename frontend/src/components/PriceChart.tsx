import React from 'react';
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
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface PriceChartProps {
    historicalPrices: number[];
    predictedPrices: number[];
    dates: string[];
    futureDates: string[];
}

export const PriceChart: React.FC<PriceChartProps> = ({
    historicalPrices,
    predictedPrices,
    dates,
    futureDates
}) => {
    const data = {
        labels: [...dates, ...futureDates],
        datasets: [
            {
                label: 'Historical Price',
                data: [...historicalPrices, ...Array(futureDates.length).fill(null)],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            },
            {
                label: 'Predicted Price',
                data: [...Array(dates.length).fill(null), ...predictedPrices],
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
            title: {
                display: true,
                text: 'Stock Price Prediction'
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
        <div className="price-chart">
            <Line data={data} options={options} />
        </div>
    );
}; 