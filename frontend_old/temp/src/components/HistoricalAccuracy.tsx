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

interface HistoricalAccuracyProps {
    accuracyData: {
        dates: string[];
        accuracy: number[];
        confidence: number[];
    };
}

export const HistoricalAccuracy: React.FC<HistoricalAccuracyProps> = ({ accuracyData }) => {
    const data = {
        labels: accuracyData.dates,
        datasets: [
            {
                label: 'Prediction Accuracy',
                data: accuracyData.accuracy,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                yAxisID: 'y'
            },
            {
                label: 'Confidence Level',
                data: accuracyData.confidence,
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                yAxisID: 'y1'
            }
        ]
    };

    const options = {
        responsive: true,
        interaction: {
            mode: 'index' as const,
            intersect: false,
        },
        scales: {
            y: {
                type: 'linear' as const,
                display: true,
                position: 'left' as const,
                title: {
                    display: true,
                    text: 'Accuracy %'
                }
            },
            y1: {
                type: 'linear' as const,
                display: true,
                position: 'right' as const,
                title: {
                    display: true,
                    text: 'Confidence %'
                },
                grid: {
                    drawOnChartArea: false,
                }
            }
        }
    };

    return (
        <div className="historical-accuracy-chart">
            <h3>Historical Prediction Performance</h3>
            <div style={{ height: '400px' }}>
                <Line data={data} options={options} />
            </div>
        </div>
    );
}; 