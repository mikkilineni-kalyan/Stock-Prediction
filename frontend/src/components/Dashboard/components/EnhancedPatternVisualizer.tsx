import React from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ChartData,
    ChartOptions,
    ChartDataset
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-plugin-zoom';
import 'chartjs-plugin-annotation';

// Register components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface Pattern {
    name: string;
    confidence: number;
    direction: string;
    start_idx: number;
    end_idx: number;
}

interface Props {
    prices: number[];
    dates: string[];
    patterns: Pattern[];
    volumes: number[];
    indicators: {
        sma20?: number[];
        sma50?: number[];
        rsi?: number[];
        macd?: number[];
        signal?: number[];
    };
}

const EnhancedPatternVisualizer: React.FC<Props> = ({
    prices,
    dates,
    patterns,
    volumes,
    indicators
}) => {
    const datasets: ChartDataset<"line", number[]>[] = [
        {
            type: 'line',
            label: 'Price',
            data: prices,
            borderColor: 'rgb(75, 192, 192)',
            yAxisID: 'y',
            tension: 0.1
        }
    ];

    // Add volume dataset
    if (volumes.length > 0) {
        datasets.push({
            type: 'line',
            label: 'Volume',
            data: volumes,
            backgroundColor: 'rgba(53, 162, 235, 0.5)',
            yAxisID: 'y1',
            hidden: true
        });
    }

    // Add SMA indicators if available
    if (indicators.sma20) {
        datasets.push({
            type: 'line',
            label: 'SMA 20',
            data: indicators.sma20,
            borderColor: 'rgba(255, 99, 132, 1)',
            yAxisID: 'y',
            hidden: true
        });
    }

    if (indicators.sma50) {
        datasets.push({
            type: 'line',
            label: 'SMA 50',
            data: indicators.sma50,
            borderColor: 'rgba(54, 162, 235, 1)',
            yAxisID: 'y',
            hidden: true
        });
    }

    const chartData: ChartData<'line', number[], string> = {
        labels: dates,
        datasets
    };

    const options: ChartOptions<'line'> = {
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
                    text: 'Price'
                }
            },
            y1: {
                type: 'linear' as const,
                display: true,
                position: 'right' as const,
                title: {
                    display: true,
                    text: 'Volume'
                },
                grid: {
                    drawOnChartArea: false
                }
            }
        },
        plugins: {
            legend: {
                position: 'top' as const,
            },
            title: {
                display: true,
                text: 'Stock Price Analysis'
            },
            // @ts-ignore
            zoom: {
                zoom: {
                    wheel: { enabled: true },
                    pinch: { enabled: true },
                    mode: 'xy'
                },
                pan: { enabled: true }
            },
            // @ts-ignore
            annotation: {
                annotations: patterns.map(pattern => ({
                    type: 'box',
                    xMin: dates[pattern.start_idx],
                    xMax: dates[pattern.end_idx],
                    backgroundColor: pattern.direction === 'bullish' 
                        ? 'rgba(75, 192, 192, 0.25)'
                        : 'rgba(255, 99, 132, 0.25)',
                    borderColor: pattern.direction === 'bullish'
                        ? 'rgb(75, 192, 192)'
                        : 'rgb(255, 99, 132)',
                    borderWidth: 2,
                    label: {
                        display: true,
                        content: `${pattern.name} (${Math.round(pattern.confidence)}%)`,
                        position: 'start'
                    }
                }))
            }
        }
    };

    return (
        <div className="chart-container" style={{ position: 'relative', height: '70vh', width: '100%' }}>
            <Line options={options} data={chartData} />
        </div>
    );
};

export default EnhancedPatternVisualizer;