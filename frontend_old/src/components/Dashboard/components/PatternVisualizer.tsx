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
    ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Pattern } from '../../../types/dashboard';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface PatternVisualizerProps {
    data: {
        labels: string[];
        datasets: Array<{
            label: string;
            data: number[];
            borderColor?: string;
            backgroundColor?: string;
            fill?: boolean;
            yAxisID?: string;
        }>;
    };
    options: ChartOptions<'line'>;
    patterns: Pattern[];
}

const PatternVisualizer: React.FC<PatternVisualizerProps> = ({
    data,
    options,
    patterns
}) => {
    return (
        <div className="pattern-visualizer">
            <div className="chart-container">
                <Line data={data} options={options} />
            </div>
            <div className="patterns-legend">
                {patterns.map((pattern, index) => (
                    <div key={index} className="pattern-item">
                        <span className="pattern-name">{pattern.name}</span>
                        <span className="pattern-confidence">
                            {(pattern.confidence * 100).toFixed(0)}%
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default PatternVisualizer; 