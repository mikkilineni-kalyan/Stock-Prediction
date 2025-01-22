import React from 'react';
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

// Remove the extending interface and define it directly
interface PlotDataPoint {
    x: (number | string)[];
    y: number[];
    type: 'scatter' | 'bar' | 'line' | 'box';
    mode?: 'lines' | 'markers' | 'lines+markers';
    name?: string;
    marker?: {
        color?: string;
        size?: number;
    };
    line?: {
        color?: string;
        width?: number;
    };
}

interface AdvancedChartsProps {
    data: {
        prices: number[];
        dates: string[];
        volumes: number[];
        indicators?: {
            rsi?: number[];
            macd?: {
                line: number[];
                signal: number[];
                histogram: number[];
            };
        };
    };
}

const AdvancedCharts: React.FC<AdvancedChartsProps> = ({ data }) => {
    const plotData: PlotDataPoint[] = [
        {
            x: data.dates,
            y: data.prices,
            type: 'scatter',
            name: 'Price',
            mode: 'lines+markers'
        },
        {
            x: data.dates,
            y: data.volumes,
            type: 'bar',
            name: 'Volume',
            marker: {
                color: 'rgba(0,0,0,0.1)'
            }
        }
    ];

    const layout: Partial<Layout> = {
        title: 'Advanced Technical Analysis',
        showlegend: true,
        xaxis: {
            title: 'Date',
            rangeslider: { visible: true }
        },
        yaxis: {
            title: 'Price',
            side: 'left'
        },
        yaxis2: {
            title: 'Volume',
            overlaying: 'y',
            side: 'right'
        }
    };

    return (
        <div className="advanced-charts">
            <Plot
                data={plotData as Data[]}
                layout={layout}
                style={{ width: '100%', height: '600px' }}
            />
        </div>
    );
};

export default AdvancedCharts; 