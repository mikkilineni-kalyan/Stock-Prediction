import React from 'react';
import Plot from 'react-plotly.js';

interface IndicatorProps {
  data: {
    dates: string[];
    prices: number[];
    ma20: number[];
    ma50: number[];
    rsi: number[];
    bollingerBands: {
      upper: number[];
      middle: number[];
      lower: number[];
    };
  };
}

const TechnicalIndicators: React.FC<IndicatorProps> = ({ data }) => {
  return (
    <div className="technical-indicators">
      <div className="indicator-charts">
        <Plot
          data={[
            // Price and Moving Averages
            {
              x: data.dates,
              y: data.prices,
              type: 'scatter',
              name: 'Price',
              line: { color: '#2E86C1' }
            },
            {
              x: data.dates,
              y: data.ma20,
              type: 'scatter',
              name: '20-day MA',
              line: { color: '#E67E22' }
            },
            {
              x: data.dates,
              y: data.ma50,
              type: 'scatter',
              name: '50-day MA',
              line: { color: '#8E44AD' }
            }
          ]}
          layout={{
            title: 'Price and Moving Averages',
            height: 400,
            showlegend: true
          }}
        />
        
        <Plot
          data={[
            // RSI
            {
              x: data.dates,
              y: data.rsi,
              type: 'scatter',
              name: 'RSI',
              line: { color: '#2ECC71' }
            },
            // Overbought/Oversold lines
            {
              x: data.dates,
              y: Array(data.dates.length).fill(70),
              type: 'scatter',
              name: 'Overbought',
              line: { dash: 'dash', color: '#E74C3C' }
            },
            {
              x: data.dates,
              y: Array(data.dates.length).fill(30),
              type: 'scatter',
              name: 'Oversold',
              line: { dash: 'dash', color: '#E74C3C' }
            }
          ]}
          layout={{
            title: 'RSI',
            height: 300,
            showlegend: true
          }}
        />
      </div>
    </div>
  );
};

export default TechnicalIndicators;

