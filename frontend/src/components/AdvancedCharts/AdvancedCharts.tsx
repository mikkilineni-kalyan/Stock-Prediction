import React from 'react';
import Plot from 'react-plotly.js';

interface AdvancedChartsProps {
  data: {
    dates: string[];
    prices: number[];
    predictions: number[];
  };
}

const AdvancedCharts: React.FC<AdvancedChartsProps> = ({ data }) => {
  const trace1 = {
    x: data.dates,
    y: data.prices,
    type: 'scatter',
    mode: 'lines',
    name: 'Historical Prices',
    line: { color: '#17BECF' }
  };

  const trace2 = {
    x: data.dates,
    y: data.predictions,
    type: 'scatter',
    mode: 'lines',
    name: 'Predictions',
    line: { color: '#7F7F7F' }
  };

  const layout = {
    title: 'Stock Price History & Predictions',
    xaxis: {
      title: 'Date',
      rangeslider: { visible: true }
    },
    yaxis: {
      title: 'Price'
    }
  };

  return (
    <Plot
      data={[trace1, trace2] as any}
      layout={layout}
      style={{ width: '100%', height: '400px' }}
    />
  );
};

export default AdvancedCharts;
