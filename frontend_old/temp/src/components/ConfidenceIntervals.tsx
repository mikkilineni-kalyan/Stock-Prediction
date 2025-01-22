import React from 'react';
import Plot from 'react-plotly.js';

interface ConfidenceProps {
  predictions: {
    dates: string[];
    predicted: number[];
    upperBound: number[];
    lowerBound: number[];
  };
}

const ConfidenceIntervals: React.FC<ConfidenceProps> = ({ predictions }) => {
  return (
    <div className="confidence-intervals">
      <Plot
        data={[
          // Main prediction line
          {
            x: predictions.dates,
            y: predictions.predicted,
            type: 'scatter',
            name: 'Predicted',
            line: { color: '#2E86C1' }
          },
          // Confidence interval
          {
            x: predictions.dates.concat(predictions.dates.slice().reverse()),
            y: predictions.upperBound.concat(predictions.lowerBound.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(46, 134, 193, 0.2)',
            line: { color: 'transparent' },
            name: 'Confidence Interval'
          }
        ]}
        layout={{
          title: 'Price Prediction with Confidence Intervals',
          height: 400,
          showlegend: true
        }}
      />
    </div>
  );
};

export default ConfidenceIntervals;
