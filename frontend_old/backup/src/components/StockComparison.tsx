import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

interface ComparisonData {
  dates: string[];
  prices1: number[];
  prices2: number[];
  symbol1: string;
  symbol2: string;
}

const StockComparison: React.FC<{ mainSymbol: string }> = ({ mainSymbol }) => {
  const [compareSymbol, setCompareSymbol] = useState('');
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);

  const fetchComparisonData = async () => {
    try {
      const response = await axios.get<ComparisonData>(
        `http://localhost:5000/api/compare/${mainSymbol}/${compareSymbol}`
      );
      setComparisonData(response.data);
    } catch (error) {
      console.error('Error fetching comparison data:', error);
    }
  };

  return (
    <div className="stock-comparison">
      <h3>Compare with Another Stock</h3>
      <div className="comparison-input">
        <input
          type="text"
          value={compareSymbol}
          onChange={(e) => setCompareSymbol(e.target.value.toUpperCase())}
          placeholder="Enter symbol (e.g., MSFT)"
        />
        <button onClick={fetchComparisonData}>Compare</button>
      </div>

      {comparisonData && (
        <Plot
          data={[
            {
              x: comparisonData.dates,
              y: comparisonData.prices1,
              type: 'scatter',
              name: comparisonData.symbol1,
              line: { color: '#2E86C1' }
            },
            {
              x: comparisonData.dates,
              y: comparisonData.prices2,
              type: 'scatter',
              name: comparisonData.symbol2,
              line: { color: '#e74c3c' }
            }
          ]}
          layout={{
            title: `${comparisonData.symbol1} vs ${comparisonData.symbol2}`,
            showlegend: true,
            height: 400,
            yaxis: { title: 'Normalized Price (%)' }
          }}
        />
      )}
    </div>
  );
};

export default StockComparison;
