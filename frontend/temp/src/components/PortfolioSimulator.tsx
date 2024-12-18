import React, { useState } from 'react';
import Plot from 'react-plotly.js';

interface SimulationResult {
  dates: string[];
  portfolioValue: number[];
  initialInvestment: number;
  returns: number;
  riskMetrics: {
    sharpeRatio: number;
    maxDrawdown: number;
    volatility: number;
  };
}

const PortfolioSimulator: React.FC = () => {
  const [investment, setInvestment] = useState<string>('');
  const [selectedStocks, setSelectedStocks] = useState<Array<{
    symbol: string;
    allocation: number;
  }>>([]);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [timeHorizon, setTimeHorizon] = useState<string>('1y');

  const addStock = () => {
    setSelectedStocks([...selectedStocks, { symbol: '', allocation: 0 }]);
  };

  const updateStock = (index: number, field: 'symbol' | 'allocation', value: string) => {
    const updated = selectedStocks.map((stock, i) => {
      if (i === index) {
        return { ...stock, [field]: field === 'allocation' ? Number(value) : value };
      }
      return stock;
    });
    setSelectedStocks(updated);
  };

  const removeStock = (index: number) => {
    setSelectedStocks(selectedStocks.filter((_, i) => i !== index));
  };

  const runSimulation = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/simulate-portfolio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          investment: Number(investment),
          stocks: selectedStocks,
          timeHorizon
        }),
      });
      const result = await response.json();
      setSimulationResult(result);
    } catch (error) {
      console.error('Simulation failed:', error);
    }
  };

  return (
    <div className="portfolio-simulator">
      <h3>Portfolio Simulator</h3>
      
      <div className="simulation-inputs">
        <div className="input-group">
          <label>Initial Investment ($)</label>
          <input
            type="number"
            value={investment}
            onChange={(e) => setInvestment(e.target.value)}
            placeholder="Enter amount"
          />
        </div>

        <div className="input-group">
          <label>Time Horizon</label>
          <select 
            value={timeHorizon} 
            onChange={(e) => setTimeHorizon(e.target.value)}
          >
            <option value="1m">1 Month</option>
            <option value="3m">3 Months</option>
            <option value="6m">6 Months</option>
            <option value="1y">1 Year</option>
            <option value="3y">3 Years</option>
            <option value="5y">5 Years</option>
          </select>
        </div>
      </div>

      <div className="stocks-allocation">
        <h4>Stock Allocation</h4>
        <button onClick={addStock}>Add Stock</button>
        
        {selectedStocks.map((stock, index) => (
          <div key={index} className="stock-input-row">
            <input
              type="text"
              value={stock.symbol}
              onChange={(e) => updateStock(index, 'symbol', e.target.value)}
              placeholder="Stock Symbol"
            />
            <input
              type="number"
              value={stock.allocation}
              onChange={(e) => updateStock(index, 'allocation', e.target.value)}
              placeholder="Allocation %"
            />
            <button onClick={() => removeStock(index)}>Remove</button>
          </div>
        ))}
      </div>

      <button 
        className="simulate-button"
        onClick={runSimulation}
        disabled={!investment || selectedStocks.length === 0}
      >
        Run Simulation
      </button>

      {simulationResult && (
        <div className="simulation-results">
          <Plot
            data={[
              {
                x: simulationResult.dates,
                y: simulationResult.portfolioValue,
                type: 'scatter',
                name: 'Portfolio Value',
                line: { color: '#2E86C1' }
              }
            ]}
            layout={{
              title: 'Portfolio Value Over Time',
              height: 400,
              showlegend: true
            }}
          />

          <div className="metrics-grid">
            <div className="metric-card">
              <h4>Total Returns</h4>
              <div className="value">
                {((simulationResult.returns - 1) * 100).toFixed(2)}%
              </div>
            </div>
            <div className="metric-card">
              <h4>Sharpe Ratio</h4>
              <div className="value">
                {simulationResult.riskMetrics.sharpeRatio.toFixed(2)}
              </div>
            </div>
            <div className="metric-card">
              <h4>Max Drawdown</h4>
              <div className="value">
                {(simulationResult.riskMetrics.maxDrawdown * 100).toFixed(2)}%
              </div>
            </div>
            <div className="metric-card">
              <h4>Volatility</h4>
              <div className="value">
                {(simulationResult.riskMetrics.volatility * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PortfolioSimulator;
