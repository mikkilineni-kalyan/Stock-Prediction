import React, { useState, useEffect } from 'react';

interface PortfolioStock {
  symbol: string;
  shares: number;
  purchasePrice: number;
  currentPrice: number;
  totalValue: number;
  gainLoss: number;
  gainLossPercent: number;
}

const PortfolioTracker: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioStock[]>(() => {
    const saved = localStorage.getItem('portfolio');
    return saved ? JSON.parse(saved) : [];
  });
  const [newStock, setNewStock] = useState({
    symbol: '',
    shares: '',
    purchasePrice: ''
  });

  useEffect(() => {
    localStorage.setItem('portfolio', JSON.stringify(portfolio));
  }, [portfolio]);

  const addToPortfolio = async () => {
    if (!newStock.symbol || !newStock.shares || !newStock.purchasePrice) return;

    try {
      const response = await fetch(`http://localhost:5000/api/stock/${newStock.symbol}`);
      const data = await response.json();
      const currentPrice = data.currentPrice;

      const newPortfolioStock: PortfolioStock = {
        symbol: newStock.symbol.toUpperCase(),
        shares: Number(newStock.shares),
        purchasePrice: Number(newStock.purchasePrice),
        currentPrice: currentPrice,
        totalValue: currentPrice * Number(newStock.shares),
        gainLoss: (currentPrice - Number(newStock.purchasePrice)) * Number(newStock.shares),
        gainLossPercent: ((currentPrice - Number(newStock.purchasePrice)) / Number(newStock.purchasePrice)) * 100
      };

      setPortfolio([...portfolio, newPortfolioStock]);
      setNewStock({ symbol: '', shares: '', purchasePrice: '' });
    } catch (error) {
      console.error('Error adding stock:', error);
    }
  };

  const removeFromPortfolio = (symbol: string) => {
    setPortfolio(portfolio.filter(stock => stock.symbol !== symbol));
  };

  const getTotalValue = () => {
    return portfolio.reduce((total, stock) => total + stock.totalValue, 0);
  };

  const getTotalGainLoss = () => {
    return portfolio.reduce((total, stock) => total + stock.gainLoss, 0);
  };

  return (
    <div className="portfolio-tracker">
      <h3>Portfolio Tracker</h3>
      
      <div className="portfolio-summary">
        <div className="summary-card">
          <h4>Total Value</h4>
          <div className="value">${getTotalValue().toFixed(2)}</div>
        </div>
        <div className="summary-card">
          <h4>Total Gain/Loss</h4>
          <div className={`value ${getTotalGainLoss() >= 0 ? 'positive' : 'negative'}`}>
            ${Math.abs(getTotalGainLoss()).toFixed(2)}
            {getTotalGainLoss() >= 0 ? ' ▲' : ' ▼'}
          </div>
        </div>
      </div>

      <div className="add-stock-form">
        <input
          type="text"
          placeholder="Symbol"
          value={newStock.symbol}
          onChange={e => setNewStock({...newStock, symbol: e.target.value})}
        />
        <input
          type="number"
          placeholder="Shares"
          value={newStock.shares}
          onChange={e => setNewStock({...newStock, shares: e.target.value})}
        />
        <input
          type="number"
          placeholder="Purchase Price"
          value={newStock.purchasePrice}
          onChange={e => setNewStock({...newStock, purchasePrice: e.target.value})}
        />
        <button onClick={addToPortfolio}>Add to Portfolio</button>
      </div>

      <div className="portfolio-table">
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Shares</th>
              <th>Purchase Price</th>
              <th>Current Price</th>
              <th>Total Value</th>
              <th>Gain/Loss</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.map(stock => (
              <tr key={stock.symbol}>
                <td>{stock.symbol}</td>
                <td>{stock.shares}</td>
                <td>${stock.purchasePrice.toFixed(2)}</td>
                <td>${stock.currentPrice.toFixed(2)}</td>
                <td>${stock.totalValue.toFixed(2)}</td>
                <td className={stock.gainLoss >= 0 ? 'positive' : 'negative'}>
                  ${Math.abs(stock.gainLoss).toFixed(2)}
                  ({stock.gainLossPercent.toFixed(2)}%)
                  {stock.gainLoss >= 0 ? ' ▲' : ' ▼'}
                </td>
                <td>
                  <button onClick={() => removeFromPortfolio(stock.symbol)}>Remove</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PortfolioTracker;
