import React, { useState } from 'react';

interface Alert {
  id: string;
  symbol: string;
  targetPrice: number;
  type: 'above' | 'below';
}

const PriceAlerts: React.FC<{ symbol: string; currentPrice: number }> = ({ symbol, currentPrice }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [targetPrice, setTargetPrice] = useState<string>('');
  const [alertType, setAlertType] = useState<'above' | 'below'>('above');

  const addAlert = () => {
    if (!targetPrice) return;
    
    const newAlert: Alert = {
      id: Date.now().toString(),
      symbol,
      targetPrice: parseFloat(targetPrice),
      type: alertType
    };
    
    setAlerts([...alerts, newAlert]);
    setTargetPrice('');
  };

  const removeAlert = (id: string) => {
    setAlerts(alerts.filter(alert => alert.id !== id));
  };

  return (
    <div className="price-alerts">
      <h3>Price Alerts</h3>
      <div className="alert-form">
        <select 
          value={alertType} 
          onChange={(e) => setAlertType(e.target.value as 'above' | 'below')}
        >
          <option value="above">Above</option>
          <option value="below">Below</option>
        </select>
        <input
          type="number"
          value={targetPrice}
          onChange={(e) => setTargetPrice(e.target.value)}
          placeholder="Target Price"
          step="0.01"
        />
        <button onClick={addAlert}>Add Alert</button>
      </div>
      
      <div className="alerts-list">
        {alerts.map(alert => (
          <div key={alert.id} className="alert-item">
            <span>Alert when {symbol} goes {alert.type} ${alert.targetPrice}</span>
            <button onClick={() => removeAlert(alert.id)}>Remove</button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PriceAlerts;
