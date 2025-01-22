import React from 'react';

interface TechnicalIndicators {
  rsi: number;
  macd: number;
  ma20: number;
  ma50: number;
  bollingerBands: {
    upper: number;
    middle: number;
    lower: number;
  };
}

const TechnicalAnalysis: React.FC<{ indicators: TechnicalIndicators }> = ({ indicators }) => {
  const getRSISignal = (rsi: number) => {
    if (rsi > 70) return { signal: 'Overbought', color: 'red' };
    if (rsi < 30) return { signal: 'Oversold', color: 'green' };
    return { signal: 'Neutral', color: 'gray' };
  };

  const getMACDSignal = (macd: number) => {
    if (macd > 0) return { signal: 'Bullish', color: 'green' };
    return { signal: 'Bearish', color: 'red' };
  };

  const rsiSignal = getRSISignal(indicators.rsi);
  const macdSignal = getMACDSignal(indicators.macd);

  return (
    <div className="technical-analysis">
      <h3>Technical Analysis Summary</h3>
      <div className="indicators-grid">
        <div className="indicator-card">
          <h4>RSI</h4>
          <div className="indicator-value">{indicators.rsi.toFixed(2)}</div>
          <div className="signal" style={{ color: rsiSignal.color }}>
            {rsiSignal.signal}
          </div>
        </div>
        
        <div className="indicator-card">
          <h4>MACD</h4>
          <div className="indicator-value">{indicators.macd.toFixed(2)}</div>
          <div className="signal" style={{ color: macdSignal.color }}>
            {macdSignal.signal}
          </div>
        </div>
        
        <div className="indicator-card">
          <h4>Moving Averages</h4>
          <div className="ma-values">
            <div>MA20: ${indicators.ma20.toFixed(2)}</div>
            <div>MA50: ${indicators.ma50.toFixed(2)}</div>
          </div>
        </div>
        
        <div className="indicator-card">
          <h4>Bollinger Bands</h4>
          <div className="bb-values">
            <div>Upper: ${indicators.bollingerBands.upper.toFixed(2)}</div>
            <div>Middle: ${indicators.bollingerBands.middle.toFixed(2)}</div>
            <div>Lower: ${indicators.bollingerBands.lower.toFixed(2)}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TechnicalAnalysis;
