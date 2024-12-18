import React from 'react';
import { Line } from 'react-chartjs-2';

interface TechnicalIndicatorsProps {
    indicators: any;
}

export const TechnicalIndicatorsChart: React.FC<TechnicalIndicatorsProps> = ({ indicators }) => {
    const rsi = indicators.RSI;
    const macd = indicators.MACD;
    const bb = indicators.BB;

    return (
        <div className="technical-indicators-chart">
            <div className="indicator">
                <h4>RSI: {rsi.toFixed(2)}</h4>
                <div className={`indicator-status ${rsi > 70 ? 'overbought' : rsi < 30 ? 'oversold' : 'neutral'}`}>
                    {rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral'}
                </div>
            </div>

            <div className="indicator">
                <h4>MACD</h4>
                <div>Signal: {macd.signal.toFixed(2)}</div>
                <div>Histogram: {macd.histogram.toFixed(2)}</div>
            </div>

            <div className="indicator">
                <h4>Bollinger Bands</h4>
                <div>Upper: {bb.upper.toFixed(2)}</div>
                <div>Middle: {bb.middle.toFixed(2)}</div>
                <div>Lower: {bb.lower.toFixed(2)}</div>
            </div>
        </div>
    );
}; 