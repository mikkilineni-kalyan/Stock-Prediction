import React from 'react';
import { StockPrediction } from '../types/types';

interface Props {
    prediction: StockPrediction;
}

export const StockPredictionComponent: React.FC<Props> = ({ prediction }) => {
    return (
        <div className="prediction-container">
            <h2>{prediction.company} ({prediction.ticker})</h2>
            <div className="current-price">
                Current Price: ${prediction.current_price.toFixed(2)}
            </div>
            
            <div className="predictions">
                {Object.entries(prediction.predictions).map(([date, pred]) => (
                    <div key={date} className="prediction-item">
                        <h3>{date}</h3>
                        <div className={`direction ${pred.direction}`}>
                            Direction: {pred.direction}
                        </div>
                        <div>Target Price: ${pred.target_price.toFixed(2)}</div>
                        <div>Range: ${pred.price_range.low.toFixed(2)} - ${pred.price_range.high.toFixed(2)}</div>
                        <div>Confidence: {(pred.confidence * 100).toFixed(1)}%</div>
                    </div>
                ))}
            </div>
            
            <div className="news-sentiment">
                <h3>News Sentiment</h3>
                <div>Score: {prediction.news_sentiment.score}/5</div>
                <div>Impact: {prediction.news_sentiment.impact}</div>
                <div>Confidence: {(prediction.news_sentiment.confidence * 100).toFixed(1)}%</div>
                <div>Sources: {prediction.news_sentiment.sources}</div>
                <div className="summary">{prediction.news_sentiment.summary}</div>
            </div>
        </div>
    );
}; 