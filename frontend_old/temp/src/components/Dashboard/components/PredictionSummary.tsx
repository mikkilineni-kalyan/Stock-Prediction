import React from 'react';
import './PredictionSummary.css';

interface PredictionData {
    technical_indicators: {
        BB: {
            upper: number;
            lower: number;
        };
        RSI: number;
        MACD: {
            line: number;
            signal: number;
        };
    };
    news_analysis: Array<{
        sentiment_score: number;
        impact_score: number;
    }>;
    patterns_detected: {
        [key: string]: boolean;
    };
    market_data: {
        prices: number[];
        volumes: number[];
    };
}

interface PredictionSummaryProps {
    data: PredictionData;
}

export const PredictionSummary: React.FC<PredictionSummaryProps> = ({ data }) => {
    const calculateOverallPrediction = () => {
        // Combine technical indicators
        const technical_score = calculateTechnicalScore(data.technical_indicators);
        
        // News sentiment score
        const news_score = calculateNewsScore(data.news_analysis);
        
        // Pattern score
        const pattern_score = calculatePatternScore(data.patterns_detected);
        
        // Weighted average (40% technical, 40% news, 20% patterns)
        const overall_score = (
            0.4 * technical_score +
            0.4 * news_score +
            0.2 * pattern_score
        );

        return {
            direction: overall_score > 0 ? 'Bullish' : 'Bearish',
            confidence: Math.min(Math.abs(overall_score) * 20, 100), // Convert to percentage
            strength: Math.abs(overall_score)
        };
    };

    const calculateTechnicalScore = (indicators: any) => {
        let score = 0;
        
        // RSI
        if (indicators.RSI > 70) score -= 1;
        else if (indicators.RSI < 30) score += 1;
        
        // MACD
        if (indicators.MACD.histogram > 0) score += 0.5;
        else score -= 0.5;
        
        // Bollinger Bands
        const bb = indicators.BB;
        const current_price = data.market_data.prices[data.market_data.prices.length - 1];
        if (current_price > bb.upper) score -= 0.5;
        else if (current_price < bb.lower) score += 0.5;
        
        return score;
    };

    const calculateNewsScore = (news: any[]) => {
        if (!news.length) return 0;
        return news.reduce((acc, item) => acc + item.sentiment_score, 0) / news.length;
    };

    const calculatePatternScore = (patterns: any) => {
        let score = 0;
        if (patterns.double_top) score -= 1;
        if (patterns.double_bottom) score += 1;
        if (patterns.head_shoulders) score -= 0.5;
        return score;
    };

    const prediction = calculateOverallPrediction();

    return (
        <div className="prediction-summary">
            <div className={`prediction-direction ${prediction.direction.toLowerCase()}`}>
                {prediction.direction}
            </div>
            <div className="prediction-details">
                <div className="confidence">
                    Confidence: {prediction.confidence.toFixed(1)}%
                </div>
                <div className="strength">
                    Signal Strength: {prediction.strength.toFixed(2)}
                </div>
            </div>
        </div>
    );
}; 