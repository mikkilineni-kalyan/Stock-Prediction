import React from 'react';
import './PatternRecognition.css';

interface Pattern {
    name: string;
    confidence: number;
    direction: string;
}

interface PatternDescriptions {
    [key: string]: string;
}

export const PatternRecognition: React.FC<{ patterns: Pattern[] }> = ({ patterns }) => {
    const getPatternDescription = (pattern: string): string => {
        const descriptions: PatternDescriptions = {
            double_top: "Bearish reversal pattern indicating potential downtrend",
            double_bottom: "Bullish reversal pattern indicating potential uptrend",
            head_shoulders: "Bearish reversal pattern with high reliability",
            triangle: "Continuation pattern indicating potential breakout"
        };
        
        return descriptions[pattern] || "Pattern description not available";
    };

    return (
        <div className="pattern-recognition">
            {patterns.map((pattern) => (
                <div key={pattern.name} className="pattern-item">
                    <div className="pattern-header">
                        <span className="pattern-name">
                            {pattern.name.toUpperCase()}
                        </span>
                        <span className="pattern-status">
                            {pattern.confidence > 0.5 ? '✓' : '×'}
                        </span>
                    </div>
                    <p className="pattern-description">
                        {getPatternDescription(pattern.name)}
                    </p>
                </div>
            ))}
        </div>
    );
}; 