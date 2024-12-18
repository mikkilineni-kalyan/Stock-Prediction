import React from 'react';
import { Line } from 'react-chartjs-2';
import { SentimentData } from '../types/types';
import './SentimentAnalysis.css';

interface SentimentAnalysisProps {
    sentimentData: SentimentData;
}

export const SentimentAnalysis: React.FC<SentimentAnalysisProps> = ({ sentimentData }) => {
    const sourceData = {
        labels: Object.keys(sentimentData.source_breakdown || {}),
        datasets: [
            {
                label: 'Average Sentiment Score',
                data: Object.values(sentimentData.source_breakdown || {}).map(s => s.avg_score),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            },
            {
                label: 'Number of Sources',
                data: Object.values(sentimentData.source_breakdown || {}).map(s => s.count),
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }
        ]
    };

    return (
        <div className="sentiment-analysis">
            <div className="sentiment-summary">
                <h3>Sentiment Analysis</h3>
                <div className={`sentiment-score ${sentimentData.impact}`}>
                    Score: {sentimentData.score.toFixed(2)}/5
                </div>
                <div className="confidence">
                    Confidence: {(sentimentData.confidence * 100).toFixed(1)}%
                </div>
                {sentimentData.historical_accuracy && (
                    <div className="historical-accuracy">
                        Historical Accuracy: {(sentimentData.historical_accuracy.accuracy * 100).toFixed(1)}%
                    </div>
                )}
            </div>

            {sentimentData.source_breakdown && (
                <div className="source-breakdown">
                    <h4>Source Breakdown</h4>
                    <div style={{ height: '300px' }}>
                        <Line
                            data={sourceData}
                            options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        beginAtZero: true
                                    }
                                }
                            }}
                        />
                    </div>
                </div>
            )}

            {sentimentData.historical_accuracy && (
                <div className="historical-stats">
                    <h4>Historical Performance</h4>
                    <div className="stats-grid">
                        <div className="stat-item">
                            <label>Total Predictions</label>
                            <span className="value">
                                {sentimentData.historical_accuracy.total_predictions}
                            </span>
                        </div>
                        <div className="stat-item">
                            <label>Correct Predictions</label>
                            <span className="value">
                                {sentimentData.historical_accuracy.correct_predictions}
                            </span>
                        </div>
                        <div className="stat-item">
                            <label>Recent Trend</label>
                            <span className="value">
                                {sentimentData.historical_accuracy.recent_trend.accuracy_trend}
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}; 