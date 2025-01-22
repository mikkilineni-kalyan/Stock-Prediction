import React from 'react';
import './NewsPanel.css';

interface NewsItem {
    title: string;
    source: string;
    published: string;
    url: string;
    sentiment_score: number;
    impact_score: number;
}

interface NewsPanelProps {
    news: NewsItem[];
}

export const NewsPanel: React.FC<NewsPanelProps> = ({ news }) => {
    const getSentimentColor = (score: number): string => {
        if (score > 0.2) return '#4CAF50';
        if (score < -0.2) return '#f44336';
        return '#FF9800';
    };

    return (
        <div className="news-panel">
            {news.map((item, index) => (
                <div key={index} className="news-item">
                    <div className="news-header">
                        <span className="news-source">{item.source}</span>
                        <span className="news-date">
                            {new Date(item.published).toLocaleDateString()}
                        </span>
                    </div>
                    <a href={item.url} target="_blank" rel="noopener noreferrer" 
                       className="news-title">
                        {item.title}
                    </a>
                    <div className="news-metrics">
                        <div className="sentiment-score" 
                             style={{ color: getSentimentColor(item.sentiment_score) }}>
                            Sentiment: {item.sentiment_score.toFixed(2)}
                        </div>
                        <div className="impact-score">
                            Impact: {item.impact_score}/5
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}; 