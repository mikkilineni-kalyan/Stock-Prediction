import React from 'react';

interface SentimentData {
  newsScore?: number;
  socialScore?: number;
  analystRating?: string;
  recommendationTrend?: {
    strongBuy: number;
    buy: number;
    hold: number;
    sell: number;
    strongSell: number;
  };
}

const MarketSentiment: React.FC<{ sentiment?: SentimentData }> = ({ sentiment }) => {
  if (!sentiment) return null;

  const getSentimentColor = (score: number = 0.5) => {
    if (score >= 0.6) return '#2ecc71';
    if (score >= 0.4) return '#f1c40f';
    return '#e74c3c';
  };

  return (
    <div className="market-sentiment">
      <h3>Market Sentiment Analysis</h3>
      <div className="sentiment-grid">
        <div className="sentiment-card">
          <h4>News Sentiment</h4>
          <div 
            className="sentiment-score"
            style={{ color: getSentimentColor(sentiment.newsScore) }}
          >
            {((sentiment.newsScore || 0) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="sentiment-card">
          <h4>Social Media Sentiment</h4>
          <div 
            className="sentiment-score"
            style={{ color: getSentimentColor(sentiment.socialScore) }}
          >
            {((sentiment.socialScore || 0) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="sentiment-card">
          <h4>Analyst Rating</h4>
          <div className="analyst-rating">
            {sentiment.analystRating || 'N/A'}
          </div>
        </div>

        {sentiment.recommendationTrend && (
          <div className="sentiment-card">
            <h4>Recommendation Trend</h4>
            <div className="recommendation-bars">
              <div className="rec-bar">
                <span>Strong Buy</span>
                <div className="bar" style={{ width: `${sentiment.recommendationTrend.strongBuy * 20}%`, backgroundColor: '#27ae60' }}></div>
                <span>{sentiment.recommendationTrend.strongBuy}</span>
              </div>
              <div className="rec-bar">
                <span>Buy</span>
                <div className="bar" style={{ width: `${sentiment.recommendationTrend.buy * 20}%`, backgroundColor: '#2ecc71' }}></div>
                <span>{sentiment.recommendationTrend.buy}</span>
              </div>
              <div className="rec-bar">
                <span>Hold</span>
                <div className="bar" style={{ width: `${sentiment.recommendationTrend.hold * 20}%`, backgroundColor: '#f1c40f' }}></div>
                <span>{sentiment.recommendationTrend.hold}</span>
              </div>
              <div className="rec-bar">
                <span>Sell</span>
                <div className="bar" style={{ width: `${sentiment.recommendationTrend.sell * 20}%`, backgroundColor: '#e74c3c' }}></div>
                <span>{sentiment.recommendationTrend.sell}</span>
              </div>
              <div className="rec-bar">
                <span>Strong Sell</span>
                <div className="bar" style={{ width: `${sentiment.recommendationTrend.strongSell * 20}%`, backgroundColor: '#c0392b' }}></div>
                <span>{sentiment.recommendationTrend.strongSell}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketSentiment;
