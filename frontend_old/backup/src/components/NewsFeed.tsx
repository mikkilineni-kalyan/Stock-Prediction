import React from 'react';

interface NewsItem {
  title: string;
  summary: string;
  url: string;
  source: string;
  publishedAt: string;
}

interface NewsFeedProps {
  symbol: string;
  news: NewsItem[];
}

const NewsFeed: React.FC<NewsFeedProps> = ({ symbol, news }) => (
  <div className="news-feed">
    <h3>Latest News for {symbol}</h3>
    <div className="news-list">
      {news.map((item, index) => (
        <div key={index} className="news-item">
          <h4>{item.title}</h4>
          <p>{item.summary}</p>
          <div className="news-meta">
            <span>{item.source}</span>
            <span>{new Date(item.publishedAt).toLocaleDateString()}</span>
          </div>
          <a href={item.url} target="_blank" rel="noopener noreferrer">
            Read More
          </a>
        </div>
      ))}
    </div>
  </div>
);

export default NewsFeed;
