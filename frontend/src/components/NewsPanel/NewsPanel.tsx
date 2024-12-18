import React from 'react';

interface NewsPanelProps {
    ticker: string;
}

const NewsPanel: React.FC<NewsPanelProps> = ({ ticker }) => {
    return (
        <div className="news-panel">
            <h3>News Analysis</h3>
            {/* News panel content */}
        </div>
    );
};

export default NewsPanel; 