import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './StockSearch.css';

const StockSearch: React.FC = () => {
    const [ticker, setTicker] = useState('');
    const navigate = useNavigate();

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (ticker) {
            navigate(`/dashboard/${ticker.toUpperCase()}`);
        }
    };

    return (
        <div className="stock-search">
            <h1>Stock Price Predictor</h1>
            <form onSubmit={handleSubmit} className="search-form">
                <input
                    type="text"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    placeholder="Enter stock ticker (e.g., AAPL)"
                    className="search-input"
                />
                <button type="submit" className="search-button">
                    Search
                </button>
            </form>
        </div>
    );
};

export default StockSearch; 