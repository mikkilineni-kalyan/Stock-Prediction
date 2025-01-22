import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './StockSearch.css';

interface StockSuggestion {
    Symbol: string;
    Name: string;
    SentimentScore: number;
    Prediction: string;
}

const StockSearch: React.FC = () => {
    const [ticker, setTicker] = useState('');
    const [suggestions, setSuggestions] = useState<StockSuggestion[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchSuggestions = async () => {
            if (ticker.length < 1) {
                setSuggestions([]);
                return;
            }

            setLoading(true);
            setError(null);

            try {
                console.log('Fetching suggestions for:', ticker);
                const response = await fetch(`/api/stocks/search?q=${encodeURIComponent(ticker)}`);
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    throw new Error('Invalid response format: Expected JSON');
                }
                
                const data = await response.json();
                console.log('Data received:', data);
                
                if (Array.isArray(data)) {
                    setSuggestions(data);
                } else {
                    throw new Error('Invalid response format: Expected array');
                }
            } catch (err) {
                console.error('Error fetching suggestions:', err);
                setError(err instanceof Error ? err.message : 'Failed to fetch suggestions');
                setSuggestions([]);
            } finally {
                setLoading(false);
            }
        };

        const debounceTimer = setTimeout(fetchSuggestions, 300);
        return () => clearTimeout(debounceTimer);
    }, [ticker]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (ticker) {
            navigate(`/dashboard/${ticker.toUpperCase()}`);
        }
    };

    const handleSuggestionClick = (symbol: string) => {
        setTicker(symbol);
        setSuggestions([]);
        navigate(`/dashboard/${symbol}`);
    };

    const getSentimentColor = (score: number) => {
        if (score >= 4.0) return 'green';
        if (score >= 3.0) return 'orange';
        return 'red';
    };

    return (
        <div className="stock-search">
            <h1>Stock Price Predictor</h1>
            <div className="search-container">
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
                
                {loading && <div className="suggestions-loading">Loading...</div>}
                {error && <div className="suggestions-error" style={{color: 'red'}}>{error}</div>}
                
                {suggestions.length > 0 && (
                    <div className="suggestions-list">
                        {suggestions.map((stock) => (
                            <div
                                key={stock.Symbol}
                                className="suggestion-item"
                                onClick={() => handleSuggestionClick(stock.Symbol)}
                            >
                                <div className="suggestion-main">
                                    <span className="suggestion-symbol">{stock.Symbol}</span>
                                    <span className="suggestion-name">{stock.Name}</span>
                                </div>
                                <div className="suggestion-sentiment">
                                    <span 
                                        className="sentiment-score"
                                        style={{ color: getSentimentColor(stock.SentimentScore || 0) }}
                                    >
                                        Score: {(stock.SentimentScore || 0).toFixed(1)}
                                    </span>
                                    <span className="prediction">{stock.Prediction || 'N/A'}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default StockSearch;