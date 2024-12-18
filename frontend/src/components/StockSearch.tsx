import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import './StockSearch.css';

interface StockSuggestion {
    symbol: string;
    name: string;
}

interface SearchCache {
    [key: string]: {
        data: StockSuggestion[];
        timestamp: number;
    };
}

interface ValidationResponse {
    valid: boolean;
    name?: string;
    exchange?: string;
    currency?: string;
    error?: string;
    details?: string;
}

const StockSearch: React.FC = () => {
    const navigate = useNavigate();
    const [ticker, setTicker] = useState('');
    const [suggestions, setSuggestions] = useState<StockSuggestion[]>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const searchContainerRef = useRef<HTMLDivElement>(null);
    const searchCache = useRef<SearchCache>({});
    const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

    // Date state with validation
    const today = new Date().toISOString().split('T')[0];
    const [startDate, setStartDate] = useState(today);
    const [endDate, setEndDate] = useState(
        new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
    );

    const popularStocks = [
        { symbol: 'AAPL', name: 'Apple Inc.' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.' },
        { symbol: 'MSFT', name: 'Microsoft Corporation' },
        { symbol: 'AMZN', name: 'Amazon.com Inc.' },
        { symbol: 'TSLA', name: 'Tesla, Inc.' }
    ];

    // Memoized fetch function
    const fetchSuggestionsFromAPI = useCallback(async (query: string) => {
        if (query.length < 1) {
            setSuggestions([]);
            setShowSuggestions(false);
            return;
        }
        
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`http://localhost:5000/api/search/stocks?q=${encodeURIComponent(query)}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (Array.isArray(data)) {
                // Transform the data to match StockSuggestion interface
                const formattedData: StockSuggestion[] = data.map(item => ({
                    symbol: item.Symbol,
                    name: item.Name
                }));
                
                setSuggestions(formattedData);
                setShowSuggestions(formattedData.length > 0);
            } else if (data.error) {
                throw new Error(data.error);
            } else {
                setSuggestions([]);
                setShowSuggestions(false);
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            setError('Failed to fetch stock suggestions');
            // Fallback to popular stocks if query matches
            const fallbackResults = popularStocks.filter(stock => 
                stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
                stock.name.toLowerCase().includes(query.toLowerCase())
            );
            setSuggestions(fallbackResults);
            setShowSuggestions(fallbackResults.length > 0);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) {
                setShowSuggestions(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        if (ticker.length >= 1) {
            const timeoutId = setTimeout(() => {
                fetchSuggestionsFromAPI(ticker);
                setShowSuggestions(true);
            }, 300);
            return () => clearTimeout(timeoutId);
        } else {
            setSuggestions([]);
            setShowSuggestions(false);
        }
    }, [ticker, fetchSuggestionsFromAPI]);

    useEffect(() => {
        console.log('Suggestions:', suggestions);
        console.log('Show suggestions:', showSuggestions);
    }, [suggestions, showSuggestions]);

    const validateDates = () => {
        const start = new Date(startDate);
        const end = new Date(endDate);
        const today = new Date();

        if (start > end) {
            setError('Start date cannot be after end date');
            return false;
        }

        if (end > new Date(today.getTime() + 30 * 24 * 60 * 60 * 1000)) {
            setError('Cannot predict more than 30 days in the future');
            return false;
        }

        return true;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setIsLoading(true);

        try {
            if (!ticker) {
                throw new Error('Please enter a stock ticker');
            }

            if (!validateDates()) {
                return;
            }

            // Validate stock ticker
            const response = await fetch(`/api/stocks/validate/${ticker.toUpperCase()}`);
            const validation: ValidationResponse = await response.json();

            if (!validation.valid) {
                throw new Error(validation.error || 'Invalid stock ticker');
            }

            navigate(`/dashboard/${ticker.toUpperCase()}?start=${startDate}&end=${endDate}`);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An error occurred');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="stock-search">
            <h1>Stock Price Predictor</h1>
            {error && <div className="error-message">{error}</div>}
            <form onSubmit={handleSubmit}>
                <div className="search-container" ref={searchContainerRef}>
                    <div className="search-input-container">
                        <input
                            type="text"
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value.toUpperCase())}
                            placeholder="Enter stock ticker (e.g., AAPL)"
                            onFocus={() => setShowSuggestions(true)}
                            className={error ? 'error' : ''}
                        />
                        {isLoading && <div className="loading-spinner" />}
                        {showSuggestions && suggestions.length > 0 && (
                            <div className="suggestions-dropdown">
                                {suggestions.map((stock) => (
                                    <div
                                        key={stock.symbol}
                                        className="suggestion-item"
                                        onClick={() => {
                                            setTicker(stock.symbol);
                                            setShowSuggestions(false);
                                            setError(null);
                                        }}
                                    >
                                        <span className="symbol">{stock.symbol}</span>
                                        <span className="name">{stock.name}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                    <button type="submit" disabled={isLoading}>
                        {isLoading ? 'Loading...' : 'Predict'}
                    </button>
                </div>

                <div className="date-inputs">
                    <div>
                        <label>Start Date:</label>
                        <input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            required
                        />
                    </div>
                    <div>
                        <label>End Date:</label>
                        <input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            required
                        />
                    </div>
                </div>

                <div className="popular-stocks">
                    <h3>Popular Stocks:</h3>
                    <div className="stock-buttons">
                        {popularStocks.map((stock) => (
                            <button
                                key={stock.symbol}
                                type="button"
                                onClick={() => {
                                    setTicker(stock.symbol);
                                    setShowSuggestions(false);
                                    setError(null);
                                }}
                            >
                                {stock.symbol}
                            </button>
                        ))}
                    </div>
                </div>
            </form>
        </div>
    );
};

export default StockSearch; 