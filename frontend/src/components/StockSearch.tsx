import React, { useState, useEffect, useRef } from 'react';
import './StockSearch.css';

interface StockSearchProps {
  onSelect: (symbol: string) => void;
  initialValue?: string;
}

interface StockSuggestion {
  symbol: string;
  name: string;
  type: string;
}

export const StockSearch: React.FC<StockSearchProps> = ({ onSelect, initialValue = '' }) => {
  const [searchTerm, setSearchTerm] = useState(initialValue);
  const [suggestions, setSuggestions] = useState<StockSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const searchStocks = async (query: string) => {
    setIsLoading(true);
    try {
      const response = await fetch(
        `http://localhost:5000/api/search-stocks?q=${encodeURIComponent(query)}`
      );
      if (!response.ok) throw new Error('Search failed');
      const data = await response.json();
      setSuggestions(data);
      setShowDropdown(true);
    } catch (error) {
      console.error('Search error:', error);
      setSuggestions([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Debounce search
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchTerm) {
        searchStocks(searchTerm);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchTerm]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setSearchTerm(value);
    if (!value) {
      setSuggestions([]);
      setShowDropdown(false);
    }
  };

  const handleSelect = (suggestion: StockSuggestion) => {
    setSearchTerm(suggestion.symbol);
    onSelect(suggestion.symbol);
    setShowDropdown(false);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="stock-search-container" ref={dropdownRef}>
      <div className="search-input-wrapper">
        <input
          type="text"
          value={searchTerm}
          onChange={handleInputChange}
          onFocus={() => {
            if (searchTerm) {
              searchStocks(searchTerm);
            }
          }}
          placeholder="Search stocks (e.g., TSLA, Tesla, AAPL)"
          className="stock-search-input"
        />
        {isLoading && <div className="search-spinner" />}
      </div>

      {showDropdown && suggestions.length > 0 && (
        <div className="stock-suggestions-dropdown">
          {suggestions.map((suggestion) => (
            <div
              key={suggestion.symbol}
              className="suggestion-item"
              onClick={() => handleSelect(suggestion)}
            >
              <div className="suggestion-symbol">{suggestion.symbol}</div>
              <div className="suggestion-name">{suggestion.name}</div>
              <div className="suggestion-type">{suggestion.type}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
