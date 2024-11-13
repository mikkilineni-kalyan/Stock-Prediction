import React from 'react';
import ErrorBoundary from './components/ErrorBoundary';
import StockPredictor from './components/StockPredictor';
import { ThemeProvider } from './contexts/ThemeContext';

function App() {
  return (
    <ThemeProvider>
      <ErrorBoundary>
        <StockPredictor />
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App;