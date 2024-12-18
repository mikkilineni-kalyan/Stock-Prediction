import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import StockSearch from './components/StockSearch/StockSearch';
import StockDashboard from './components/StockDashboard/StockDashboard';
import './App.css';

function App() {
  console.log('App component rendering'); // Debug log
  
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Stock Predictor</h1>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<StockSearch />} />
            <Route path="/dashboard/:symbol" element={<StockDashboard />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;