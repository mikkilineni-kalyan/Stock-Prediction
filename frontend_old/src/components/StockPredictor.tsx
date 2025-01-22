import React from 'react';
import Navbar from './Navbar/Navbar';
import StockSearch from './StockSearch/StockSearch';
import DashboardWrapper from './Dashboard/DashboardWrapper';
import './StockPredictor.css';

const StockPredictor: React.FC = () => {
    console.log('StockPredictor component rendering'); // Debug log
    
    return (
        <div className="app">
            <Navbar />
            <main className="main-content">
                <StockSearch />
            </main>
        </div>
    );
};

export default StockPredictor;