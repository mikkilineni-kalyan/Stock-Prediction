import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './Navbar/Navbar';
import StockSearch from './StockSearch';
import DashboardWrapper from './Dashboard/DashboardWrapper';

const StockPredictor: React.FC = () => {
    return (
        <Router>
            <div className="app">
                <Navbar />
                <main className="main-content">
                    <Routes>
                        <Route path="/" element={<StockSearch />} />
                        <Route path="/dashboard/:ticker" element={<DashboardWrapper />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
};

export default StockPredictor; 