import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar: React.FC = () => {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">Stock Predictor</Link>
      </div>
      <div className="navbar-links">
        <Link to="/" className={isHome ? 'active' : ''}>
          Search
        </Link>
        {!isHome && (
          <Link to="/" className="back-button">
            ← Back to Search
          </Link>
        )}
      </div>
    </nav>
  );
};

export default Navbar; 