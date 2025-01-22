import React, { useEffect, useState } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import StockDashboard from './StockDashboard';
import './DashboardWrapper.css';

interface ValidationResponse {
    valid: boolean;
    name?: string;
    error?: string;
}

const DashboardWrapper: React.FC = () => {
    const { ticker } = useParams<{ ticker: string }>();
    const [validation, setValidation] = useState<ValidationResponse | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const validateStock = async () => {
            if (!ticker) return;
            
            try {
                // First check if the stock exists in our database
                const response = await fetch(`http://localhost:5000/api/search/stocks?q=${ticker}`);
                if (!response.ok) {
                    throw new Error('Failed to validate stock');
                }
                
                const data = await response.json();
                const stockExists = data.some((stock: any) => stock.Symbol === ticker.toUpperCase());
                
                if (stockExists) {
                    setValidation({ valid: true });
                } else {
                    setValidation({ valid: false, error: 'Stock not found' });
                }
            } catch (error) {
                console.error('Error validating stock:', error);
                setValidation({ valid: false, error: 'Failed to validate stock' });
            } finally {
                setLoading(false);
            }
        };

        validateStock();
    }, [ticker]);

    if (loading) {
        return (
            <div className="loading-container">
                <div className="loading">Validating stock symbol...</div>
            </div>
        );
    }

    if (!ticker || !validation?.valid) {
        console.log('Invalid stock, redirecting to home');
        return <Navigate to="/" replace />;
    }

    console.log('Rendering StockDashboard for ticker:', ticker);
    return (
        <div className="dashboard-container">
            <StockDashboard ticker={ticker.toUpperCase()} />
        </div>
    );
};

export default DashboardWrapper;