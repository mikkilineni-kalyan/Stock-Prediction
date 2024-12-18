import React, { useEffect, useState } from 'react';
import { useParams, useSearchParams, Navigate } from 'react-router-dom';
import StockDashboard from './StockDashboard';

interface ValidationResponse {
    valid: boolean;
    name?: string;
    error?: string;
}

const DashboardWrapper: React.FC = () => {
    const { ticker } = useParams<{ ticker: string }>();
    const [searchParams] = useSearchParams();
    const [validation, setValidation] = useState<ValidationResponse | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const validateStock = async () => {
            if (!ticker) return;
            
            try {
                const response = await fetch(`/api/stocks/validate/${ticker}`);
                const data = await response.json();
                setValidation(data);
            } catch (error) {
                setValidation({ valid: false, error: 'Failed to validate stock' });
            } finally {
                setLoading(false);
            }
        };

        validateStock();
    }, [ticker]);

    if (loading) {
        return <div className="loading">Loading...</div>;
    }

    if (!ticker || !validation?.valid) {
        return <Navigate to="/" replace />;
    }

    return (
        <StockDashboard
            ticker={ticker.toUpperCase()}
            startDate={searchParams.get('startDate') || ''}
            endDate={searchParams.get('endDate') || ''}
            companyName={validation.name}
        />
    );
};

export default DashboardWrapper; 