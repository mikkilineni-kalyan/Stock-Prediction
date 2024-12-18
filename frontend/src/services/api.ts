import axios from 'axios';
import { StockPrediction } from '../types/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json'
    }
});

// Add response interceptor for better error handling
api.interceptors.response.use(
    response => response,
    (error) => {
        console.error('API Error:', error);
        if (error.response?.data?.error) {
            throw new Error(error.response.data.error);
        }
        throw new Error('Failed to fetch data from server');
    }
);

export const stockApi = {
    async getPrediction(ticker: string, startDate?: Date | null, endDate?: Date | null): Promise<StockPrediction> {
        try {
            console.log('Fetching prediction for:', ticker, startDate, endDate);
            let url = `/api/predict/${ticker}`;
            
            // Add date parameters if provided
            const params = new URLSearchParams();
            if (startDate) {
                params.append('start_date', startDate.toISOString().split('T')[0]);
            }
            if (endDate) {
                params.append('end_date', endDate.toISOString().split('T')[0]);
            }

            const queryString = params.toString();
            if (queryString) {
                url += `?${queryString}`;
            }

            console.log('Making request to:', url);
            const response = await api.get<StockPrediction>(url);
            console.log('API Response:', response.data);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            if (error instanceof Error) {
                throw error;
            }
            throw new Error('An unknown error occurred');
        }
    }
}; 