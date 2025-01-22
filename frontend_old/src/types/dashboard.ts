export interface DashboardProps {
    ticker: string;
    startDate: string;
    endDate: string;
    companyName?: string;
}

export interface ValidationResponse {
    valid: boolean;
    name?: string;
    error?: string;
}

export interface Pattern {
    name: string;
    confidence: number;
    direction: string;
    start_idx: number;
    end_idx: number;
}

export interface StockData {
    prices: number[];
    dates: string[];
    volumes: number[];
    patterns: Pattern[];
    indicators: {
        rsi: number[];
        macd: {
            line: number[];
            signal: number[];
            histogram: number[];
        };
        bollinger: {
            upper: number[];
            middle: number[];
            lower: number[];
        };
    };
} 