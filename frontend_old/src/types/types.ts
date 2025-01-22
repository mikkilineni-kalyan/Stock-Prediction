export interface TimeFrame {
    startDate: Date;
    endDate: Date;
    label: string;
}

export interface PredictionData {
    symbol: string;
    currentPrice: number;
    predictions: number[];
    dates: string[];
    confidenceIntervals: {
        upper: number[];
        lower: number[];
    };
    metrics: {
        volatility: number;
        predictedChange: number;
        averageVolume: number;
    };
}

export interface SentimentData {
    score: number;
    impact: string;
    confidence: number;
    sources: number;
    summary: string;
    historical_correlation?: number;
    source_breakdown?: {
        [key: string]: {
            count: number;
            avg_score: number;
        };
    };
    historical_accuracy?: {
        accuracy: number;
        correlation: number;
        total_predictions: number;
        correct_predictions: number;
        average_confidence: number;
        recent_trend: {
            recent_accuracy: number;
            accuracy_trend: string;
        };
    };
}

export interface PriceRange {
    low: number;
    high: number;
}

export interface DailyPrediction {
    target_price: number;
    price_range: PriceRange;
    confidence: number;
    direction: string;
    expected_change_percent: number;
    days_ahead: number;
    is_trading_day: boolean;
    trading_hours?: [string, string];
}

export interface StockPrediction {
    ticker: string;
    company: string;
    current_price: number;
    historical_prices: number[];
    predictions: {
        [date: string]: {
            target_price: number;
            price_range: {
                low: number;
                high: number;
            };
            confidence: number;
            direction: string;
            expected_change_percent: number;
            days_ahead: number;
            is_trading_day: boolean;
            trading_hours?: [string, string];
        };
    };
    news_sentiment: {
        score: number;
        impact: string;
        confidence: number;
        sources: number;
        summary: string;
    };
    metadata: {
        last_updated: string;
        prediction_dates: string[];
        historical_dates: string[];
    };
}

export interface NewsSource {
  id: string;
  name: string;
  type: 'financial' | 'general' | 'sec' | 'social';
  reliability_score: number;
}

export interface APIError {
    error: string;
}
