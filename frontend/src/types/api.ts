export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface StockAPIResponse {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  historicalData: {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
  technicalIndicators: {
    rsi: number[];
    macd: number[];
    signal: number[];
    sma: number[];
    ema: number[];
  };
  prediction: {
    next_day: number;
    confidence: number;
    trend: string;
    predicted_return: number;
  };
  news_sentiment: {
    score: number;
    impact: string;
    confidence: number;
    sources: number;
    summary: string;
  };
}
