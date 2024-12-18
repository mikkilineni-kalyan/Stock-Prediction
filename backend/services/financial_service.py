import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class FinancialService:
    def __init__(self):
        self.seeking_alpha_headers = {
            'X-RapidAPI-Key': os.getenv('SEEKING_ALPHA_API_KEY'),
            'X-RapidAPI-Host': 'seeking-alpha.p.rapidapi.com'
        }

    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive financial metrics for a stock"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            
            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            # Calculate key metrics
            metrics = self._calculate_financial_metrics(info, balance_sheet, income_stmt, cash_flow)
            
            # Get analyst recommendations
            recommendations = self._get_analyst_recommendations(ticker)
            
            # Get SEC filings
            sec_filings = self._get_sec_filings(ticker)
            
            return {
                'metrics': metrics,
                'recommendations': recommendations,
                'sec_filings': sec_filings,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting financial metrics: {str(e)}")
            return {}

    def _calculate_financial_metrics(
        self,
        info: Dict[str, Any],
        balance_sheet: pd.DataFrame,
        income_stmt: pd.DataFrame,
        cash_flow: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate key financial metrics"""
        try:
            # Current metrics
            market_cap = info.get('marketCap', 0)
            current_price = info.get('currentPrice', 0)
            
            # Valuation metrics
            pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
            pb_ratio = info.get('priceToBook', 0)
            ps_ratio = info.get('priceToSalesTrailing12Months', 0)
            
            # Growth metrics
            revenue_growth = self._calculate_growth_rate(income_stmt, 'Total Revenue')
            earnings_growth = self._calculate_growth_rate(income_stmt, 'Net Income')
            
            # Financial health metrics
            if not balance_sheet.empty and not income_stmt.empty:
                current_ratio = (
                    balance_sheet.loc['Total Current Assets', balance_sheet.columns[0]] /
                    balance_sheet.loc['Total Current Liabilities', balance_sheet.columns[0]]
                )
                debt_to_equity = (
                    balance_sheet.loc['Total Liabilities', balance_sheet.columns[0]] /
                    balance_sheet.loc['Total Stockholder Equity', balance_sheet.columns[0]]
                )
                roa = (
                    income_stmt.loc['Net Income', income_stmt.columns[0]] /
                    balance_sheet.loc['Total Assets', balance_sheet.columns[0]]
                )
                roe = (
                    income_stmt.loc['Net Income', income_stmt.columns[0]] /
                    balance_sheet.loc['Total Stockholder Equity', balance_sheet.columns[0]]
                )
            else:
                current_ratio = debt_to_equity = roa = roe = 0
            
            # Calculate intrinsic value using DCF
            intrinsic_value = self._calculate_intrinsic_value(income_stmt, cash_flow, info)
            
            return {
                'current_metrics': {
                    'market_cap': market_cap,
                    'current_price': current_price,
                    'intrinsic_value': intrinsic_value,
                    'value_difference': ((intrinsic_value - current_price) / current_price * 100) if current_price else 0
                },
                'valuation_metrics': {
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'ps_ratio': ps_ratio
                },
                'growth_metrics': {
                    'revenue_growth': revenue_growth,
                    'earnings_growth': earnings_growth
                },
                'health_metrics': {
                    'current_ratio': current_ratio,
                    'debt_to_equity': debt_to_equity,
                    'return_on_assets': roa,
                    'return_on_equity': roe
                }
            }
        except Exception as e:
            print(f"Error calculating financial metrics: {str(e)}")
            return {}

    def _calculate_growth_rate(self, df: pd.DataFrame, metric: str) -> float:
        """Calculate compound annual growth rate"""
        try:
            if metric not in df.index or df.empty:
                return 0
                
            values = df.loc[metric]
            if len(values) < 2:
                return 0
                
            start_value = values.iloc[-1]
            end_value = values.iloc[0]
            periods = len(values) - 1
            
            if start_value <= 0 or end_value <= 0:
                return 0
                
            return (pow(end_value / start_value, 1/periods) - 1) * 100
        except Exception as e:
            print(f"Error calculating growth rate: {str(e)}")
            return 0

    def _calculate_intrinsic_value(
        self,
        income_stmt: pd.DataFrame,
        cash_flow: pd.DataFrame,
        info: Dict[str, Any]
    ) -> float:
        """Calculate intrinsic value using DCF model"""
        try:
            if income_stmt.empty or cash_flow.empty:
                return 0
                
            # Get free cash flows
            fcf = cash_flow.loc['Free Cash Flow'] if 'Free Cash Flow' in cash_flow.index else cash_flow.loc['Operating Cash Flow'] - cash_flow.loc['Capital Expenditure']
            
            # Calculate average FCF growth rate
            fcf_growth_rate = self._calculate_growth_rate(pd.DataFrame(fcf), 'Free Cash Flow')
            
            # Use lower of historical growth rate or analyst growth rate
            growth_rate = min(
                fcf_growth_rate,
                info.get('earningsGrowth', 0) * 100
            )
            
            # Conservative growth rate between 2% and 15%
            growth_rate = max(2, min(15, growth_rate))
            
            # Terminal growth rate
            terminal_growth = 2  # 2% terminal growth
            
            # Discount rate (WACC or required return)
            discount_rate = max(8, info.get('returnOnEquity', 10))  # Minimum 8% discount rate
            
            # Latest FCF
            latest_fcf = fcf.iloc[0]
            
            # Project cash flows for 5 years
            projected_fcf = []
            for i in range(1, 6):
                projected_fcf.append(latest_fcf * (1 + growth_rate/100)**i)
            
            # Calculate terminal value
            terminal_value = (projected_fcf[-1] * (1 + terminal_growth/100)) / (discount_rate/100 - terminal_growth/100)
            
            # Calculate present value of projected cash flows
            present_value = sum([
                cf / (1 + discount_rate/100)**i
                for i, cf in enumerate(projected_fcf, 1)
            ])
            
            # Add present value of terminal value
            present_value += terminal_value / (1 + discount_rate/100)**5
            
            # Get shares outstanding
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            if shares_outstanding:
                return present_value / shares_outstanding
            return 0
            
        except Exception as e:
            print(f"Error calculating intrinsic value: {str(e)}")
            return 0

    def _get_analyst_recommendations(self, ticker: str) -> Dict[str, Any]:
        """Get analyst recommendations from Seeking Alpha"""
        try:
            url = f"https://seeking-alpha.p.rapidapi.com/symbols/{ticker}/recommendations"
            response = requests.get(url, headers=self.seeking_alpha_headers)
            data = response.json()
            
            return {
                'rating': data.get('rating', {}).get('rating', 'N/A'),
                'score': data.get('rating', {}).get('score', 0),
                'recommendations': data.get('recommendations', [])
            }
        except Exception as e:
            print(f"Error getting analyst recommendations: {str(e)}")
            return {}

    def _get_sec_filings(self, ticker: str) -> List[Dict[str, Any]]:
        """Get recent SEC filings"""
        try:
            url = f"https://api.sec-api.io?token={os.getenv('SEC_API_KEY')}"
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker} AND formType:(10-K OR 10-Q)"
                    }
                },
                "from": "0",
                "size": "10",
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            
            response = requests.post(url, json=query)
            data = response.json()
            
            return [
                {
                    'type': filing.get('formType'),
                    'filed_at': filing.get('filedAt'),
                    'url': filing.get('linkToFilingDetails')
                }
                for filing in data.get('filings', [])
            ]
        except Exception as e:
            print(f"Error getting SEC filings: {str(e)}")
            return []

    def get_undervalued_stocks(self, market: str = 'US') -> List[Dict[str, Any]]:
        """Find undervalued stocks based on various metrics"""
        try:
            # Get list of stocks
            if market == 'US':
                tickers = pd.read_csv('nasdaq-listed.csv')['Symbol'].tolist()
            else:
                return []
            
            undervalued_stocks = []
            
            for ticker in tickers[:100]:  # Limit initial analysis to first 100 stocks
                try:
                    metrics = self.get_financial_metrics(ticker)
                    
                    if not metrics:
                        continue
                        
                    current_metrics = metrics.get('current_metrics', {})
                    valuation_metrics = metrics.get('valuation_metrics', {})
                    growth_metrics = metrics.get('growth_metrics', {})
                    health_metrics = metrics.get('health_metrics', {})
                    
                    # Score the stock based on various factors
                    value_score = self._calculate_value_score(
                        current_metrics,
                        valuation_metrics,
                        growth_metrics,
                        health_metrics
                    )
                    
                    if value_score >= 7:  # Only include stocks with high value scores
                        undervalued_stocks.append({
                            'ticker': ticker,
                            'value_score': value_score,
                            'metrics': metrics
                        })
                        
                except Exception as e:
                    print(f"Error analyzing {ticker}: {str(e)}")
                    continue
            
            # Sort by value score
            undervalued_stocks.sort(key=lambda x: x['value_score'], reverse=True)
            
            return undervalued_stocks[:20]  # Return top 20 undervalued stocks
            
        except Exception as e:
            print(f"Error finding undervalued stocks: {str(e)}")
            return []

    def _calculate_value_score(
        self,
        current_metrics: Dict[str, Any],
        valuation_metrics: Dict[str, Any],
        growth_metrics: Dict[str, Any],
        health_metrics: Dict[str, Any]
    ) -> float:
        """Calculate a value score (0-10) based on various metrics"""
        score = 0
        
        # Value difference from intrinsic value (up to 3 points)
        value_diff = current_metrics.get('value_difference', 0)
        if value_diff > 20:
            score += 3
        elif value_diff > 10:
            score += 2
        elif value_diff > 0:
            score += 1
        
        # PE ratio (up to 2 points)
        pe = valuation_metrics.get('pe_ratio', 0)
        if 0 < pe < 15:
            score += 2
        elif 15 <= pe < 25:
            score += 1
        
        # Growth metrics (up to 2 points)
        revenue_growth = growth_metrics.get('revenue_growth', 0)
        earnings_growth = growth_metrics.get('earnings_growth', 0)
        if revenue_growth > 10 and earnings_growth > 10:
            score += 2
        elif revenue_growth > 5 and earnings_growth > 5:
            score += 1
        
        # Financial health (up to 3 points)
        if health_metrics.get('current_ratio', 0) > 1.5:
            score += 1
        if health_metrics.get('debt_to_equity', float('inf')) < 1:
            score += 1
        if health_metrics.get('return_on_equity', 0) > 15:
            score += 1
        
        return score
