import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

class AlertService:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.use_tls = os.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
        self.email = os.getenv('EMAIL_ADDRESS', '')
        self.password = os.getenv('EMAIL_PASSWORD', '')

    def send_alert(self, to_email: str, subject: str, content: str):
        """Send an email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(content, 'html'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)

            print(f"Alert sent successfully to {to_email}")
            return True
        except Exception as e:
            print(f"Error sending alert: {str(e)}")
            return False

    def format_stock_alert(
        self,
        ticker: str,
        prediction: Dict[str, Any],
        news: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """Format stock alert email content"""
        try:
            # Current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Format prediction details
            pred_return = prediction.get('predicted_return', 0)
            confidence = prediction.get('confidence', 0)
            trend = prediction.get('trend', 'neutral')
            
            # Format news sentiment
            news_sentiment = news.get('sentiment', {})
            sentiment_score = news_sentiment.get('overall_score', 3)
            news_confidence = news_sentiment.get('confidence', 0)
            
            # Format key metrics
            current_price = metrics.get('current_metrics', {}).get('current_price', 0)
            intrinsic_value = metrics.get('current_metrics', {}).get('intrinsic_value', 0)
            value_diff = metrics.get('current_metrics', {}).get('value_difference', 0)
            
            # Create HTML content
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .neutral {{ color: gray; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f8f9fa; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>Stock Alert: {ticker}</h2>
                    <p>Generated at: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h3>Prediction Summary</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Expected Return</td>
                            <td class="{self._get_trend_class(pred_return)}">{pred_return:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Confidence</td>
                            <td>{confidence:.1f}%</td>
                        </tr>
                        <tr>
                            <td>Trend</td>
                            <td class="{trend.lower()}">{trend.upper()}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h3>News Sentiment</h3>
                    <table>
                        <tr>
                            <td>Overall Sentiment</td>
                            <td class="{self._get_sentiment_class(sentiment_score)}">
                                {self._format_sentiment(sentiment_score)}
                            </td>
                        </tr>
                        <tr>
                            <td>Confidence</td>
                            <td>{news_confidence:.1f}%</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h3>Valuation Metrics</h3>
                    <table>
                        <tr>
                            <td>Current Price</td>
                            <td>${current_price:.2f}</td>
                        </tr>
                        <tr>
                            <td>Intrinsic Value</td>
                            <td>${intrinsic_value:.2f}</td>
                        </tr>
                        <tr>
                            <td>Value Difference</td>
                            <td class="{self._get_trend_class(value_diff)}">
                                {value_diff:.1f}%
                            </td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h3>Latest News</h3>
                    <ul>
            """
            
            # Add latest news items
            for item in news.get('news', [])[:5]:
                pub_date = item['published_at'].strftime('%Y-%m-%d %H:%M')
                html += f"""
                    <li>
                        <strong>{item['title']}</strong><br>
                        <small>Source: {item['source']} | {pub_date}</small><br>
                        <a href="{item['url']}">Read more</a>
                    </li>
                """
            
            html += """
                    </ul>
                </div>
            </body>
            </html>
            """
            
            return html
        except Exception as e:
            print(f"Error formatting stock alert: {str(e)}")
            return ""

    def format_daily_report(
        self,
        predictions: List[Dict[str, Any]],
        undervalued_stocks: List[Dict[str, Any]]
    ) -> str:
        """Format daily stock report email content"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .neutral {{ color: gray; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f8f9fa; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>Daily Stock Market Report</h2>
                    <p>Generated at: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h3>Top Stock Predictions</h3>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th>Expected Return</th>
                            <th>Confidence</th>
                            <th>News Sentiment</th>
                        </tr>
            """
            
            # Add predictions
            for pred in sorted(predictions, key=lambda x: abs(x['predicted_return']), reverse=True)[:10]:
                html += f"""
                    <tr>
                        <td>{pred['ticker']}</td>
                        <td class="{self._get_trend_class(pred['predicted_return'])}">
                            {pred['predicted_return']:.2f}%
                        </td>
                        <td>{pred['confidence']:.1f}%</td>
                        <td class="{self._get_sentiment_class(pred['news_sentiment'])}">
                            {self._format_sentiment(pred['news_sentiment'])}
                        </td>
                    </tr>
                """
            
            html += """
                    </table>
                </div>
                
                <div class="section">
                    <h3>Undervalued Stocks</h3>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th>Value Score</th>
                            <th>Price Difference</th>
                            <th>Growth Potential</th>
                        </tr>
            """
            
            # Add undervalued stocks
            for stock in undervalued_stocks[:10]:
                metrics = stock['metrics']
                value_diff = metrics['current_metrics']['value_difference']
                growth = metrics['growth_metrics']['earnings_growth']
                
                html += f"""
                    <tr>
                        <td>{stock['ticker']}</td>
                        <td>{stock['value_score']:.1f}/10</td>
                        <td class="{self._get_trend_class(value_diff)}">
                            {value_diff:.1f}%
                        </td>
                        <td class="{self._get_trend_class(growth)}">
                            {growth:.1f}%
                        </td>
                    </tr>
                """
            
            html += """
                    </table>
                </div>
            </body>
            </html>
            """
            
            return html
        except Exception as e:
            print(f"Error formatting daily report: {str(e)}")
            return ""

    def _get_trend_class(self, value: float) -> str:
        """Get CSS class based on trend value"""
        if value > 0:
            return 'positive'
        elif value < 0:
            return 'negative'
        return 'neutral'

    def _get_sentiment_class(self, score: float) -> str:
        """Get CSS class based on sentiment score"""
        if score > 3.5:
            return 'positive'
        elif score < 2.5:
            return 'negative'
        return 'neutral'

    def _format_sentiment(self, score: float) -> str:
        """Format sentiment score as text"""
        if score > 4:
            return 'Very Positive'
        elif score > 3.5:
            return 'Positive'
        elif score > 2.5:
            return 'Neutral'
        elif score > 1.5:
            return 'Negative'
        return 'Very Negative'
