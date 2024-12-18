import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict
import os
from jinja2 import Template

class StockAlertSystem:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = os.getenv('ALERT_EMAIL')
        self.sender_password = os.getenv('ALERT_EMAIL_PASSWORD')

    def send_hourly_alert(self, predictions: List[Dict], recipient_email: str) -> bool:
        """Send hourly stock alerts for significant predictions"""
        # Filter predictions with score >= 3
        significant_predictions = [p for p in predictions if p.get('impact_score', 0) >= 3]
        
        if not significant_predictions:
            return False

        # Create email content using template
        email_content = self._create_hourly_alert_content(significant_predictions)
        
        # Send email
        return self._send_email(
            recipient_email,
            "Hourly Stock Alert - High Impact Predictions",
            email_content
        )

    def send_daily_analysis(self, predictions: List[Dict], recipient_email: str) -> bool:
        """Send daily long-term analysis for undervalued stocks"""
        email_content = self._create_daily_analysis_content(predictions)
        
        return self._send_email(
            recipient_email,
            "Daily Stock Analysis Report",
            email_content
        )

    def send_weekly_report(self, predictions: List[Dict], recipient_email: str) -> bool:
        """Send weekly comprehensive stock analysis"""
        email_content = self._create_weekly_report_content(predictions)
        
        return self._send_email(
            recipient_email,
            "Weekly Stock Prediction Report",
            email_content
        )

    def _create_hourly_alert_content(self, predictions: List[Dict]) -> str:
        template = Template("""
        <h2>Stock Market Alerts</h2>
        <p>Below are the news highlights that are predicted to affect the market:</p>
        
        {% for pred in predictions %}
        <div style="margin-bottom: 15px;">
            <p><strong>{{ pred.ticker }}</strong> - {{ pred.impact_score }}/5 - 
               {{ pred.prediction.direction }} - ({{ pred.market_data.current_price }})</p>
            
            <p><strong>Summary:</strong> {{ pred.analysis }}</p>
            
            {% if pred.news_items %}
            <p><strong>Recent News:</strong></p>
            <ul>
                {% for news in pred.news_items[:3] %}
                <li>{{ news.title }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </div>
        {% endfor %}
        """)
        
        return template.render(predictions=predictions)

    def _create_daily_analysis_content(self, predictions: List[Dict]) -> str:
        template = Template("""
        <h2>Daily Stock Analysis Report</h2>
        <p>Analysis Date: {{ date }}</p>
        
        {% for pred in predictions %}
        <div style="margin-bottom: 20px;">
            <h3>{{ pred.ticker }} Analysis</h3>
            
            <p><strong>Current Price:</strong> {{ pred.market_data.current_price }}</p>
            <p><strong>Prediction:</strong> {{ pred.prediction.direction }} 
               (Confidence: {{ pred.prediction.confidence }}%)</p>
            
            <h4>Technical Analysis</h4>
            <p>{{ pred.analysis }}</p>
            
            <h4>News Impact</h4>
            <ul>
                {% for news in pred.news_items[:5] %}
                <li>{{ news.title }} (Impact: {{ news.impact_score }}/5)</li>
                {% endfor %}
            </ul>
        </div>
        {% endfor %}
        """)
        
        return template.render(predictions=predictions, date=datetime.now().strftime('%Y-%m-%d'))

    def _create_weekly_report_content(self, predictions: List[Dict]) -> str:
        template = Template("""
        <h2>Weekly Stock Prediction Report</h2>
        <p>Week of {{ date }}</p>
        
        {% for pred in predictions %}
        <div style="margin-bottom: 25px;">
            <h3>{{ pred.ticker }} - Weekly Forecast</h3>
            
            <div style="margin-left: 15px;">
                <p><strong>Current Status:</strong></p>
                <ul>
                    <li>Price: {{ pred.market_data.current_price }}</li>
                    <li>Weekly Change: {{ pred.market_data.price_change_percent }}%</li>
                    <li>Volume Ratio: {{ pred.market_data.volume_ratio }}</li>
                </ul>
                
                <p><strong>Prediction:</strong></p>
                <ul>
                    <li>Direction: {{ pred.prediction.direction }}</li>
                    <li>Confidence: {{ pred.prediction.confidence }}%</li>
                    <li>Impact Score: {{ pred.impact_score }}/5</li>
                </ul>
                
                <p><strong>Analysis:</strong> {{ pred.analysis }}</p>
                
                {% if pred.news_items %}
                <p><strong>Key News Drivers:</strong></p>
                <ul>
                    {% for news in pred.news_items[:3] %}
                    <li>{{ news.title }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
        </div>
        {% endfor %}
        """)
        
        return template.render(predictions=predictions, date=datetime.now().strftime('%Y-%m-%d'))

    def _send_email(self, recipient: str, subject: str, content: str) -> bool:
        """Send email with the given content"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = recipient

            # Attach HTML content
            html_part = MIMEText(content, 'html')
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            return True

        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False 