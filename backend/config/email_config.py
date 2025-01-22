import os
from dotenv import load_dotenv

load_dotenv()

EMAIL_CONFIG = {
    'SMTP_SERVER': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'SMTP_PORT': int(os.getenv('SMTP_PORT', '587')),
    'ALERT_EMAIL': os.getenv('ALERT_EMAIL'),
    'ALERT_EMAIL_PASSWORD': os.getenv('ALERT_EMAIL_PASSWORD'),
    'ALERT_FREQUENCY': {
        'hourly': True,
        'daily': True,
        'weekly': True
    },
    'ALERT_THRESHOLDS': {
        'impact_score_min': 3,  # Minimum impact score for alerts
        'confidence_min': 70    # Minimum confidence percentage
    }
} 