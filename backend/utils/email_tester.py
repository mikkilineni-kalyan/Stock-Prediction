import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_email_config():
    """Test email configuration by sending a test email."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get email configuration
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_username = os.getenv('SMTP_USERNAME')
        smtp_password = os.getenv('SMTP_PASSWORD')
        use_tls = os.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
        
        # Validate configuration
        if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
            raise ValueError("Missing email configuration. Please check your .env file.")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = smtp_username  # Send to self for testing
        msg['Subject'] = 'Stock Prediction System - Email Test'
        
        body = """
        This is a test email from your Stock Prediction System.
        
        If you received this email, your email configuration is working correctly.
        
        Configuration Details:
        - SMTP Server: {server}
        - SMTP Port: {port}
        - TLS Enabled: {tls}
        - Username: {username}
        """.format(
            server=smtp_server,
            port=smtp_port,
            tls=use_tls,
            username=smtp_username
        )
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server
        logger.info(f"Connecting to SMTP server {smtp_server}:{smtp_port}")
        server = smtplib.SMTP(smtp_server, smtp_port)
        
        if use_tls:
            logger.info("Enabling TLS")
            server.starttls()
        
        # Login
        logger.info("Logging in to SMTP server")
        server.login(smtp_username, smtp_password)
        
        # Send email
        logger.info("Sending test email")
        server.send_message(msg)
        
        # Close connection
        server.quit()
        
        logger.info("Test email sent successfully!")
        return True, "Email configuration test successful"
        
    except Exception as e:
        error_msg = f"Email configuration test failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

if __name__ == '__main__':
    success, message = test_email_config()
    print(message)
