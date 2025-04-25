import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import url_for

logger = logging.getLogger(__name__)

# Email configuration
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USERNAME)
APP_NAME = "Trading Bot"

def send_email(to_email, subject, html_content, text_content=None):
    """
    Send an email
    
    Args:
        to_email (str): Recipient email address
        subject (str): Email subject
        html_content (str): HTML content of the email
        text_content (str, optional): Plain text content of the email
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logger.error("SMTP credentials not configured")
        return False
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = f"{APP_NAME} <{FROM_EMAIL}>"
    msg['To'] = to_email
    
    # Add text content if provided, otherwise create a simple version from HTML
    if text_content is None:
        text_content = f"Please view this email in a modern email client. {subject}"
    
    msg.attach(MIMEText(text_content, 'plain'))
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email sent to {to_email}: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

def send_password_reset_email(email, token):
    """
    Send a password reset email
    
    Args:
        email (str): Recipient email address
        token (str): Password reset token
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Generate the reset URL - this assumes the app is running on port 5000
    reset_url = url_for('reset_password', token=token, _external=True)
    
    subject = f"{APP_NAME} - Password Reset"
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px;">Password Reset Request</h2>
            <p>You requested a password reset for your {APP_NAME} account.</p>
            <p>Please click the button below to reset your password:</p>
            <p style="text-align: center;">
                <a href="{reset_url}" style="display: inline-block; background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">Reset Password</a>
            </p>
            <p>Or copy and paste this link in your browser:</p>
            <p style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; word-break: break-all;">
                {reset_url}
            </p>
            <p>This link will expire in 1 hour.</p>
            <p>If you did not request a password reset, please ignore this email or contact support if you have concerns.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="color: #7f8c8d; font-size: 12px; text-align: center;">
                &copy; {APP_NAME}. All rights reserved.
            </p>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    Password Reset Request
    
    You requested a password reset for your {APP_NAME} account.
    
    Please visit the following link to reset your password:
    
    {reset_url}
    
    This link will expire in 1 hour.
    
    If you did not request a password reset, please ignore this email or contact support if you have concerns.
    """
    
    return send_email(email, subject, html_content, text_content)

def send_notification_email(email, notification_title, notification_message):
    """
    Send a notification email
    
    Args:
        email (str): Recipient email address
        notification_title (str): Notification title
        notification_message (str): Notification message
        
    Returns:
        bool: True if successful, False otherwise
    """
    subject = f"{APP_NAME} - {notification_title}"
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px;">{notification_title}</h2>
            <p>{notification_message}</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="color: #7f8c8d; font-size: 12px; text-align: center;">
                &copy; {APP_NAME}. All rights reserved.
            </p>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    {notification_title}
    
    {notification_message}
    """
    
    return send_email(email, subject, html_content, text_content)
