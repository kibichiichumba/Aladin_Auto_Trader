import os
import logging
import requests
from datetime import datetime
from app import db
from models import Notification, User
from utils.auth import send_notification_email

logger = logging.getLogger(__name__)

# Telegram API URL
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

def send_notification(user_id, title, message, notification_type="info", related_trade_id=None):
    """
    Send a notification to a user through the platform and external channels
    
    Args:
        user_id (int): User ID
        title (str): Notification title
        message (str): Notification message
        notification_type (str): Notification type (info, trade, error, risk)
        related_trade_id (int, optional): ID of related trade
        
    Returns:
        bool: True if all notifications were sent successfully
    """
    logger.info(f"Sending notification to user {user_id}: {title}")
    
    # Create notification in database
    try:
        notification = Notification(
            user_id=user_id,
            title=title,
            message=message,
            type=notification_type,
            related_trade_id=related_trade_id
        )
        db.session.add(notification)
        db.session.commit()
    except Exception as e:
        logger.error(f"Error creating notification record: {e}")
        db.session.rollback()
        return False
    
    # Get user settings
    from models import UserSettings
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    
    if not settings or not settings.enable_notifications:
        logger.info(f"Notifications disabled for user {user_id}")
        return True
    
    # Get user email
    user = User.query.get(user_id)
    
    success = True
    
    # Send email notification
    try:
        if user and user.email:
            send_notification_email(user.email, title, message)
    except Exception as e:
        logger.error(f"Error sending email notification: {e}")
        success = False
    
    # Send Telegram notification
    if settings.telegram_chat_id:
        try:
            send_telegram_notification(settings.telegram_chat_id, title, message)
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            success = False
    
    # Send Discord notification
    if settings.discord_webhook:
        try:
            send_discord_notification(settings.discord_webhook, title, message, notification_type)
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            success = False
    
    return success

def send_telegram_notification(chat_id, title, message):
    """
    Send a notification via Telegram
    
    Args:
        chat_id (str): Telegram chat ID
        title (str): Notification title
        message (str): Notification message
        
    Returns:
        bool: True if successful, False otherwise
    """
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not telegram_token:
        logger.warning("Telegram bot token not configured")
        return False
    
    formatted_message = f"*{title}*\n\n{message}"
    
    try:
        response = requests.post(
            TELEGRAM_API_URL.format(token=telegram_token),
            json={
                "chat_id": chat_id,
                "text": formatted_message,
                "parse_mode": "Markdown"
            }
        )
        
        if response.status_code == 200:
            logger.info(f"Telegram notification sent to chat ID {chat_id}")
            return True
        else:
            logger.error(f"Error sending Telegram notification: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Exception sending Telegram notification: {e}")
        return False

def send_discord_notification(webhook_url, title, message, notification_type="info"):
    """
    Send a notification via Discord webhook
    
    Args:
        webhook_url (str): Discord webhook URL
        title (str): Notification title
        message (str): Notification message
        notification_type (str): Notification type (info, trade, error, risk)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine color based on notification type
    colors = {
        "info": 0x3498db,  # Blue
        "trade": 0x2ecc71,  # Green
        "error": 0xe74c3c,  # Red
        "risk": 0xf39c12    # Yellow
    }
    
    color = colors.get(notification_type, 0x95a5a6)  # Default to gray
    
    # Format timestamp
    timestamp = datetime.utcnow().isoformat()
    
    # Create Discord embed
    embed = {
        "title": title,
        "description": message,
        "color": color,
        "timestamp": timestamp,
        "footer": {
            "text": "Trading Bot Notification"
        }
    }
    
    payload = {
        "embeds": [embed]
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 204:
            logger.info("Discord notification sent")
            return True
        else:
            logger.error(f"Error sending Discord notification: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Exception sending Discord notification: {e}")
        return False

def get_unread_notifications(user_id, limit=10):
    """
    Get unread notifications for a user
    
    Args:
        user_id (int): User ID
        limit (int): Maximum number of notifications to return
        
    Returns:
        list: List of notification objects
    """
    return Notification.query.filter_by(
        user_id=user_id,
        is_read=False
    ).order_by(Notification.timestamp.desc()).limit(limit).all()

def mark_notification_read(notification_id):
    """
    Mark a notification as read
    
    Args:
        notification_id (int): Notification ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        notification = Notification.query.get(notification_id)
        
        if notification:
            notification.is_read = True
            db.session.commit()
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        db.session.rollback()
        return False

def mark_all_read(user_id):
    """
    Mark all notifications as read for a user
    
    Args:
        user_id (int): User ID
        
    Returns:
        int: Number of notifications marked as read
    """
    try:
        notifications = Notification.query.filter_by(
            user_id=user_id,
            is_read=False
        ).all()
        
        count = len(notifications)
        
        for notification in notifications:
            notification.is_read = True
            
        db.session.commit()
        return count
        
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        db.session.rollback()
        return 0
