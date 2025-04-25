import logging
import json
from datetime import datetime
from app import db
from models import Trade, Notification, UserSettings
from utils.trading import place_trade, close_trade
from utils.notification import send_notification
from utils.risk_management import check_risk_limits

logger = logging.getLogger(__name__)

def process_tradingview_signal(user_id, webhook_data, api_key, is_demo):
    """
    Process a webhook signal from TradingView
    
    Args:
        user_id (int): User ID
        webhook_data (dict): Webhook payload data
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        
    Returns:
        dict: Processing result
    """
    logger.info(f"Processing TradingView webhook for user {user_id}")
    
    try:
        # Validate webhook data
        required_fields = ['symbol', 'action']
        for field in required_fields:
            if field not in webhook_data:
                logger.error(f"Missing required field: {field}")
                return {"status": "error", "message": f"Missing required field: {field}"}
        
        symbol = webhook_data['symbol']
        action = webhook_data['action'].upper()
        
        # Process different action types
        if action in ['BUY', 'SELL']:
            return process_entry_signal(user_id, webhook_data, api_key, is_demo)
        elif action in ['CLOSE', 'EXIT']:
            return process_exit_signal(user_id, webhook_data, api_key, is_demo)
        else:
            logger.warning(f"Unknown action type: {action}")
            return {"status": "error", "message": f"Unknown action type: {action}"}
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}

def process_entry_signal(user_id, webhook_data, api_key, is_demo):
    """
    Process a trade entry signal
    
    Args:
        user_id (int): User ID
        webhook_data (dict): Webhook payload data
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        
    Returns:
        dict: Processing result
    """
    symbol = webhook_data['symbol']
    direction = webhook_data['action'].upper()
    
    # Get optional parameters
    quantity = float(webhook_data.get('quantity', 0.1))
    take_profit = float(webhook_data.get('take_profit')) if 'take_profit' in webhook_data else None
    stop_loss = float(webhook_data.get('stop_loss')) if 'stop_loss' in webhook_data else None
    strategy = webhook_data.get('strategy', 'TradingView')
    
    # Check user settings
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    if not settings or not settings.trading_enabled:
        logger.warning(f"Trading disabled for user {user_id}")
        return {"status": "error", "message": "Trading is disabled in user settings"}
    
    # Check risk limits
    if not check_risk_limits(user_id, symbol, direction, quantity, settings):
        logger.warning(f"Trade rejected due to risk limits for user {user_id}")
        send_notification(
            user_id,
            "Trade Rejected",
            f"Your {direction} order for {symbol} from TradingView was rejected because it exceeded your risk limits.",
            "risk"
        )
        return {"status": "error", "message": "Trade rejected due to risk limits"}
    
    # Place trade
    try:
        trade = place_trade(
            user_id,
            symbol,
            direction,
            quantity,
            api_key,
            is_demo,
            take_profit=take_profit,
            stop_loss=stop_loss,
            strategy=strategy
        )
        
        if trade:
            logger.info(f"Trade placed successfully: ID {trade.id}")
            return {
                "status": "success", 
                "message": f"Trade placed successfully: {direction} {symbol}",
                "trade_id": trade.id
            }
        else:
            logger.error("Failed to place trade")
            return {"status": "error", "message": "Failed to place trade"}
            
    except Exception as e:
        logger.error(f"Error placing trade: {e}")
        return {"status": "error", "message": str(e)}

def process_exit_signal(user_id, webhook_data, api_key, is_demo):
    """
    Process a trade exit signal
    
    Args:
        user_id (int): User ID
        webhook_data (dict): Webhook payload data
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        
    Returns:
        dict: Processing result
    """
    symbol = webhook_data['symbol']
    
    # Find open trades for this symbol
    open_trades = Trade.query.filter_by(
        user_id=user_id,
        symbol=symbol,
        status='OPEN'
    ).all()
    
    if not open_trades:
        logger.warning(f"No open trades found for {symbol}")
        return {"status": "warning", "message": "No open trades found for this symbol"}
    
    closed_trades = []
    
    # Close each trade
    for trade in open_trades:
        try:
            result = close_trade(trade, api_key, is_demo)
            
            if result:
                logger.info(f"Trade closed successfully: ID {trade.id}")
                closed_trades.append(trade.id)
            else:
                logger.error(f"Failed to close trade: ID {trade.id}")
                
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
    
    if closed_trades:
        return {
            "status": "success",
            "message": f"Closed {len(closed_trades)} trades for {symbol}",
            "closed_trades": closed_trades
        }
    else:
        return {"status": "error", "message": "Failed to close any trades"}

def process_mt4_mt5_signal(user_id, mt_data, api_key, is_demo):
    """
    Process a signal from MT4/MT5
    
    Args:
        user_id (int): User ID
        mt_data (dict): MT4/MT5 signal data
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        
    Returns:
        dict: Processing result
    """
    # Implementation would depend on how MT4/MT5 data is structured
    # Similar to TradingView webhook but with different field names
    
    logger.info(f"Processing MT4/MT5 signal for user {user_id}")
    
    try:
        # Map MT4/MT5 fields to standard format
        webhook_data = {
            'symbol': mt_data.get('symbol'),
            'action': mt_data.get('cmd', 'BUY'),  # MT4 uses "cmd" for command
            'quantity': float(mt_data.get('volume', 0.1)),
            'take_profit': float(mt_data.get('tp')) if mt_data.get('tp') else None,
            'stop_loss': float(mt_data.get('sl')) if mt_data.get('sl') else None,
            'strategy': 'MT4/MT5'
        }
        
        # Process based on action type
        action = webhook_data['action'].upper()
        if action in ['0', 'BUY']:
            webhook_data['action'] = 'BUY'
            return process_entry_signal(user_id, webhook_data, api_key, is_demo)
        elif action in ['1', 'SELL']:
            webhook_data['action'] = 'SELL'
            return process_entry_signal(user_id, webhook_data, api_key, is_demo)
        elif action in ['2', 'CLOSE']:
            webhook_data['action'] = 'CLOSE'
            return process_exit_signal(user_id, webhook_data, api_key, is_demo)
        else:
            logger.warning(f"Unknown MT4/MT5 action: {action}")
            return {"status": "error", "message": f"Unknown MT4/MT5 action: {action}"}
    
    except Exception as e:
        logger.error(f"Error processing MT4/MT5 signal: {e}")
        return {"status": "error", "message": str(e)}
