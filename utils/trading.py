import logging
import time
from datetime import datetime, timedelta
from app import db
from models import Trade, Notification, UserSettings
from utils.capital_api import (
    place_market_order, 
    close_position, 
    get_market_prices, 
    get_open_positions
)
from utils.notification import send_notification
from utils.risk_management import calculate_position_size, check_risk_limits

logger = logging.getLogger(__name__)

def place_trade(user_id, symbol, direction, quantity, api_key, is_demo, take_profit=None, stop_loss=None, strategy=None, ai_confidence=None):
    """
    Place a trade and record it in the database
    
    Args:
        user_id (int): User ID
        symbol (str): Trading symbol (e.g., "EURUSD")
        direction (str): "BUY" or "SELL"
        quantity (float): Trade size
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        take_profit (float, optional): Take profit level
        stop_loss (float, optional): Stop loss level
        strategy (str, optional): Strategy name that generated this trade
        ai_confidence (float, optional): AI confidence level (0-1)
        
    Returns:
        Trade: The created trade object or None if failed
    """
    logger.info(f"Placing trade: {direction} {symbol} x{quantity} for user {user_id}")
    
    # Check if we're in simulation mode
    simulation_mode = api_key and api_key.startswith('SIM_MODE_')
    if simulation_mode:
        logger.info("Using simulation mode for placing trade")
    
    # Check risk limits
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    if settings and not check_risk_limits(user_id, symbol, direction, quantity, settings):
        logger.warning(f"Trade rejected due to risk limits for user {user_id}")
        send_notification(
            user_id,
            "Trade Rejected",
            f"Your {direction} order for {symbol} was rejected because it exceeded your risk limits.",
            "risk"
        )
        return None
    
    try:
        # Get current market price
        price_data = get_market_prices(symbol, api_key, is_demo)
        
        # Determine entry price based on direction
        entry_price = price_data['ask'] if direction == 'BUY' else price_data['bid']
        
        # Place the order on Capital.com
        order_result = place_market_order(
            symbol,
            direction,
            quantity,
            api_key,
            is_demo,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Create trade record in database
        trade = Trade(
            user_id=user_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.utcnow(),
            take_profit=take_profit,
            stop_loss=stop_loss,
            status='OPEN',
            strategy_used=strategy,
            ai_confidence=ai_confidence,
            notes=f"Deal ID: {order_result.get('deal_id', 'Unknown')}"
        )
        
        db.session.add(trade)
        db.session.commit()
        
        # Send notification
        notification_message = f"Trade executed at {entry_price}. Quantity: {quantity}"
        if simulation_mode:
            notification_message += " (SIMULATION MODE)"
            
        send_notification(
            user_id,
            f"New Trade: {direction} {symbol}",
            notification_message,
            "trade",
            trade.id
        )
        
        logger.info(f"Trade placed successfully: ID {trade.id} for user {user_id}")
        return trade
        
    except Exception as e:
        logger.error(f"Error placing trade: {e}")
        db.session.rollback()
        
        # Send error notification
        send_notification(
            user_id,
            "Trade Error",
            f"Failed to place {direction} {symbol} trade: {str(e)}",
            "error"
        )
        
        return None

def close_trade(trade, api_key, is_demo):
    """
    Close an open trade
    
    Args:
        trade (Trade): The trade to close
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Closing trade: ID {trade.id} ({trade.direction} {trade.symbol})")
    
    # Check if we're in simulation mode
    simulation_mode = api_key and api_key.startswith('SIM_MODE_')
    if simulation_mode:
        logger.info("Using simulation mode for closing trade")
    
    try:
        # Get the position ID from trade notes
        deal_id = None
        if trade.notes and "Deal ID:" in trade.notes:
            deal_id = trade.notes.split("Deal ID:")[1].strip().split()[0]
        
        # If we have a deal ID, close it via API
        if deal_id:
            close_result = close_position(deal_id, api_key, is_demo)
            logger.info(f"Position closure result: {close_result}")
        
        # Get current market price
        price_data = get_market_prices(trade.symbol, api_key, is_demo)
        
        # Determine exit price based on direction
        exit_price = price_data['bid'] if trade.direction == 'BUY' else price_data['ask']
        
        # Update trade record
        pnl = trade.close_trade(exit_price)
        db.session.commit()
        
        # Send notification
        notification_message = f"Trade closed at {exit_price}. P&L: {pnl:.2f} ({trade.pnl_percentage:.2f}%)"
        if simulation_mode:
            notification_message += " (SIMULATION MODE)"
            
        send_notification(
            trade.user_id,
            f"Trade Closed: {trade.direction} {trade.symbol}",
            notification_message,
            "trade",
            trade.id
        )
        
        logger.info(f"Trade closed successfully: ID {trade.id}")
        return True
        
    except Exception as e:
        logger.error(f"Error closing trade: {e}")
        db.session.rollback()
        
        # Send error notification
        send_notification(
            trade.user_id,
            "Trade Closure Error",
            f"Failed to close {trade.direction} {trade.symbol} trade: {str(e)}",
            "error",
            trade.id
        )
        
        return False

def monitor_open_trades(api_key, is_demo, user_id):
    """
    Monitor open trades and update their status
    
    Args:
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        user_id (int): User ID
    """
    logger.info(f"Monitoring open trades for user {user_id}")
    
    try:
        # Get open trades from database
        open_trades = Trade.query.filter_by(user_id=user_id, status='OPEN').all()
        
        if not open_trades:
            logger.info(f"No open trades found for user {user_id}")
            return
        
        # Get current positions from Capital.com
        positions = get_open_positions(api_key, is_demo)
        position_ids = [p['position_id'] for p in positions]
        
        # Check if we're in simulation mode
        simulation_mode = api_key and api_key.startswith('SIM_MODE_')
        
        for trade in open_trades:
            # Extract deal ID from trade notes
            deal_id = None
            if trade.notes and "Deal ID:" in trade.notes:
                deal_id = trade.notes.split("Deal ID:")[1].strip().split()[0]
            
            # In simulation mode, randomly close some trades for demonstration
            if simulation_mode:
                # For simulation mode, we'll close trades based on these criteria:
                # - If stop loss or take profit would have been triggered
                # - Randomly with a small probability to simulate market movement
                import random
                
                # Get current market price
                price_data = get_market_prices(trade.symbol, api_key, is_demo)
                current_price = price_data['bid'] if trade.direction == 'BUY' else price_data['ask']
                
                should_close = False
                close_reason = ""
                
                # Check for stop loss hit
                if trade.stop_loss:
                    if (trade.direction == 'BUY' and current_price <= trade.stop_loss) or \
                       (trade.direction == 'SELL' and current_price >= trade.stop_loss):
                        should_close = True
                        close_reason = "Stop loss triggered"
                
                # Check for take profit hit
                if trade.take_profit:
                    if (trade.direction == 'BUY' and current_price >= trade.take_profit) or \
                       (trade.direction == 'SELL' and current_price <= trade.take_profit):
                        should_close = True
                        close_reason = "Take profit triggered"
                
                # Small random chance to close (1% per check, so roughly 5% per hour)
                if random.random() < 0.01:
                    should_close = True
                    close_reason = "Market movement"
                
                if should_close:
                    # Update trade record
                    exit_price = current_price
                    pnl = trade.close_trade(exit_price)
                    db.session.commit()
                    
                    # Send notification
                    send_notification(
                        user_id,
                        f"Trade Auto-Closed: {trade.direction} {trade.symbol}",
                        f"Trade was closed at {exit_price} ({close_reason}). P&L: {pnl:.2f} ({trade.pnl_percentage:.2f}%)",
                        "trade",
                        trade.id
                    )
                    
                    logger.info(f"Simulated trade closure: ID {trade.id} - {close_reason}")
            
            # For real API mode, check if the position is still open
            elif deal_id and deal_id not in position_ids:
                # Get current market price
                price_data = get_market_prices(trade.symbol, api_key, is_demo)
                exit_price = price_data['bid'] if trade.direction == 'BUY' else price_data['ask']
                
                # Update trade record
                pnl = trade.close_trade(exit_price)
                db.session.commit()
                
                # Send notification
                send_notification(
                    user_id,
                    f"Trade Auto-Closed: {trade.direction} {trade.symbol}",
                    f"Trade was closed at {exit_price}. P&L: {pnl:.2f} ({trade.pnl_percentage:.2f}%)",
                    "trade",
                    trade.id
                )
                
                logger.info(f"Trade marked as closed: ID {trade.id}")
                
    except Exception as e:
        logger.error(f"Error monitoring trades: {e}")
        
def setup_trading_scheduler(scheduler):
    """
    Set up scheduled tasks for trading
    
    Args:
        scheduler: APScheduler instance
    """
    from app import app
    
    # Add jobs to the scheduler
    with app.app_context():
        # Monitor trades every 5 minutes
        scheduler.add_job(
            monitor_all_user_trades,
            'interval',
            minutes=5,
            id='monitor_trades',
            replace_existing=True
        )
        
        # Run auto-trader every hour
        try:
            from utils.auto_trader import run_auto_trading
            scheduler.add_job(
                run_auto_trading,
                'interval',
                hours=1,
                id='auto_trader',
                replace_existing=True
            )
            logger.info("Auto-trader scheduled to run hourly")
        except Exception as e:
            logger.error(f"Error setting up auto-trader scheduler: {e}")
        
        # Sync with GitHub daily (if enabled)
        scheduler.add_job(
            sync_all_github_repos,
            'cron',
            hour=0,  # At midnight
            id='github_sync',
            replace_existing=True
        )
        
        logger.info("Trading scheduler initialized")

def monitor_all_user_trades():
    """
    Monitor trades for all users
    """
    from app import app
    from models import User
    
    with app.app_context():
        users = User.query.filter(User.api_key.isnot(None)).all()
        
        for user in users:
            settings = UserSettings.query.filter_by(user_id=user.id).first()
            if settings and settings.trading_enabled:
                try:
                    # Also pass the use_simulation_mode flag
                    monitor_open_trades(user.api_key, user.is_demo_account, user.id)
                except Exception as e:
                    logger.error(f"Error monitoring trades for user {user.id}: {e}")

def sync_all_github_repos():
    """
    Sync with GitHub for all users who have it enabled
    """
    from app import app
    from models import User
    from utils.github_integration import sync_with_github
    
    with app.app_context():
        users = User.query.all()
        
        for user in users:
            settings = UserSettings.query.filter_by(user_id=user.id).first()
            if settings and settings.auto_github_sync and settings.github_token and settings.github_repo:
                try:
                    sync_with_github(settings.github_token, settings.github_repo)
                    logger.info(f"GitHub sync completed for user {user.id}")
                except Exception as e:
                    logger.error(f"Error syncing GitHub for user {user.id}: {e}")
                    send_notification(
                        user.id,
                        "GitHub Sync Failed",
                        f"Failed to sync with GitHub: {str(e)}",
                        "error"
                    )
