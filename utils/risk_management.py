import logging
from datetime import datetime, timedelta
from app import db
from models import Trade

logger = logging.getLogger(__name__)

def calculate_position_size(account_balance, risk_percent, stop_loss_pips, symbol_value_per_pip=10):
    """
    Calculate position size based on risk management principles
    
    Args:
        account_balance (float): Account balance
        risk_percent (float): Risk percentage per trade (e.g., 2%)
        stop_loss_pips (float): Stop loss in pips
        symbol_value_per_pip (float): Value per pip for the symbol
        
    Returns:
        float: Calculated position size
    """
    if stop_loss_pips <= 0:
        logger.warning("Invalid stop loss value")
        return 0
    
    # Calculate risk amount
    risk_amount = account_balance * (risk_percent / 100)
    
    # Calculate position size
    position_size = risk_amount / (stop_loss_pips * symbol_value_per_pip)
    
    return position_size

def check_risk_limits(user_id, symbol, direction, quantity, settings):
    """
    Check if a trade is within risk management limits
    
    Args:
        user_id (int): User ID
        symbol (str): Trading symbol
        direction (str): Trade direction (BUY/SELL)
        quantity (float): Trade size
        settings (UserSettings): User settings
        
    Returns:
        bool: True if within limits, False otherwise
    """
    # Check daily loss limit
    if settings.max_daily_loss > 0:
        daily_loss = calculate_daily_loss(user_id)
        max_loss = settings.max_daily_loss
        
        if daily_loss >= max_loss:
            logger.warning(f"Daily loss limit exceeded: {daily_loss}% > {max_loss}%")
            return False
    
    # Check max exposure per symbol
    symbol_exposure = calculate_symbol_exposure(user_id, symbol)
    if symbol_exposure + quantity > 5:  # Max 5 lots per symbol (configurable)
        logger.warning(f"Symbol exposure limit exceeded: {symbol_exposure + quantity} > 5")
        return False
    
    # Check max number of open trades (to prevent overtrading)
    open_trades_count = get_open_trades_count(user_id)
    if open_trades_count >= 10:  # Max 10 open trades (configurable)
        logger.warning(f"Max open trades limit exceeded: {open_trades_count} >= 10")
        return False
    
    # Check for conflicting positions (e.g., BUY and SELL on same symbol)
    if has_conflicting_position(user_id, symbol, direction):
        logger.warning(f"Conflicting position detected for {symbol} {direction}")
        return False
    
    return True

def calculate_daily_loss(user_id):
    """
    Calculate daily loss percentage
    
    Args:
        user_id (int): User ID
        
    Returns:
        float: Daily loss percentage
    """
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())
    
    # Get closed trades for today
    closed_trades = Trade.query.filter(
        Trade.user_id == user_id,
        Trade.status == 'CLOSED',
        Trade.exit_time.between(today_start, today_end)
    ).all()
    
    total_pnl = sum(trade.pnl for trade in closed_trades if trade.pnl is not None)
    
    # Calculate as percentage of account (would need account balance)
    # For now, just return the absolute loss
    return abs(total_pnl) if total_pnl < 0 else 0

def calculate_symbol_exposure(user_id, symbol):
    """
    Calculate current exposure for a symbol
    
    Args:
        user_id (int): User ID
        symbol (str): Trading symbol
        
    Returns:
        float: Total exposure for the symbol
    """
    open_trades = Trade.query.filter(
        Trade.user_id == user_id,
        Trade.symbol == symbol,
        Trade.status == 'OPEN'
    ).all()
    
    return sum(trade.quantity for trade in open_trades)

def get_open_trades_count(user_id):
    """
    Get count of open trades for a user
    
    Args:
        user_id (int): User ID
        
    Returns:
        int: Number of open trades
    """
    return Trade.query.filter(
        Trade.user_id == user_id,
        Trade.status == 'OPEN'
    ).count()

def has_conflicting_position(user_id, symbol, direction):
    """
    Check if user has conflicting position
    
    Args:
        user_id (int): User ID
        symbol (str): Trading symbol
        direction (str): Trade direction (BUY/SELL)
        
    Returns:
        bool: True if conflicting position exists, False otherwise
    """
    opposite_direction = 'SELL' if direction == 'BUY' else 'BUY'
    
    return Trade.query.filter(
        Trade.user_id == user_id,
        Trade.symbol == symbol,
        Trade.direction == opposite_direction,
        Trade.status == 'OPEN'
    ).count() > 0

def calculate_drawdown(user_id):
    """
    Calculate current drawdown from peak
    
    Args:
        user_id (int): User ID
        
    Returns:
        float: Drawdown percentage
    """
    # Get all closed trades
    trades = Trade.query.filter(
        Trade.user_id == user_id,
        Trade.status == 'CLOSED'
    ).order_by(Trade.exit_time).all()
    
    if not trades:
        return 0
    
    # Calculate cumulative P&L
    cumulative_pnl = 0
    peak = 0
    current_drawdown = 0
    max_drawdown = 0
    
    for trade in trades:
        if trade.pnl is not None:
            cumulative_pnl += trade.pnl
            
            # Update peak if new high
            if cumulative_pnl > peak:
                peak = cumulative_pnl
                current_drawdown = 0
            else:
                current_drawdown = (peak - cumulative_pnl) / peak if peak > 0 else 0
                
            # Update max drawdown
            max_drawdown = max(max_drawdown, current_drawdown)
    
    return max_drawdown * 100  # Convert to percentage

def apply_trailing_stop(trade, current_price, trailing_pips=20):
    """
    Apply trailing stop to a trade
    
    Args:
        trade (Trade): Trade to apply trailing stop to
        current_price (float): Current market price
        trailing_pips (int): Trailing stop distance in pips
        
    Returns:
        float: New stop loss level or None if no change
    """
    if trade.status != 'OPEN':
        return None
    
    # Convert pips to price
    pip_value = 0.0001  # For forex pairs like EURUSD (0.01 for JPY pairs)
    trailing_distance = trailing_pips * pip_value
    
    if trade.direction == 'BUY':
        # For BUY trades, we move stop loss up as price increases
        if trade.stop_loss is None or current_price - trailing_distance > trade.stop_loss:
            new_stop_loss = current_price - trailing_distance
            return new_stop_loss
    else:  # SELL
        # For SELL trades, we move stop loss down as price decreases
        if trade.stop_loss is None or current_price + trailing_distance < trade.stop_loss:
            new_stop_loss = current_price + trailing_distance
            return new_stop_loss
    
    return None
