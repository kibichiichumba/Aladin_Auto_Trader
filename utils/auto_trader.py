import logging
import time
import random
from datetime import datetime
from app import db
from models import User, Trade, UserSettings, AIAnalysis
from utils.trading import place_trade
from utils.capital_api import get_market_prices
from utils.analysis import get_market_data, run_technical_analysis, get_ai_predictions

logger = logging.getLogger(__name__)

def auto_trade_for_user(user_id, api_key, is_demo_account):
    """
    Automatically execute trades based on AI predictions and technical analysis 
    for a specific user.
    
    Args:
        user_id (int): User ID
        api_key (str): Capital.com API key
        is_demo_account (bool): Whether account is demo
    """
    logger.info(f"Running auto-trader for user {user_id}")
    
    # Check if auto-trading is enabled in user settings
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    if not settings or not settings.trading_enabled:
        logger.info(f"Auto-trading is disabled for user {user_id}")
        return
    
    # Determine if we're in simulation mode
    simulation_mode = api_key and api_key.startswith('SIM_MODE_')
    
    try:
        # List of symbols to consider trading
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "US30", "GOLD", "SILVER", "OIL"]
        
        for symbol in symbols:
            logger.info(f"Analyzing {symbol} for potential trades")
            
            try:
                # Get market data for this symbol
                market_data = get_market_data(symbol, "1h", api_key, is_demo_account)
                if not market_data:
                    logger.warning(f"Could not get market data for {symbol}")
                    continue
                    
                # Run technical analysis
                ta_results = run_technical_analysis(market_data)
                logger.info(f"Technical analysis results type: {type(ta_results)}")
                
                # Convert numpy types to Python native types if needed
                if isinstance(ta_results, dict):
                    for key, value in list(ta_results.items()):
                        if hasattr(value, 'dtype') and 'numpy' in str(type(value)):
                            logger.info(f"Converting numpy value for {key}: {type(value)}")
                            ta_results[key] = float(value)
                
                # Get AI prediction if enabled
                ai_prediction = None
                ai_confidence = None
                if settings.use_ai_analysis:
                    try:
                        prediction = get_ai_predictions(symbol, "1h")
                        logger.info(f"AI prediction result type: {type(prediction)}")
                        
                        if prediction and isinstance(prediction, dict):
                            ai_prediction = prediction.get('prediction')
                            ai_confidence = prediction.get('confidence')
                            logger.info(f"Got AI prediction: {ai_prediction} with confidence {ai_confidence}")
                        else:
                            logger.warning(f"Unexpected prediction format: {prediction}")
                    except Exception as e:
                        logger.error(f"Error getting AI prediction for {symbol}: {e}", exc_info=True)
                
                # Get Numerai prediction before decide_trade_action
                from utils.numerai_integration import get_numerai_prediction
                numerai_data = None
                try:
                    numerai_data = get_numerai_prediction(symbol)
                    logger.info(f"Numerai prediction type: {type(numerai_data)}")
                except Exception as e:
                    logger.error(f"Error getting Numerai prediction: {e}", exc_info=True)
                
                # Determine trade direction based on technical indicators
                # and/or AI prediction if available
                logger.info(f"Deciding trade action with AI prediction: {ai_prediction}, confidence: {ai_confidence}")
                trade_decision = decide_trade_action(ta_results, ai_prediction, ai_confidence)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                continue
            
            if trade_decision['action'] == 'NONE':
                logger.info(f"No trade opportunity identified for {symbol}")
                continue
                
            # Calculate position size based on risk settings
            position_size = calculate_position_size(
                settings.risk_per_trade, 
                trade_decision['stop_loss_pips'], 
                symbol, 
                api_key, 
                is_demo_account
            )
            
            # Execute the trade
            if trade_decision['action'] in ['BUY', 'SELL']:
                logger.info(f"Placing {trade_decision['action']} trade for {symbol} (auto-trading)")
                
                try:
                    trade = place_trade(
                        user_id=user_id,
                        symbol=symbol,
                        direction=trade_decision['action'],
                        quantity=position_size,
                        api_key=api_key,
                        is_demo=is_demo_account,
                        take_profit=trade_decision['take_profit'],
                        stop_loss=trade_decision['stop_loss'],
                        strategy=trade_decision['strategy'],
                        ai_confidence=ai_confidence
                    )
                    
                    if trade:
                        logger.info(f"Auto-trade placed successfully: {symbol} {trade_decision['action']}")
                    else:
                        logger.warning(f"Failed to place auto-trade for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error placing auto-trade for {symbol}: {e}")
            
            # Add some delay between analyzing different symbols
            time.sleep(1)
                
    except Exception as e:
        logger.error(f"Error in auto_trade_for_user for user {user_id}: {e}")


def decide_trade_action(technical_analysis, ai_prediction=None, ai_confidence=None):
    """
    Decide whether to place a trade based on technical analysis and AI prediction.
    
    Args:
        technical_analysis (dict): Technical analysis results
        ai_prediction (str, optional): AI prediction (BUY, SELL, NEUTRAL)
        ai_confidence (float, optional): AI confidence level (0-1)
        
    Returns:
        dict: Trade decision containing action, stop loss, take profit, etc.
    """
    from utils.numerai_integration import get_numerai_prediction
    
    # Extract key technical indicators with safeguards for both dict and float values
    
    # Handle MACD - could be a dict or string
    if isinstance(technical_analysis.get('macd'), dict):
        macd = technical_analysis.get('macd', {}).get('signal', 'NEUTRAL')
    else:
        macd = 'NEUTRAL'
    
    # Handle RSI - could be a dict with 'value' or a direct float
    rsi_data = technical_analysis.get('rsi', 50)
    if isinstance(rsi_data, dict):
        rsi = rsi_data.get('value', 50)
    else:
        # If it's a direct value (float or numpy.float64), convert to Python float
        try:
            rsi = float(rsi_data)
        except (TypeError, ValueError):
            rsi = 50
    logger.info(f"RSI value type: {type(rsi)}, value: {rsi}")
    
    # Handle moving averages
    moving_avgs = technical_analysis.get('moving_averages', {})
    if isinstance(moving_avgs, dict):
        sma_trend = moving_avgs.get('sma_trend', 'NEUTRAL')
        ema_trend = moving_avgs.get('ema_trend', 'NEUTRAL')
    else:
        sma_trend = 'NEUTRAL'
        ema_trend = 'NEUTRAL'
    
    # Default response - no trade
    decision = {
        'action': 'NONE',
        'stop_loss': None,
        'take_profit': None,
        'stop_loss_pips': 0,
        'strategy': 'Ensemble Trading (Numerai + OpenAI + Technical)',
        'reason': 'No clear signal'
    }
    
    # Get symbol from technical analysis if available
    symbol = technical_analysis.get('symbol', 'UNKNOWN')
    
    # Get Numerai prediction for longer-term view
    numerai_data = None
    try:
        numerai_data = get_numerai_prediction(symbol)
        # Add detailed logging for debugging
        logger.info(f"Numerai prediction type: {type(numerai_data)}")
        if isinstance(numerai_data, dict):
            direction = numerai_data.get('direction', 'NEUTRAL')
            confidence = numerai_data.get('confidence', 0.5)
            logger.info(f"Got Numerai prediction for {symbol}: {direction} (confidence: {confidence})")
        else:
            logger.warning(f"Unexpected Numerai prediction type: {type(numerai_data)}")
            numerai_data = {
                "direction": "NEUTRAL",
                "confidence": 0.5
            }
    except Exception as e:
        logger.error(f"Could not get Numerai prediction: {e}", exc_info=True)
        numerai_data = {
            "direction": "NEUTRAL",
            "confidence": 0.5
        }
    
    # Count bullish and bearish signals with weighted scoring
    bullish_score = 0
    bearish_score = 0
    
    # Technical indicators (base score)
    # MACD signal
    if macd == 'BUY':
        bullish_score += 1.0
    elif macd == 'SELL':
        bearish_score += 1.0
    
    # RSI signal
    if rsi < 30:  # Oversold
        bullish_score += 1.2  # Slightly more weight to oversold
    elif rsi > 70:  # Overbought
        bearish_score += 1.2
    
    # Moving Averages trends
    if sma_trend == 'UP':
        bullish_score += 0.8
    elif sma_trend == 'DOWN':
        bearish_score += 0.8
        
    if ema_trend == 'UP':
        bullish_score += 0.8
    elif ema_trend == 'DOWN':
        bearish_score += 0.8
    
    # Include AI prediction if available (high weight)
    if ai_prediction and ai_confidence:
        if ai_prediction == 'BUY':
            weight = min(2.5, 1.5 + ai_confidence)  # Scale weight by confidence
            bullish_score += weight
            logger.info(f"AI prediction adds {weight:.2f} to bullish score")
        elif ai_prediction == 'SELL':
            weight = min(2.5, 1.5 + ai_confidence)
            bearish_score += weight
            logger.info(f"AI prediction adds {weight:.2f} to bearish score")
    
    # Include Numerai prediction if available (highest weight for strong signals)
    if numerai_data:
        try:
            # Ensure we're working with standard Python types
            direction = str(numerai_data.get('direction', 'NEUTRAL'))
            confidence = float(numerai_data.get('confidence', 0.5))
            
            if direction == 'BUY':
                weight = min(3.0, 1.5 + confidence * 2)
                bullish_score += weight
                logger.info(f"Numerai prediction adds {weight:.2f} to bullish score")
            elif direction == 'SELL':
                weight = min(3.0, 1.5 + confidence * 2)
                bearish_score += weight
                logger.info(f"Numerai prediction adds {weight:.2f} to bearish score")
        except Exception as e:
            logger.error(f"Error processing Numerai prediction: {e}")
    
    # Get current market prices for stop loss / take profit calculation
    current_price = 0
    
    # Handle different formats of technical_analysis
    if isinstance(technical_analysis, dict):
        current_price = technical_analysis.get('current_price', 0)
        
        # Try to extract from the last candle if current_price not available
        if current_price == 0 and 'last_candle' in technical_analysis:
            last_candle = technical_analysis.get('last_candle', {})
            if isinstance(last_candle, dict) and 'close' in last_candle:
                current_price = float(last_candle['close'])
        
        # If still 0, try to get it from bollinger bands middle value
        if current_price == 0 and 'bollinger_bands' in technical_analysis:
            bb = technical_analysis.get('bollinger_bands', {})
            if isinstance(bb, dict) and 'middle' in bb:
                current_price = float(bb['middle'])
    
    # Log score comparison
    logger.info(f"Trade decision scores - Bullish: {bullish_score:.2f}, Bearish: {bearish_score:.2f}")
    
    # Minimum score thresholds - higher than before to ensure stronger signals
    min_score = 3.0  # Minimum score required to open a position
    min_score_difference = 1.5  # Minimum difference between bullish and bearish scores
    
    # Make trading decision based on scores
    if bullish_score >= min_score and (bullish_score - bearish_score) >= min_score_difference:
        decision['action'] = 'BUY'
        decision['reason'] = f"Bullish score: {bullish_score:.2f}, Bearish: {bearish_score:.2f}"
        
        # Set stop loss and take profit (2:1 risk-reward ratio)
        stop_loss_pips = calculate_stop_loss_pips(current_price)
        decision['stop_loss'] = round(current_price - stop_loss_pips, 5)
        decision['take_profit'] = round(current_price + (stop_loss_pips * 2), 5)
        decision['stop_loss_pips'] = stop_loss_pips
        
        # Adaptive stop loss based on volatility if available in technical analysis
        if 'atr' in technical_analysis:
            atr = technical_analysis.get('atr', 0)
            if atr > 0:
                # Use ATR-based stop loss (1.5x ATR)
                adaptive_stop = current_price - (atr * 1.5)
                decision['stop_loss'] = round(adaptive_stop, 5)
                decision['take_profit'] = round(current_price + (atr * 3), 5)  # 2:1 reward-risk
                decision['stop_loss_pips'] = atr * 1.5
                decision['reason'] += f", using ATR-based stop loss ({atr:.5f})"
        
    elif bearish_score >= min_score and (bearish_score - bullish_score) >= min_score_difference:
        decision['action'] = 'SELL'
        decision['reason'] = f"Bearish score: {bearish_score:.2f}, Bullish: {bullish_score:.2f}"
        
        # Set stop loss and take profit (2:1 risk-reward ratio)
        stop_loss_pips = calculate_stop_loss_pips(current_price)
        decision['stop_loss'] = round(current_price + stop_loss_pips, 5)
        decision['take_profit'] = round(current_price - (stop_loss_pips * 2), 5)
        decision['stop_loss_pips'] = stop_loss_pips
        
        # Adaptive stop loss based on volatility if available
        if 'atr' in technical_analysis:
            atr = technical_analysis.get('atr', 0)
            if atr > 0:
                # Use ATR-based stop loss (1.5x ATR)
                adaptive_stop = current_price + (atr * 1.5)
                decision['stop_loss'] = round(adaptive_stop, 5)
                decision['take_profit'] = round(current_price - (atr * 3), 5)  # 2:1 reward-risk
                decision['stop_loss_pips'] = atr * 1.5
                decision['reason'] += f", using ATR-based stop loss ({atr:.5f})"
    
    # If a model made a major contribution to the decision, note it in the strategy name
    if decision['action'] != 'NONE':
        if numerai_data and isinstance(numerai_data, dict):
            direction = numerai_data.get('direction', 'NEUTRAL')
            if ((direction == 'BUY' and decision['action'] == 'BUY') or 
                (direction == 'SELL' and decision['action'] == 'SELL')):
                decision['strategy'] = f"Numerai-Enhanced {decision['action']}"
        elif ai_prediction and ai_confidence and ai_confidence > 0.7:
            decision['strategy'] = f"AI-Enhanced {decision['action']}"
    
    return decision


def calculate_stop_loss_pips(price):
    """
    Calculate appropriate stop loss in pips based on volatility and price.
    
    Args:
        price (float): Current market price
        
    Returns:
        float: Stop loss distance in pips
    """
    # Simple formula - higher prices have wider stops
    # This is a basic implementation and should be refined based on
    # actual volatility calculations for each instrument
    if price < 1:  # Forex pairs like EUR/USD
        return 0.0020  # 20 pips
    elif price < 100:  # Some commodities, minor currency pairs
        return price * 0.01  # 1% of price
    elif price < 1000:  # Higher-priced commodities, indices
        return price * 0.005  # 0.5% of price
    else:  # Very high-priced assets like BTC
        return price * 0.02  # 2% of price


def calculate_position_size(risk_percentage, stop_loss_pips, symbol, api_key, is_demo):
    """
    Calculate appropriate position size based on risk percentage and stop loss.
    
    Args:
        risk_percentage (float): Risk per trade as percentage of account
        stop_loss_pips (float): Stop loss distance in pips
        symbol (str): Trading symbol
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        
    Returns:
        float: Position size
    """
    from utils.capital_api import get_account_balance
    
    try:
        # Get account balance
        account_info = get_account_balance(api_key, is_demo)
        account_balance = account_info.get('balance', 10000)  # Default to 10000 if balance not available
        
        # Calculate risk amount in currency
        risk_amount = account_balance * (risk_percentage / 100)
        
        # Calculate position size based on stop loss pips
        # Note: This is a simplified calculation
        # In real trading, you need to account for currency pair, pip value, etc.
        if stop_loss_pips > 0:
            position_size = risk_amount / stop_loss_pips
        else:
            position_size = 0.01  # Minimum position size
            
        # Limit position size within reasonable bounds
        min_size = 0.01
        max_size = account_balance * 0.1  # Max 10% of account
        
        position_size = max(min_size, min(position_size, max_size))
        
        # Round to 2 decimal places
        return round(position_size, 2)
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0.01  # Return minimum position size on error


def run_auto_trading():
    """
    Main function to run auto-trading for all eligible users.
    This function should be called by the scheduler.
    """
    from app import app
    
    with app.app_context():
        # Get all users with API keys and auto-trading enabled
        try:
            # Query users with settings that have trading enabled
            users = User.query.join(UserSettings).filter(
                User.api_key.isnot(None),  # Must have API key
                UserSettings.trading_enabled == True  # Auto-trading must be enabled
            ).all()
            
            if not users:
                logger.info("No users with auto-trading enabled found")
                return
            
            logger.info(f"Running auto-trader for {len(users)} users")
            
            for user in users:
                try:
                    auto_trade_for_user(user.id, user.api_key, user.is_demo_account)
                except Exception as e:
                    logger.error(f"Error running auto-trader for user {user.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in run_auto_trading: {e}")