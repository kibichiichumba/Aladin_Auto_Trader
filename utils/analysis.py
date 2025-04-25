import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app import db
from models import MarketData, AIAnalysis
from utils.capital_api import get_historical_prices, get_market_prices
from utils.ai_model import generate_market_prediction

logger = logging.getLogger(__name__)

def get_market_data(symbol, timeframe, api_key, is_demo):
    """
    Get market data for a symbol and timeframe
    
    Args:
        symbol (str): Market symbol
        timeframe (str): Timeframe (e.g., "1m", "5m", "1h", "1d")
        api_key (str): Capital.com API key
        is_demo (bool): Whether using demo account
        
    Returns:
        list: Market data as OHLCV list of dictionaries
    """
    logger.info(f"Getting market data for {symbol} ({timeframe})")
    
    # Convert timeframe to Capital.com format
    resolution_map = {
        "1m": "MINUTE",
        "5m": "MINUTE_5",
        "15m": "MINUTE_15",
        "30m": "MINUTE_30",
        "1h": "HOUR",
        "4h": "HOUR_4",
        "1d": "DAY"
    }
    
    resolution = resolution_map.get(timeframe, "DAY")
    
    # Calculate time range based on timeframe
    end_date = datetime.utcnow()
    
    if timeframe == "1m":
        start_date = end_date - timedelta(days=1)
    elif timeframe in ["5m", "15m", "30m"]:
        start_date = end_date - timedelta(days=7)
    elif timeframe in ["1h", "4h"]:
        start_date = end_date - timedelta(days=30)
    else:  # daily
        start_date = end_date - timedelta(days=365)
    
    # Format dates for API
    from_date = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    to_date = end_date.strftime("%Y-%m-%dT%H:%M:%S")
    
    try:
        # Get historical data from Capital.com API
        history = get_historical_prices(symbol, resolution, from_date, to_date, api_key, is_demo)
        
        # Convert to OHLCV format
        ohlcv_data = []
        for i, price in enumerate(history['prices']):
            if i == 0:
                # First candle needs special handling
                ohlcv_data.append({
                    'timestamp': price['timestamp'],
                    'open': price['mid'],
                    'high': price['mid'],
                    'low': price['mid'],
                    'close': price['mid'],
                    'volume': 0
                })
            else:
                # Update the previous candle's close
                ohlcv_data[-1]['close'] = price['mid']
                
                # Add a new candle
                ohlcv_data.append({
                    'timestamp': price['timestamp'],
                    'open': ohlcv_data[-1]['close'],
                    'high': max(ohlcv_data[-1]['close'], price['mid']),
                    'low': min(ohlcv_data[-1]['close'], price['mid']),
                    'close': price['mid'],
                    'volume': 0
                })
        
        # Store data in database for future use
        store_market_data(symbol, timeframe, ohlcv_data)
        
        return ohlcv_data
    
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        
        # Try to retrieve from database as fallback
        stored_data = get_stored_market_data(symbol, timeframe)
        if stored_data:
            logger.info(f"Using stored market data for {symbol} ({timeframe})")
            return stored_data
            
        raise Exception(f"Failed to get market data: {str(e)}")

def store_market_data(symbol, timeframe, data):
    """
    Store market data in the database
    
    Args:
        symbol (str): Market symbol
        timeframe (str): Timeframe
        data (list): Market data as OHLCV list
    """
    try:
        # Store only the most recent 100 candles to save space
        recent_data = data[-100:] if len(data) > 100 else data
        
        for candle in recent_data:
            # Check if record already exists
            # Handle both string ISO format and integer timestamp formats
            if isinstance(candle['timestamp'], int):
                timestamp = datetime.fromtimestamp(candle['timestamp'] / 1000 if candle['timestamp'] > 1000000000000 else candle['timestamp'])
            else:
                try:
                    timestamp = datetime.fromisoformat(candle['timestamp'].replace('Z', '+00:00'))
                except (AttributeError, ValueError):
                    # If we can't parse it, use current time as fallback
                    logger.warning(f"Could not parse timestamp: {candle['timestamp']}, using current time instead")
                    timestamp = datetime.utcnow()
            
            existing = MarketData.query.filter_by(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp
            ).first()
            
            if not existing:
                market_data = MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle.get('volume', 0)
                )
                db.session.add(market_data)
        
        db.session.commit()
    except Exception as e:
        logger.error(f"Error storing market data: {e}")
        db.session.rollback()

def get_stored_market_data(symbol, timeframe):
    """
    Get stored market data from database
    
    Args:
        symbol (str): Market symbol
        timeframe (str): Timeframe
        
    Returns:
        list: Market data as OHLCV list or None
    """
    try:
        market_data = MarketData.query.filter_by(
            symbol=symbol,
            timeframe=timeframe
        ).order_by(MarketData.timestamp).all()
        
        if not market_data:
            return None
            
        return [{
            'timestamp': md.timestamp.isoformat(),
            'open': md.open,
            'high': md.high,
            'low': md.low,
            'close': md.close,
            'volume': md.volume
        } for md in market_data]
    
    except Exception as e:
        logger.error(f"Error retrieving stored market data: {e}")
        return None

def run_technical_analysis(market_data):
    """
    Run technical analysis on market data
    
    Args:
        market_data (list): Market data as OHLCV list
        
    Returns:
        dict: Technical analysis results
    """
    if not market_data or len(market_data) < 50:  # Require enough data points for reliable analysis
        logger.warning(f"Insufficient market data for analysis: {len(market_data) if market_data else 0} points")
        return {
            'sma': {'sma_20': None, 'sma_50': None, 'sma_trend': 'NEUTRAL'},
            'ema': {'ema_12': None, 'ema_26': None},
            'rsi': 50.0,  # Neutral RSI value
            'macd': {'macd': None, 'signal': None, 'histogram': None},
            'bollinger_bands': {'upper': None, 'middle': None, 'lower': None},
            'signal': {'direction': 'NEUTRAL', 'strength': 0.0},
            'current_price': market_data[-1]['close'] if market_data else 0,
            'last_candle': market_data[-1] if market_data else None
        }
    
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(market_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        # SMA (Simple Moving Average)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # EMA (Exponential Moving Average)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        # Handle potential division by zero
        rs = gain / loss.replace(0, float('inf'))
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fix any NaN or infinity values in RSI
        df['rsi'] = df['rsi'].clip(0, 100).fillna(50)
        
        # MACD (Moving Average Convergence Divergence)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Generate trading signal based on indicators
        signal = 'NEUTRAL'
        signal_strength = 0
        
        try:
            # MACD signal - with try/except to handle any potential NaN values
            if not pd.isna(df['macd'].iloc[-1]) and not pd.isna(df['macd_signal'].iloc[-1]) and \
               not pd.isna(df['macd'].iloc[-2]) and not pd.isna(df['macd_signal'].iloc[-2]):
                if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
                    signal = 'BUY'
                    signal_strength += 1
                elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
                    signal = 'SELL'
                    signal_strength += 1
            
            # RSI signal
            rsi_value = df['rsi'].iloc[-1]
            if not pd.isna(rsi_value):
                if rsi_value < 30:
                    if signal == 'SELL':
                        signal = 'NEUTRAL'
                    else:
                        signal = 'BUY'
                        signal_strength += 1
                elif rsi_value > 70:
                    if signal == 'BUY':
                        signal = 'NEUTRAL'
                    else:
                        signal = 'SELL'
                        signal_strength += 1
            
            # SMA crossover - with try/except to handle any potential NaN values
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            sma_20_prev = df['sma_20'].iloc[-2]
            sma_50_prev = df['sma_50'].iloc[-2]
            
            if not any(pd.isna(val) for val in [sma_20, sma_50, sma_20_prev, sma_50_prev]):
                if sma_20 > sma_50 and sma_20_prev <= sma_50_prev:
                    if signal != 'SELL':
                        signal = 'BUY'
                        signal_strength += 1
                elif sma_20 < sma_50 and sma_20_prev >= sma_50_prev:
                    if signal != 'BUY':
                        signal = 'SELL'
                        signal_strength += 1
            
            # Price vs Bollinger Bands
            if not pd.isna(df['bb_lower'].iloc[-1]) and not pd.isna(df['bb_upper'].iloc[-1]):
                if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
                    if signal == 'SELL':
                        signal = 'NEUTRAL'
                    else:
                        signal = 'BUY'
                        signal_strength += 0.5
                elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
                    if signal == 'BUY':
                        signal = 'NEUTRAL'
                    else:
                        signal = 'SELL'
                        signal_strength += 0.5
        except (IndexError, KeyError) as e:
            logger.error(f"Error calculating signals: {e}")
            # Keep default neutral signal
        
        # Get last values for return
        last_row = df.iloc[-1]
        
        # Create and return result object
        return {
            'sma': {
                'sma_20': round(last_row['sma_20'], 4) if not pd.isna(last_row['sma_20']) else None,
                'sma_50': round(last_row['sma_50'], 4) if not pd.isna(last_row['sma_50']) else None
            },
            'ema': {
                'ema_12': round(last_row['ema_12'], 4) if not pd.isna(last_row['ema_12']) else None,
                'ema_26': round(last_row['ema_26'], 4) if not pd.isna(last_row['ema_26']) else None
            },
            'rsi': round(last_row['rsi'], 2) if not pd.isna(last_row['rsi']) else None,
            'macd': {
                'macd': round(last_row['macd'], 4) if not pd.isna(last_row['macd']) else None,
                'signal': round(last_row['macd_signal'], 4) if not pd.isna(last_row['macd_signal']) else None,
                'histogram': round(last_row['macd_hist'], 4) if not pd.isna(last_row['macd_hist']) else None
            },
            'bollinger_bands': {
                'upper': round(last_row['bb_upper'], 4) if not pd.isna(last_row['bb_upper']) else None,
                'middle': round(last_row['bb_middle'], 4) if not pd.isna(last_row['bb_middle']) else None,
                'lower': round(last_row['bb_lower'], 4) if not pd.isna(last_row['bb_lower']) else None
            },
            'signal': {
                'direction': signal,
                'strength': min(signal_strength, 3) / 3  # Normalize to 0-1
            }
        }
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}", exc_info=True)
        return {
            'sma': {'sma_20': None, 'sma_50': None, 'sma_trend': 'NEUTRAL'},
            'ema': {'ema_12': None, 'ema_26': None},
            'rsi': 50.0,
            'macd': {'macd': None, 'signal': None, 'histogram': None},
            'bollinger_bands': {'upper': None, 'middle': None, 'lower': None},
            'signal': {'direction': 'NEUTRAL', 'strength': 0.0}
        }

def get_ai_predictions(symbol, timeframe):
    """
    Get AI predictions for a symbol
    
    Args:
        symbol (str): Market symbol
        timeframe (str): Timeframe
        
    Returns:
        dict: AI prediction results
    """
    from utils.data_transformers import default_pipeline
    from utils.numerai_integration import get_numerai_prediction
    # First check if we have recent predictions
    recent_prediction = AIAnalysis.query.filter_by(
        symbol=symbol,
        timeframe=timeframe
    ).order_by(AIAnalysis.timestamp.desc()).first()
    
    # If prediction is less than 1 hour old, return it
    if recent_prediction and (datetime.utcnow() - recent_prediction.timestamp).total_seconds() < 3600:
        return {
            'prediction': recent_prediction.prediction,
            'confidence': recent_prediction.confidence,
            'model_used': recent_prediction.model_used,
            'sentiment_score': recent_prediction.sentiment_score,
            'technical_score': recent_prediction.technical_score,
            'timestamp': recent_prediction.timestamp.isoformat()
        }
    
    # Otherwise, generate a new prediction
    try:
        # Get stored market data
        market_data = get_stored_market_data(symbol, timeframe)
        
        if not market_data or len(market_data) < 50:
            return None
            
        # Run technical analysis
        technical_analysis = run_technical_analysis(market_data)
        
        # Generate prediction using AI model
        prediction_result = generate_market_prediction(
            symbol,
            market_data,
            technical_analysis
        )
        
        # Store prediction in database
        # Convert list to string for features_used if it's a list
        features_used = prediction_result.get('features_used')
        if isinstance(features_used, list):
            features_used = ','.join(features_used)
            
        ai_analysis = AIAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            prediction=prediction_result['prediction'],
            confidence=float(prediction_result['confidence']),  # Ensure it's a Python float
            model_used=prediction_result['model_used'],
            features_used=features_used,
            sentiment_score=float(prediction_result.get('sentiment_score', 0)),  # Default 0 if missing
            technical_score=float(prediction_result.get('technical_score', 0)),  # Default 0 if missing  
            timeframe=timeframe
        )
        
        db.session.add(ai_analysis)
        db.session.commit()
        
        return {
            'prediction': ai_analysis.prediction,
            'confidence': ai_analysis.confidence,
            'model_used': ai_analysis.model_used,
            'sentiment_score': ai_analysis.sentiment_score,
            'technical_score': ai_analysis.technical_score,
            'timestamp': ai_analysis.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating AI prediction: {e}")
        return None
