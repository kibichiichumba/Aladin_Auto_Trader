import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize the OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_market_prediction(symbol, market_data, technical_analysis, news_data=None):
    """
    Generate a market prediction using AI
    
    Args:
        symbol (str): Market symbol
        market_data (list): Market data as OHLCV list
        technical_analysis (dict): Technical analysis results
        news_data (list, optional): News data related to the symbol
        
    Returns:
        dict: Prediction results
    """
    # Import here to avoid circular imports
    from utils.numerai_integration import get_numerai_prediction
    from utils.data_transformers import default_pipeline
    
    logger.info(f"Generating AI prediction for {symbol}")
    
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set, using fallback model")
        return fallback_prediction_model(technical_analysis)
    
    try:
        # Step 1: Apply data transformers to prepare data for AI models
        transformed_data = default_pipeline.transform(market_data)
        
        # Step 2: Get Numerai-based prediction for longer-term outlook
        numerai_prediction = get_numerai_prediction(symbol)
        
        # Step 3: Try to use OpenAI for enhanced analysis if API key available
        openai_analysis = _analyze_with_openai(symbol, transformed_data, technical_analysis, news_data)
        
        # Step 4: Combine both models with technical analysis for final prediction
        combined_prediction = _combine_predictions(
            symbol=symbol,
            numerai_prediction=numerai_prediction,
            openai_analysis=openai_analysis,
            technical_analysis=technical_analysis,
            transformed_data=transformed_data
        )
        
        return combined_prediction
        
    except Exception as e:
        logger.error(f"Error in AI prediction: {str(e)}", exc_info=True)
        return fallback_prediction_model(technical_analysis)

def fallback_prediction_model(technical_analysis):
    """
    Fallback prediction model when OpenAI is not available
    
    Args:
        technical_analysis (dict): Technical analysis results
        
    Returns:
        dict: Prediction results
    """
    prediction = "NEUTRAL"
    confidence = 0.5
    technical_score = 0
    features_used = ["rsi", "macd", "signal", "sma", "bollinger_bands"]
    
    if not technical_analysis or not isinstance(technical_analysis, dict):
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_used": "Basic Fallback",
            "features_used": features_used,
            "rationale": "Insufficient technical data available",
            "technical_score": 0,
            "sentiment_score": 0
        }
    
    # Start with signals from technical indicators
    signal_direction = "NEUTRAL"
    signal_strength = 0.5
    
    # Use the technical analysis signal as our prediction if available
    if "signal" in technical_analysis:
        signal = technical_analysis["signal"]
        if isinstance(signal, dict):
            signal_direction = signal.get("direction", "NEUTRAL")
            signal_strength = signal.get("strength", 0.5)
        else:
            signal_direction = signal if signal in ["BUY", "SELL"] else "NEUTRAL"
    
    # Initialize scores
    buy_points = 0
    sell_points = 0
    
    # Check RSI for trend strength
    rsi = technical_analysis.get("rsi", 50)
    if isinstance(rsi, (int, float)):
        if rsi < 30:  # Oversold
            buy_points += 2
            technical_score += 0.3
        elif rsi > 70:  # Overbought
            sell_points += 2
            technical_score -= 0.3
        elif rsi < 45:  # Slightly bullish
            buy_points += 1
            technical_score += 0.1
        elif rsi > 55:  # Slightly bearish
            sell_points += 1
            technical_score -= 0.1
    
    # Check MACD for momentum
    macd_data = technical_analysis.get("macd", {})
    if isinstance(macd_data, dict):
        macd_value = macd_data.get("macd", 0)
        macd_signal = macd_data.get("signal", 0)
        histogram = macd_data.get("histogram", 0)
        
        if isinstance(macd_value, (int, float)) and isinstance(macd_signal, (int, float)):
            if macd_value > macd_signal:  # Bullish crossover
                buy_points += 2
                technical_score += 0.2
            elif macd_value < macd_signal:  # Bearish crossover
                sell_points += 2
                technical_score -= 0.2
            
            # Check histogram for momentum
            if isinstance(histogram, (int, float)) and histogram > 0 and histogram > macd_value / 10:
                buy_points += 1
                technical_score += 0.1
            elif isinstance(histogram, (int, float)) and histogram < 0 and abs(histogram) > abs(macd_value) / 10:
                sell_points += 1
                technical_score -= 0.1
    
    # Check SMA for trend
    sma_data = technical_analysis.get("sma", {})
    if isinstance(sma_data, dict):
        sma_20 = sma_data.get("sma_20", 0)
        sma_50 = sma_data.get("sma_50", 0)
        
        if isinstance(sma_20, (int, float)) and isinstance(sma_50, (int, float)) and sma_20 > 0 and sma_50 > 0:
            if sma_20 > sma_50:  # Bullish trend
                buy_points += 1
                technical_score += 0.15
            elif sma_20 < sma_50:  # Bearish trend
                sell_points += 1
                technical_score -= 0.15
    
    # Check Bollinger Bands for volatility breakouts
    bb_data = technical_analysis.get("bollinger_bands", {})
    if isinstance(bb_data, dict):
        bb_upper = bb_data.get("upper", 0)
        bb_lower = bb_data.get("lower", 0)
        bb_middle = bb_data.get("middle", 0)
        
        # Get the last price if available
        last_price = 0
        if "current_price" in technical_analysis:
            last_price = technical_analysis.get("current_price")
        elif "last_candle" in technical_analysis:
            last_candle = technical_analysis.get("last_candle", {})
            if isinstance(last_candle, dict):
                last_price = last_candle.get("close", 0)
        
        # Check for Bollinger Band signals
        if (last_price is not None and last_price > 0 and 
            bb_upper is not None and bb_lower is not None and 
            isinstance(bb_upper, (int, float)) and isinstance(bb_lower, (int, float))):
            if last_price > bb_upper:  # Price above upper band - potential overbought
                sell_points += 1
                technical_score -= 0.1
            elif last_price < bb_lower:  # Price below lower band - potential oversold
                buy_points += 1
                technical_score += 0.1
    
    # Make prediction based on points
    if buy_points > sell_points:
        prediction = "BUY"
        confidence = min(0.9, 0.5 + (buy_points - sell_points) * 0.05)
    elif sell_points > buy_points:
        prediction = "SELL"
        confidence = min(0.9, 0.5 + (sell_points - buy_points) * 0.05)
    else:
        prediction = "NEUTRAL"
        confidence = 0.5
    
    # If technical analysis already has a strong signal, use it
    if signal_direction in ["BUY", "SELL"] and signal_strength > 0.7:
        prediction = signal_direction
        confidence = max(confidence, signal_strength)
    
    # Clamp technical score between -1 and 1
    technical_score = max(-1.0, min(1.0, technical_score))
    
    rationale = f"Based on technical analysis: RSI={rsi}, MACD={macd_data.get('macd', 'N/A')}, Buy score={buy_points}, Sell score={sell_points}"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "model_used": "Advanced Technical Analysis",
        "features_used": features_used,
        "rationale": rationale,
        "technical_score": technical_score,
        "sentiment_score": 0
    }

def technical_analysis_to_text(ta):
    """
    Convert technical analysis results to readable text
    
    Args:
        ta (dict): Technical analysis results
        
    Returns:
        str: Text representation of technical analysis
    """
    text = ""
    
    # SMA
    if ta.get('sma'):
        sma = ta['sma']
        text += "Moving Averages:\n"
        if sma.get('sma_20') and sma.get('sma_50'):
            text += f"- SMA 20: {sma['sma_20']}\n"
            text += f"- SMA 50: {sma['sma_50']}\n"
            text += f"- SMA Trend: {'Bullish' if sma['sma_20'] > sma['sma_50'] else 'Bearish'}\n"
    
    # RSI
    if ta.get('rsi') is not None:
        text += f"RSI (14): {ta['rsi']}\n"
        if ta['rsi'] < 30:
            text += "- RSI indicates oversold conditions\n"
        elif ta['rsi'] > 70:
            text += "- RSI indicates overbought conditions\n"
        else:
            text += "- RSI is in neutral territory\n"
    
    # MACD
    if ta.get('macd'):
        macd = ta['macd']
        if all(v is not None for v in [macd.get('macd'), macd.get('signal'), macd.get('histogram')]):
            text += f"MACD: {macd['macd']}, Signal: {macd['signal']}, Hist: {macd['histogram']}\n"
            if macd['macd'] > macd['signal']:
                text += "- MACD is bullish (MACD > Signal)\n"
            else:
                text += "- MACD is bearish (MACD < Signal)\n"
    
    # Bollinger Bands
    if ta.get('bollinger_bands'):
        bb = ta['bollinger_bands']
        if all(v is not None for v in [bb.get('upper'), bb.get('middle'), bb.get('lower')]):
            text += f"Bollinger Bands: Upper: {bb['upper']}, Middle: {bb['middle']}, Lower: {bb['lower']}\n"
    
    # Signal
    if ta.get('signal'):
        signal = ta['signal']
        text += f"Overall Signal: {signal['direction']} (Strength: {signal['strength']:.2f})\n"
    
    return text

def summarize_market_data(market_data):
    """
    Create a summary of market data
    
    Args:
        market_data (list): Market data as OHLCV list
        
    Returns:
        str: Summary of market data
    """
    if not market_data:
        return "No market data available"
        
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(market_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get recent price changes
    recent_data = df.tail(10)  # Last 10 periods
    price_change = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
    percent_change = (price_change / recent_data['close'].iloc[0]) * 100
    
    # Calculate volatility
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].std() * 100  # Multiply by 100 for percentage
    
    # Identify key price levels
    current_price = df['close'].iloc[-1]
    high_30d = df['high'].tail(30).max()
    low_30d = df['low'].tail(30).min()
    
    # Create summary text
    summary = f"""
    Current price: {current_price:.5f}
    Recent price change: {price_change:.5f} ({percent_change:.2f}%)
    30-day high: {high_30d:.5f}
    30-day low: {low_30d:.5f}
    Volatility (std of returns): {volatility:.2f}%
    
    Last 5 periods:
    """
    
    # Add the last 5 periods
    for i, row in df.tail(5).iterrows():
        summary += f"- {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: Open: {row['open']:.5f}, High: {row['high']:.5f}, Low: {row['low']:.5f}, Close: {row['close']:.5f}\n"
    
    return summary

def _analyze_with_openai(symbol, transformed_data, technical_analysis, news_data=None):
    """
    Analyze market data using OpenAI
    
    Args:
        symbol (str): Market symbol
        transformed_data (dict): Transformed market data
        technical_analysis (dict): Technical analysis results
        news_data (list, optional): News data related to the symbol
        
    Returns:
        dict: Analysis results
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set, skipping OpenAI analysis")
        return None
        
    try:
        # Prepare market data for the prompt
        market_summary = summarize_market_data(transformed_data.get('original_data', []))
        
        # Convert technical analysis to text
        ta_text = technical_analysis_to_text(technical_analysis)
        
        # Prepare news data if available
        news_text = ""
        if news_data:
            news_text = "Recent market news:\n"
            for news in news_data[:3]:  # Just use the 3 most recent news items
                news_text += f"- {news['title']} ({news['published_at']})\n"
                
        # Extract key features from transformed data if available
        feature_text = ""
        if 'feature_transformer' in transformed_data:
            features = transformed_data['feature_transformer']
            feature_text = "Key Features:\n"
            
            if 'trend' in features:
                trend = features['trend']
                feature_text += f"- Trend: {'Bullish' if trend.get('linear_slope', 0) > 0 else 'Bearish'}\n"
                feature_text += f"- ADX (Trend Strength): {trend.get('adx', 'N/A')}\n"
                
            if 'momentum' in features:
                momentum = features['momentum']
                feature_text += f"- RSI: {momentum.get('rsi', 'N/A')}\n"
                feature_text += f"- MACD: {momentum.get('macd', 'N/A')}\n"
                
            if 'volatility' in features:
                volatility = features['volatility']
                feature_text += f"- ATR: {volatility.get('atr', 'N/A')}\n"
                feature_text += f"- Bollinger Width: {volatility.get('bb_width', 'N/A')}\n"
        
        # Create prompt for GPT
        prompt = f"""
        You are an expert financial analyst specializing in forex and CFD trading. Analyze the following data for {symbol}:
        
        Market data:
        {market_summary}
        
        Technical Analysis:
        {ta_text}
        
        {feature_text}
        
        {news_text}
        
        Based on this data, provide a trading prediction in the following JSON format:
        {{
            "prediction": "BUY", "SELL", or "NEUTRAL",
            "confidence": (a number between 0 and 1),
            "rationale": "Brief explanation of your reasoning",
            "technical_score": (a number between -1 and 1 where negative values favor selling),
            "sentiment_score": (a number between -1 and 1 where negative values indicate negative sentiment)
        }}
        
        Respond only with the JSON.
        """
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2  # Lower temperature for more consistent results
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add model info to result
        result["model_used"] = "OpenAI GPT-4o"
        
        logger.info(f"AI prediction generated for {symbol}: {result['prediction']} (confidence: {result['confidence']})")
        return result
        
    except Exception as e:
        logger.error(f"Error in OpenAI analysis: {e}")
        return None


def _combine_predictions(symbol, numerai_prediction, openai_analysis, technical_analysis, transformed_data):
    """
    Combine predictions from multiple models
    
    Args:
        symbol (str): Market symbol
        numerai_prediction (dict): Prediction from Numerai
        openai_analysis (dict): Analysis from OpenAI
        technical_analysis (dict): Technical analysis results
        transformed_data (dict): Transformed market data
        
    Returns:
        dict: Combined prediction
    """
    # Default to technical analysis
    fallback = fallback_prediction_model(technical_analysis)
    
    # Use OpenAI if available
    if openai_analysis:
        primary_prediction = openai_analysis
    else:
        primary_prediction = fallback
        
    # Start with the primary prediction (OpenAI or fallback)
    result = {
        'symbol': symbol,
        'prediction': primary_prediction['prediction'],
        'confidence': primary_prediction['confidence'],
        'model_used': 'Ensemble (Numerai + OpenAI + Technical)',
        'rationale': primary_prediction.get('rationale', 'Based on combined model analysis'),
        'technical_score': primary_prediction.get('technical_score', 0),
        'sentiment_score': primary_prediction.get('sentiment_score', 0)
    }
    
    # Adjust with Numerai's longer-term prediction if available
    if numerai_prediction and isinstance(numerai_prediction, dict):
        # Safely get direction and confidence, handling different data types
        try:
            direction = str(numerai_prediction.get('direction', 'NEUTRAL'))
            confidence = float(numerai_prediction.get('confidence', 0.5))
            
            if direction != 'NEUTRAL':
                # Numerai boosts confidence if it agrees with primary prediction
                if direction == primary_prediction['prediction']:
                    result['confidence'] = min(0.95, float(result['confidence']) + 0.1)
                    result['rationale'] = f"Both short-term and Numerai models agree on {result['prediction']} signal. {result['rationale']}"
                
                # Numerai reduces confidence if it disagrees with primary prediction
                elif direction != primary_prediction['prediction']:
                    result['confidence'] = max(0.3, float(result['confidence']) - 0.1)
                    
                    # Strong disagreement might change prediction
                    if confidence > 0.8 and float(result['confidence']) < 0.6:
                        result['prediction'] = direction
                        result['confidence'] = confidence
                        result['rationale'] = f"Numerai model shows strong {result['prediction']} signal overriding short-term analysis."
                    else:
                        result['rationale'] = f"Short-term analysis suggests {primary_prediction['prediction']} but Numerai model shows {direction} for longer-term. {result['rationale']}"
            
            # Add Numerai confidence as a separate field
            result['numerai_confidence'] = confidence
        except Exception as e:
            logger.error(f"Error processing Numerai prediction in combine_predictions: {e}")
            result['numerai_confidence'] = 0
    else:
        result['numerai_confidence'] = 0
    
    # Add feature importances if available
    if 'feature_transformer' in transformed_data:
        result['features_used'] = list(transformed_data['feature_transformer'].keys())
        
    return result


def analyze_news_sentiment(news_title, news_content=""):
    """
    Analyze sentiment of market news using OpenAI
    
    Args:
        news_title (str): News title
        news_content (str, optional): News content
        
    Returns:
        dict: Sentiment analysis result
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set, returning neutral sentiment")
        return {"sentiment": "neutral", "score": 0, "confidence": 0.5}
    
    try:
        content = f"{news_title}"
        if news_content:
            # Limit content length to avoid token limits
            content += f"\n\n{news_content[:1000]}..."
        
        prompt = f"""
        Analyze the sentiment of this financial news for traders:
        
        {content}
        
        Provide your analysis in JSON format:
        {{
            "sentiment": "bullish", "bearish", or "neutral",
            "score": (a number between -1 and 1, where negative is bearish),
            "confidence": (a number between 0 and 1),
            "summary": "1-2 sentence summary of the impact on markets"
        }}
        
        Respond only with the JSON.
        """
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"News sentiment analysis: {result['sentiment']} (confidence: {result['confidence']})")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing news sentiment: {e}")
        return {"sentiment": "neutral", "score": 0, "confidence": 0.5}
