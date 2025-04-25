import os
import logging
import requests
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger(__name__)

# Capital.com API endpoints
DEMO_API_URL = "https://demo-api-capital.backend-capital.com"
LIVE_API_URL = "https://api-capital.backend-capital.com"
LOGIN_URL = "https://capital.com/trading/login"

def get_api_url(is_demo=True):
    """Get the appropriate API URL based on account type"""
    return DEMO_API_URL if is_demo else LIVE_API_URL

def extract_api_key(email, password, is_demo=True):
    """
    Auto-login to Capital.com and extract API key using Selenium
    
    Args:
        email (str): Capital.com account email
        password (str): Capital.com account password
        is_demo (bool): Whether to use demo or live account
        
    Returns:
        str: API key
    """
    logger.info(f"Attempting to extract API key for {email} (Demo: {is_demo})")
    
    # For now, use a manual API key approach since Selenium has issues on Replit
    # Instead of auto-extracting, we'll provide a helper function to validate a provided key
    # In production, you would replace this with your actual API key
    # or implement a working Selenium solution on a different platform
    
    # This is a fallback method and should be replaced with proper API key extraction
    # when running on a platform that supports Selenium properly
    logger.warning("Selenium-based API key extraction is not working on this platform.")
    logger.warning("Using fallback method. Please provide API key manually in settings.")
    
    # For development only - simulated API key
    # In production, the user must provide their real API key
    return None

def validate_api_key(api_key, is_demo=True):
    """
    Validate the API key by making a test request
    
    Args:
        api_key (str): The API key to validate
        is_demo (bool): Whether to use demo or live environment
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not api_key:
        logger.warning("API key empty")
        return False
        
    # Check for simulation mode API key
    if api_key.startswith('SIM_MODE_'):
        logger.info("Simulation mode API key detected, validation not needed")
        return True
        
    if len(api_key.strip()) < 10:
        logger.warning("API key too short")
        return False
        
    # Clean up the API key (remove any whitespace, quotes, etc.)
    api_key = api_key.strip()
    
    base_url = get_api_url(is_demo)
    
    # Try different header formats
    headers_options = [
        # Standard format
        {
            'X-CAP-API-KEY': api_key,
            'Content-Type': 'application/json'
        },
        # Alternate format
        {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        # Simple format
        {
            'X-CAP-API-KEY': api_key
        }
    ]
    
    # Try the accounts endpoint instead of session
    endpoints = [
        "/api/v1/session",
        "/api/v1/accounts",
        "/api/v1/ping"  # Some APIs have a simple health check endpoint
    ]
    
    for headers in headers_options:
        for endpoint in endpoints:
            try:
                logger.debug(f"Trying endpoint {endpoint} with headers: {headers}")
                response = requests.get(f"{base_url}{endpoint}", headers=headers)
                
                logger.debug(f"Response status: {response.status_code}, Response: {response.text[:100]}...")
                
                if response.status_code == 200:
                    logger.info("API key validation successful")
                    return True
                elif response.status_code == 401 or response.status_code == 403:
                    logger.warning(f"API key unauthorized: {response.status_code}")
                    continue
                else:
                    logger.warning(f"Unexpected status code: {response.status_code}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error validating API key with format {headers} on endpoint {endpoint}: {e}")
                continue
    
    logger.error("All API key validation attempts failed")
    return False

def get_account_balance(api_key, is_demo=True):
    """
    Get account balance using the API key
    
    Args:
        api_key (str): The Capital.com API key
        is_demo (bool): Whether to use demo or live environment
        
    Returns:
        dict: Account balance information
    """
    # Check for simulation mode
    if api_key and api_key.startswith('SIM_MODE_'):
        # Return simulated account data for testing
        return {
            'balance': 10000.00,  # Simulated balance of $10,000
            'currency': 'USD',
            'profit_loss': 250.00,  # Simulated profit
            'available': 9750.00  # Simulated available funds
        }
    
    # Real API request
    base_url = get_api_url(is_demo)
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{base_url}/api/v1/accounts", headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Error fetching account balance: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code}")
            
        accounts_data = response.json()
        
        if not accounts_data.get('accounts'):
            raise Exception("No accounts found")
            
        # Get the first account (most users will have only one)
        account = accounts_data['accounts'][0]
        
        return {
            'balance': account.get('balance', 0),
            'currency': account.get('currency', 'USD'),
            'profit_loss': account.get('profitLoss', 0),
            'available': account.get('available', 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        raise Exception(f"Failed to get account balance: {str(e)}")

def get_market_prices(symbol, api_key, is_demo=True):
    """
    Get current market prices for a symbol
    
    Args:
        symbol (str): The market symbol (e.g., "EURUSD")
        api_key (str): The Capital.com API key
        is_demo (bool): Whether to use demo or live environment
        
    Returns:
        dict: Market price information
    """
    # Check for simulation mode
    if api_key and api_key.startswith('SIM_MODE_'):
        # Return simulated price data based on the symbol
        import random
        import time
        
        # Base prices for common symbols in simulation mode
        base_prices = {
            'EURUSD': 1.10,
            'GBPUSD': 1.27,
            'USDJPY': 150.25,
            'BTCUSD': 65000.00,
            'US30': 38500.00,
            'GOLD': 2300.50,
            'SILVER': 27.35,
            'OIL': 85.20,
            'TESLA': 175.30,
            'AMAZON': 180.45,
            'APPLE': 170.75,
            'NETFLIX': 630.20
        }
        
        # Use a default price for unknown symbols
        base_price = base_prices.get(symbol.upper(), 100.00)
        
        # Add some random variation
        spread = base_price * 0.0002  # 0.02% spread
        variation = base_price * 0.001 * (random.random() - 0.5)  # ±0.05% random variation
        
        bid = base_price + variation
        ask = bid + spread
        
        return {
            'symbol': symbol,
            'bid': round(bid, 5),
            'ask': round(ask, 5),
            'timestamp': int(time.time() * 1000)  # Current timestamp in milliseconds
        }
    
    # Real API request
    base_url = get_api_url(is_demo)
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{base_url}/api/v1/prices/{symbol}", headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Error fetching market prices: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code}")
            
        price_data = response.json()
        
        return {
            'symbol': symbol,
            'bid': price_data.get('bid', 0),
            'ask': price_data.get('ask', 0),
            'timestamp': price_data.get('timestamp', '')
        }
        
    except Exception as e:
        logger.error(f"Error getting market prices: {e}")
        raise Exception(f"Failed to get market prices: {str(e)}")

def place_market_order(symbol, direction, quantity, api_key, is_demo=True, stop_loss=None, take_profit=None):
    """
    Place a market order
    
    Args:
        symbol (str): The market symbol (e.g., "EURUSD")
        direction (str): "BUY" or "SELL"
        quantity (float): The trade size
        api_key (str): The Capital.com API key
        is_demo (bool): Whether to use demo or live environment
        stop_loss (float, optional): Stop loss price
        take_profit (float, optional): Take profit price
        
    Returns:
        dict: Order confirmation data
    """
    # Check for simulation mode
    if api_key and api_key.startswith('SIM_MODE_'):
        # Generate a simulated deal ID
        import time
        import uuid
        
        deal_id = f"SIM_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        return {
            'deal_id': deal_id,
            'status': 'ACCEPTED',
            'affected_deals': [],
            'simulation': True
        }
    
    # Real API request
    base_url = get_api_url(is_demo)
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    order_data = {
        "epic": symbol,
        "direction": direction,
        "size": quantity,
        "orderType": "MARKET"
    }
    
    if stop_loss:
        order_data["stopLevel"] = stop_loss
        
    if take_profit:
        order_data["profitLevel"] = take_profit
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/positions", 
            headers=headers, 
            data=json.dumps(order_data)
        )
        
        if response.status_code not in [200, 201]:
            logger.error(f"Error placing order: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code} - {response.text}")
            
        order_response = response.json()
        
        return {
            'deal_id': order_response.get('dealId', ''),
            'status': order_response.get('dealStatus', ''),
            'affected_deals': order_response.get('affectedDeals', [])
        }
        
    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        raise Exception(f"Failed to place market order: {str(e)}")

def close_position(position_id, api_key, is_demo=True):
    """
    Close an open position
    
    Args:
        position_id (str): The position ID to close
        api_key (str): The Capital.com API key
        is_demo (bool): Whether to use demo or live environment
        
    Returns:
        dict: Position closure confirmation
    """
    # Check for simulation mode
    if api_key and api_key.startswith('SIM_MODE_') or (position_id and position_id.startswith('SIM_')):
        # Handle simulated trade closure
        return {
            'deal_id': f"CLOSE_{position_id}",
            'status': 'CLOSED',
            'simulation': True
        }
    
    # Real API request
    base_url = get_api_url(is_demo)
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.delete(
            f"{base_url}/api/v1/positions/{position_id}", 
            headers=headers
        )
        
        if response.status_code not in [200, 204]:
            logger.error(f"Error closing position: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code} - {response.text}")
            
        if response.status_code == 204:  # No content response
            return {'status': 'CLOSED'}
            
        closure_data = response.json()
        
        return {
            'deal_id': closure_data.get('dealId', ''),
            'status': closure_data.get('dealStatus', '')
        }
        
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise Exception(f"Failed to close position: {str(e)}")

def get_open_positions(api_key, is_demo=True):
    """
    Get all open positions
    
    Args:
        api_key (str): The Capital.com API key
        is_demo (bool): Whether to use demo or live environment
        
    Returns:
        list: Open positions
    """
    # Check for simulation mode
    if api_key and api_key.startswith('SIM_MODE_'):
        # In simulation mode, we use the database to get positions
        # This function doesn't have access to the DB, so it will return an empty list
        # The actual open positions will be fetched directly from the database in trading.py
        return []
    
    # Real API request
    base_url = get_api_url(is_demo)
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{base_url}/api/v1/positions", headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Error fetching open positions: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code}")
            
        positions_data = response.json()
        
        positions = []
        if positions_data.get('positions'):
            for pos in positions_data['positions']:
                positions.append({
                    'position_id': pos.get('position', {}).get('dealId', ''),
                    'symbol': pos.get('market', {}).get('epic', ''),
                    'direction': pos.get('position', {}).get('direction', ''),
                    'size': pos.get('position', {}).get('size', 0),
                    'open_level': pos.get('position', {}).get('level', 0),
                    'created_date': pos.get('position', {}).get('createdDate', ''),
                    'profit_loss': pos.get('position', {}).get('profitAndLoss', 0),
                    'stop_level': pos.get('position', {}).get('stopLevel', None),
                    'profit_level': pos.get('position', {}).get('profitLevel', None)
                })
                
        return positions
        
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        raise Exception(f"Failed to get open positions: {str(e)}")

def get_historical_prices(symbol, resolution, from_date, to_date, api_key, is_demo=True):
    """
    Get historical price data
    
    Args:
        symbol (str): The market symbol (e.g., "EURUSD")
        resolution (str): Time resolution (e.g., "MINUTE", "HOUR", "DAY")
        from_date (str): Start date in ISO format
        to_date (str): End date in ISO format
        api_key (str): The Capital.com API key
        is_demo (bool): Whether to use demo or live environment
        
    Returns:
        dict: Historical price data
    """
    # Check for simulation mode
    if api_key and api_key.startswith('SIM_MODE_'):
        # Generate simulated historical data
        import random
        import time
        from datetime import datetime, timedelta
        
        # Base prices for common symbols in simulation mode
        base_prices = {
            'EURUSD': 1.10,
            'GBPUSD': 1.27,
            'USDJPY': 150.25,
            'BTCUSD': 65000.00,
            'US30': 38500.00,
            'GOLD': 2300.50,
            'SILVER': 27.35,
            'OIL': 85.20,
            'TESLA': 175.30,
            'AMAZON': 180.45,
            'APPLE': 170.75,
            'NETFLIX': 630.20
        }
        
        # Use a default price for unknown symbols
        base_price = base_prices.get(symbol.upper(), 100.00)
        
        # Convert dates to datetime objects
        try:
            from_datetime = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
            to_datetime = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
        except ValueError:
            # If the dates are in milliseconds
            try:
                from_datetime = datetime.fromtimestamp(int(from_date) / 1000)
                to_datetime = datetime.fromtimestamp(int(to_date) / 1000)
            except Exception:
                # Default to last 30 days
                to_datetime = datetime.now()
                from_datetime = to_datetime - timedelta(days=30)
        
        # Calculate interval between data points based on resolution
        if resolution.upper() == 'MINUTE':
            interval = timedelta(minutes=1)
            points = min(1440, int((to_datetime - from_datetime).total_seconds() / 60))  # max 1 day of minute data
        elif resolution.upper() == 'HOUR':
            interval = timedelta(hours=1)
            points = min(720, int((to_datetime - from_datetime).total_seconds() / 3600))  # max 30 days of hourly data
        else:  # DAY or any other
            interval = timedelta(days=1)
            points = min(365, int((to_datetime - from_datetime).days) + 1)  # max 1 year of daily data
        
        prices = []
        current_dt = from_datetime
        current_price = base_price
        
        # Generate simulated price data
        for _ in range(points):
            # Add some random variation (more for longer timeframes)
            if resolution.upper() == 'MINUTE':
                variation = current_price * 0.0002 * (random.random() - 0.5)  # ±0.01% random variation
            elif resolution.upper() == 'HOUR':
                variation = current_price * 0.001 * (random.random() - 0.5)  # ±0.05% random variation
            else:  # DAY
                variation = current_price * 0.004 * (random.random() - 0.5)  # ±0.2% random variation
            
            # Apply trend bias (65% chance of continuing previous direction)
            if len(prices) > 0:
                last_mid = (prices[-1]['bid'] + prices[-1]['ask']) / 2
                if current_price > last_mid and random.random() < 0.65:
                    variation = abs(variation)  # Continue uptrend
                elif current_price < last_mid and random.random() < 0.65:
                    variation = -abs(variation)  # Continue downtrend
            
            current_price += variation
            spread = current_price * 0.0002  # 0.02% spread
            
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            timestamp = int(current_dt.timestamp() * 1000)  # Convert to milliseconds
            
            prices.append({
                'timestamp': timestamp,
                'bid': round(bid, 5),
                'ask': round(ask, 5),
                'mid': round(current_price, 5)
            })
            
            current_dt += interval
        
        return {
            'symbol': symbol,
            'resolution': resolution,
            'from': from_date,
            'to': to_date,
            'prices': prices
        }
    
    # Real API request
    base_url = get_api_url(is_demo)
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    params = {
        'resolution': resolution,
        'from': from_date,
        'to': to_date
    }
    
    try:
        response = requests.get(
            f"{base_url}/api/v1/prices/{symbol}/history", 
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            logger.error(f"Error fetching historical prices: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code}")
            
        history_data = response.json()
        
        # Transform the data into a more usable format
        prices = []
        for i, timestamp in enumerate(history_data.get('timestamps', [])):
            if i < len(history_data.get('bid', [])) and i < len(history_data.get('ask', [])):
                prices.append({
                    'timestamp': timestamp,
                    'bid': history_data['bid'][i],
                    'ask': history_data['ask'][i],
                    'mid': (history_data['bid'][i] + history_data['ask'][i]) / 2
                })
                
        return {
            'symbol': symbol,
            'resolution': resolution,
            'from': from_date,
            'to': to_date,
            'prices': prices
        }
        
    except Exception as e:
        logger.error(f"Error getting historical prices: {e}")
        raise Exception(f"Failed to get historical prices: {str(e)}")
