"""
Custom Data Transformers for Trading Bot
This module contains various data transformers that convert and process
market data into formats suitable for model consumption.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BaseTransformer:
    """Base class for all data transformers."""
    
    def __init__(self, name="base_transformer"):
        """Initialize the transformer with a name."""
        self.name = name
        
    def transform(self, data):
        """
        Transform the data.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement transform()")
    
    def inverse_transform(self, transformed_data):
        """
        Inverse transform the data (if applicable).
        Should be implemented by subclasses that support reversing transformations.
        """
        raise NotImplementedError("Subclasses must implement inverse_transform()")


class OHLCVTransformer(BaseTransformer):
    """
    Transformer for OHLCV (Open, High, Low, Close, Volume) data.
    Converts OHLCV data into a normalized format suitable for models.
    """
    
    def __init__(self, window_size=20, normalize=True):
        """
        Initialize the OHLCV transformer.
        
        Args:
            window_size (int): Size of the rolling window for normalization
            normalize (bool): Whether to normalize the data
        """
        super().__init__(name="ohlcv_transformer")
        self.window_size = window_size
        self.normalize = normalize
        self.last_mean = None
        self.last_std = None
        
    def transform(self, data):
        """
        Transform OHLCV data into model features.
        
        Args:
            data (list): List of dictionaries with OHLCV data
            
        Returns:
            dict: Dictionary with transformed features
        """
        if not data or len(data) == 0:
            logger.warning("Empty data provided to OHLCVTransformer")
            return None
            
        try:
            # Convert list of dictionaries to DataFrame
            if isinstance(data, list) and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Make sure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col.lower() in df.columns for col in required_cols):
                # Try to standardize column names
                for col in df.columns:
                    if col.lower() in ['open', 'o']:
                        df['open'] = df[col]
                    elif col.lower() in ['high', 'h']:
                        df['high'] = df[col]
                    elif col.lower() in ['low', 'l']:
                        df['low'] = df[col]
                    elif col.lower() in ['close', 'c']:
                        df['close'] = df[col]
                    elif col.lower() in ['volume', 'vol', 'v']:
                        df['volume'] = df[col]
                        
            # Calculate derived features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']).diff()
            df['range'] = df['high'] - df['low']
            df['body'] = abs(df['close'] - df['open'])
            df['body_pct'] = df['body'] / df['range']
            df['upper_wick'] = df.apply(lambda x: max(x['high'] - x['close'], x['high'] - x['open']), axis=1)
            df['lower_wick'] = df.apply(lambda x: max(x['open'] - x['low'], x['close'] - x['low']), axis=1)
            
            # Create lagged features
            for i in range(1, min(6, len(df))):
                df[f'close_lag_{i}'] = df['close'].shift(i)
                df[f'returns_lag_{i}'] = df['returns'].shift(i)
                
            # Calculate rolling statistics
            df['rolling_mean'] = df['close'].rolling(self.window_size, min_periods=1).mean()
            df['rolling_std'] = df['close'].rolling(self.window_size, min_periods=1).std()
            
            # Normalize if requested
            if self.normalize:
                # Store the last mean and std for inverse transforms
                self.last_mean = df['close'].rolling(self.window_size, min_periods=1).mean().iloc[-1]
                self.last_std = df['close'].rolling(self.window_size, min_periods=1).std().iloc[-1]
                
                # Skip timestamp for normalization
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df[col] = (df[col] - df[col].rolling(self.window_size, min_periods=1).mean()) / \
                              df[col].rolling(self.window_size, min_periods=1).std().replace(0, 1)
            
            # Fill NaN values
            df = df.fillna(0)
            
            # Return as dictionary of features
            return {
                'ohlcv_data': df.to_dict('records'),
                'last_row': df.iloc[-1].to_dict(),
                'shape': df.shape,
                'normalized': self.normalize,
                'window_size': self.window_size
            }
            
        except Exception as e:
            logger.error(f"Error in OHLCVTransformer: {str(e)}")
            return None
            
    def inverse_transform(self, transformed_data):
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            transformed_data (dict): Dictionary with transformed features
            
        Returns:
            dict: Dictionary with inverse-transformed features
        """
        if not transformed_data or not self.normalize or not self.last_mean or not self.last_std:
            return transformed_data
            
        try:
            # Create a copy to avoid modifying the original
            inverted = transformed_data.copy()
            
            # Convert the last row back to original scale
            last_row = inverted['last_row'].copy()
            for key, value in last_row.items():
                if isinstance(value, (int, float)):
                    last_row[key] = value * self.last_std + self.last_mean
                    
            inverted['last_row'] = last_row
            
            # Convert all rows in ohlcv_data
            if 'ohlcv_data' in inverted:
                ohlcv_data = []
                for row in inverted['ohlcv_data']:
                    row_copy = row.copy()
                    for key, value in row_copy.items():
                        if isinstance(value, (int, float)):
                            row_copy[key] = value * self.last_std + self.last_mean
                    ohlcv_data.append(row_copy)
                inverted['ohlcv_data'] = ohlcv_data
                
            return inverted
                
        except Exception as e:
            logger.error(f"Error in inverse_transform: {str(e)}")
            return transformed_data


class FeatureTransformer(BaseTransformer):
    """
    Transformer for extracting specific features from market data.
    Creates features like technical indicators that models can use.
    """
    
    def __init__(self, features_to_extract=None):
        """
        Initialize the feature transformer.
        
        Args:
            features_to_extract (list): List of features to extract
        """
        super().__init__(name="feature_transformer")
        self.features_to_extract = features_to_extract or [
            'trend', 'momentum', 'volatility', 'volume', 'swing'
        ]
        
    def transform(self, data):
        """
        Extract features from market data.
        
        Args:
            data (list or DataFrame): Market data
            
        Returns:
            dict: Dictionary with extracted features
        """
        if not data:
            logger.warning("Empty data provided to FeatureTransformer")
            return None
            
        try:
            # Convert to DataFrame if not already
            if isinstance(data, list) and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'ohlcv_data' in data:
                df = pd.DataFrame(data['ohlcv_data'])
            else:
                df = data.copy()
                
            features = {}
            
            # Extract trend features
            if 'trend' in self.features_to_extract:
                features['trend'] = self._extract_trend_features(df)
                
            # Extract momentum features
            if 'momentum' in self.features_to_extract:
                features['momentum'] = self._extract_momentum_features(df)
                
            # Extract volatility features
            if 'volatility' in self.features_to_extract:
                features['volatility'] = self._extract_volatility_features(df)
                
            # Extract volume features
            if 'volume' in self.features_to_extract:
                features['volume'] = self._extract_volume_features(df)
                
            # Extract swing features
            if 'swing' in self.features_to_extract:
                features['swing'] = self._extract_swing_features(df)
                
            return features
            
        except Exception as e:
            logger.error(f"Error in FeatureTransformer: {str(e)}")
            return None
            
    def _extract_trend_features(self, df):
        """Extract trend-related features."""
        try:
            # Ensure we have the necessary columns
            if 'close' not in df.columns:
                return {}
                
            # Simple moving averages
            df['sma5'] = df['close'].rolling(5, min_periods=1).mean()
            df['sma10'] = df['close'].rolling(10, min_periods=1).mean()
            df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
            
            # Moving average crossovers
            df['sma5_10_cross'] = np.where(df['sma5'] > df['sma10'], 1, -1)
            df['sma10_20_cross'] = np.where(df['sma10'] > df['sma20'], 1, -1)
            
            # Trend strength
            df['adx'] = self._calculate_adx(df)
            
            # Linear regression
            df['linear_slope'] = self._calculate_linear_slope(df['close'])
            
            # Return last row as feature dictionary
            return df.iloc[-1][['sma5', 'sma10', 'sma20', 'sma5_10_cross', 'sma10_20_cross', 'adx', 'linear_slope']].to_dict()
            
        except Exception as e:
            logger.error(f"Error extracting trend features: {str(e)}")
            return {}
            
    def _extract_momentum_features(self, df):
        """Extract momentum-related features."""
        try:
            # Ensure we have the necessary columns
            if 'close' not in df.columns:
                return {}
                
            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df)
            
            # MACD (Moving Average Convergence Divergence)
            macd_result = self._calculate_macd(df)
            df['macd'] = macd_result['macd']
            df['macd_signal'] = macd_result['signal']
            df['macd_hist'] = macd_result['histogram']
            
            # Rate of Change
            df['roc'] = df['close'].pct_change(10)
            
            # Return last row as feature dictionary
            return df.iloc[-1][['rsi', 'macd', 'macd_signal', 'macd_hist', 'roc']].to_dict()
            
        except Exception as e:
            logger.error(f"Error extracting momentum features: {str(e)}")
            return {}
            
    def _extract_volatility_features(self, df):
        """Extract volatility-related features."""
        try:
            # Ensure we have the necessary columns
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                return {}
                
            # Average True Range (ATR)
            df['atr'] = self._calculate_atr(df)
            
            # Bollinger Bands
            bollinger = self._calculate_bollinger_bands(df)
            df['bb_upper'] = bollinger['upper']
            df['bb_middle'] = bollinger['middle']
            df['bb_lower'] = bollinger['lower']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Historical Volatility
            df['hist_vol'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            # Return last row as feature dictionary
            return df.iloc[-1][['atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'hist_vol']].to_dict()
            
        except Exception as e:
            logger.error(f"Error extracting volatility features: {str(e)}")
            return {}
            
    def _extract_volume_features(self, df):
        """Extract volume-related features."""
        try:
            # Ensure we have the necessary columns
            if 'volume' not in df.columns:
                return {}
                
            # Volume moving averages
            df['volume_sma5'] = df['volume'].rolling(5, min_periods=1).mean()
            df['volume_sma20'] = df['volume'].rolling(20, min_periods=1).mean()
            
            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume_sma20']
            
            # On-Balance Volume (OBV)
            df['obv'] = self._calculate_obv(df)
            
            # Return last row as feature dictionary
            return df.iloc[-1][['volume', 'volume_sma5', 'volume_sma20', 'volume_ratio', 'obv']].to_dict()
            
        except Exception as e:
            logger.error(f"Error extracting volume features: {str(e)}")
            return {}
            
    def _extract_swing_features(self, df):
        """Extract swing-related features."""
        try:
            # Ensure we have the necessary columns
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                return {}
                
            # Recent swing points
            df['swing_high'] = self._identify_swings(df, 'high')
            df['swing_low'] = self._identify_swings(df, 'low')
            
            # Fibonacci retracement levels
            fib_levels = self._calculate_fibonacci_levels(df)
            for level, value in fib_levels.items():
                df[f'fib_{level}'] = value
                
            # Swing intensity
            df['swing_intensity'] = abs(df['high'] - df['low']) / df['close']
            
            # Relevant columns for this feature set
            relevant_cols = ['swing_high', 'swing_low', 'swing_intensity'] + [f'fib_{level}' for level in fib_levels.keys()]
            
            # Return last row as feature dictionary
            return df.iloc[-1][relevant_cols].to_dict()
            
        except Exception as e:
            logger.error(f"Error extracting swing features: {str(e)}")
            return {}
            
    # Helper methods for technical indicators
    def _calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index."""
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
        except:
            return pd.Series(50, index=df.index)
            
    def _calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            histogram = macd - signal
            
            return {
                'macd': macd.fillna(0),
                'signal': signal.fillna(0),
                'histogram': histogram.fillna(0)
            }
        except:
            zero_series = pd.Series(0, index=df.index)
            return {'macd': zero_series, 'signal': zero_series, 'histogram': zero_series}
            
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return atr.fillna(0)
        except:
            return pd.Series(0, index=df.index)
            
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index."""
        try:
            # Placeholder for ADX calculation
            # A real implementation would calculate +DI, -DI, and then ADX
            return pd.Series(50, index=df.index)
        except:
            return pd.Series(50, index=df.index)
            
    def _calculate_bollinger_bands(self, df, period=20, stddev=2):
        """Calculate Bollinger Bands."""
        try:
            middle = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            
            upper = middle + stddev * std
            lower = middle - stddev * std
            
            return {
                'upper': upper.fillna(df['close']),
                'middle': middle.fillna(df['close']),
                'lower': lower.fillna(df['close'])
            }
        except:
            return {
                'upper': df['close'],
                'middle': df['close'],
                'lower': df['close']
            }
            
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume."""
        try:
            obv = pd.Series(0, index=df.index)
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
                    
            return obv
        except:
            return pd.Series(0, index=df.index)
            
    def _calculate_linear_slope(self, series, period=10):
        """Calculate the linear regression slope."""
        try:
            slopes = pd.Series(index=series.index)
            
            for i in range(period, len(series)):
                y = series.iloc[i-period:i].values
                x = np.arange(period)
                slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)[0]
                slopes.iloc[i] = slope
                
            return slopes.fillna(0)
        except:
            return pd.Series(0, index=series.index)
            
    def _identify_swings(self, df, column, lookback=5):
        """Identify swing highs or lows."""
        try:
            swings = pd.Series(0, index=df.index)
            
            for i in range(lookback, len(df) - lookback):
                if column == 'high':
                    # Swing high
                    if all(df[column].iloc[i] > df[column].iloc[i-j] for j in range(1, lookback+1)) and \
                       all(df[column].iloc[i] > df[column].iloc[i+j] for j in range(1, lookback+1)):
                        swings.iloc[i] = 1
                elif column == 'low':
                    # Swing low
                    if all(df[column].iloc[i] < df[column].iloc[i-j] for j in range(1, lookback+1)) and \
                       all(df[column].iloc[i] < df[column].iloc[i+j] for j in range(1, lookback+1)):
                        swings.iloc[i] = 1
                        
            return swings
        except:
            return pd.Series(0, index=df.index)
            
    def _calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement levels."""
        try:
            # Find recent high and low
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            
            # Common Fibonacci levels
            levels = {
                '0': recent_low,
                '23_6': recent_low + 0.236 * (recent_high - recent_low),
                '38_2': recent_low + 0.382 * (recent_high - recent_low),
                '50_0': recent_low + 0.5 * (recent_high - recent_low),
                '61_8': recent_low + 0.618 * (recent_high - recent_low),
                '100': recent_high
            }
            
            return levels
        except:
            last_price = df['close'].iloc[-1] if 'close' in df.columns and len(df) > 0 else 0
            return {'0': last_price, '100': last_price}
            
    def inverse_transform(self, transformed_data):
        """
        Features don't need inverse transformation since they're not normalized.
        """
        return transformed_data


class TransformerPipeline:
    """
    Pipeline for applying multiple transformers sequentially.
    """
    
    def __init__(self, transformers=None):
        """
        Initialize transformer pipeline.
        
        Args:
            transformers (list): List of transformer instances
        """
        self.transformers = transformers or []
        
    def add_transformer(self, transformer):
        """Add a transformer to the pipeline."""
        if isinstance(transformer, BaseTransformer):
            self.transformers.append(transformer)
        else:
            raise ValueError("Transformer must be an instance of BaseTransformer")
            
    def transform(self, data):
        """
        Apply all transformers sequentially.
        
        Args:
            data: Input data to transform
            
        Returns:
            dict: Dictionary with all transformed features
        """
        result = {'original_data': data}
        
        transformed_data = data
        for transformer in self.transformers:
            try:
                transformer_result = transformer.transform(transformed_data)
                if transformer_result:
                    result[transformer.name] = transformer_result
                    transformed_data = transformer_result
            except Exception as e:
                logger.error(f"Error in transformer {transformer.name}: {str(e)}")
                
        return result
        
    def inverse_transform(self, transformed_data):
        """
        Apply inverse transformation in reverse order.
        
        Args:
            transformed_data: Transformed data to revert
            
        Returns:
            dict: Inverse transformed data
        """
        inverted_data = transformed_data
        
        for transformer in reversed(self.transformers):
            try:
                inverted_data = transformer.inverse_transform(inverted_data)
            except Exception as e:
                logger.error(f"Error in inverse transformer {transformer.name}: {str(e)}")
                
        return inverted_data


# Create default transformer pipeline
def create_default_pipeline():
    """Create the default transformer pipeline."""
    ohlcv_transformer = OHLCVTransformer(window_size=20, normalize=True)
    feature_transformer = FeatureTransformer()
    
    pipeline = TransformerPipeline([ohlcv_transformer, feature_transformer])
    return pipeline

# Default pipeline instance for easy import
default_pipeline = create_default_pipeline()