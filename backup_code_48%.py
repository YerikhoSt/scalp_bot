import time
import pandas as pd
import numpy as np
from binance.client import Client
from decimal import Decimal
import threading
from collections import deque

# === KONFIGURASI API ===
API_KEY = 'ZzezbKmN5XL41h6kIjbeVYOjGJnehVbvLJtHf2YE1AcyRPzDnNbdOUaXhnUmSrcf'
API_SECRET = '4JRNhv0xsrGLLMQPi1uog8wPOFT5dW67zqATDY4EIsobdsT5KmUGl5bvHq6WCSUf'

client = Client(API_KEY, API_SECRET)
client.futures_change_leverage(symbol='BTCUSDT', leverage=100)

# === RATE LIMITING CONFIGURATION ===
# Binance allows 20 orders per 10 seconds
MAX_ORDERS_PER_10_SECONDS = 18  # Leave some buffer
ORDER_RATE_LIMIT_WINDOW = 10  # seconds
ORDER_TIMESTAMPS = deque()  # Track recent order timestamps
RATE_LIMIT_LOCK = threading.Lock()

# Delay between symbol processing to prevent rate limiting
SYMBOL_PROCESSING_DELAY = 0.8  # seconds between each symbol
BATCH_SIZE = 5  # Process symbols in batches of 5

# === KONFIGURASI TRADING ===
SYMBOLS = [
    'BTCUSDT',
    'AVAAIUSDT', 
    'ADAUSDT',
    'SOLUSDT',
    'SUIUSDT',
    'AVAXUSDT',
    'TRXUSDT',
    'TONUSDT',
    'LTCUSDT',
    'NEARUSDT',
    'XRPUSDT',
    'DOGEUSDT',
    'DOTUSDT',
    'LINKUSDT',
    'XLMUSDT',
    'XMRUSDT',
    'ETHUSDT'
]
INTERVAL = '1m'  # 1-minute timeframe for scalping
MARGIN_RATIO = 0.005  # 0.5% of balance per trade cycle
LEVERAGE = 125
# Dynamic TP/SL Base Parameters (will be adjusted based on indicators)
BASE_STOP_LOSS_PCT = -0.5  # Base -50% ROI with 125x leverage
BASE_TAKE_PROFIT_PCT = 0.5   # Base +50% ROI with 125x leverage

# Dynamic TP/SL Adjustment Ranges
MIN_STOP_LOSS_PCT = -0.3   # Minimum -30% ROI (tight SL for strong signals)
MAX_STOP_LOSS_PCT = -0.8   # Maximum -80% ROI (wide SL for volatile markets)
MIN_TAKE_PROFIT_PCT = 0.3  # Minimum +30% ROI (quick profit in uncertain conditions)
MAX_TAKE_PROFIT_PCT = 1.0  # Maximum +100% ROI (let profits run in strong trends)

# Multi-Indicator Strategy Settings for 1-minute scalping
MIN_CONFIDENCE = 0.6  # Minimum confidence to place trade (60%)
RSI_PERIOD = 14  # Standard RSI period for 1-minute analysis
RSI_OVERBOUGHT = 70  # RSI level for short signals
RSI_OVERSOLD = 30   # RSI level for long signals

def check_rate_limit():
    """Check if we can place an order without exceeding rate limits"""
    with RATE_LIMIT_LOCK:
        current_time = time.time()
        
        # Remove timestamps older than 10 seconds
        while ORDER_TIMESTAMPS and current_time - ORDER_TIMESTAMPS[0] > ORDER_RATE_LIMIT_WINDOW:
            ORDER_TIMESTAMPS.popleft()
        
        # Check if we can place another order
        return len(ORDER_TIMESTAMPS) < MAX_ORDERS_PER_10_SECONDS

def record_order():
    """Record that we placed an order"""
    with RATE_LIMIT_LOCK:
        ORDER_TIMESTAMPS.append(time.time())

def wait_for_rate_limit():
    """Wait if we're hitting the rate limit"""
    with RATE_LIMIT_LOCK:
        current_time = time.time()
        
        # Remove old timestamps
        while ORDER_TIMESTAMPS and current_time - ORDER_TIMESTAMPS[0] > ORDER_RATE_LIMIT_WINDOW:
            ORDER_TIMESTAMPS.popleft()
        
        if len(ORDER_TIMESTAMPS) >= MAX_ORDERS_PER_10_SECONDS:
            # Calculate how long to wait
            oldest_order = ORDER_TIMESTAMPS[0]
            wait_time = ORDER_RATE_LIMIT_WINDOW - (current_time - oldest_order) + 1
            
            if wait_time > 0:
                print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

def safe_place_order(order_func, order_description):
    """Safely place an order with rate limiting"""
    try:
        # Check rate limit before placing order
        if not check_rate_limit():
            wait_for_rate_limit()
        
        # Place the order
        result = order_func()
        record_order()
        
        print(f"   ✅ {order_description}")
        return result
        
    except Exception as e:
        print(f"   ❌ {order_description} failed: {e}")
        return None

def get_balance():
    balance = client.futures_account_balance()
    usdt_balance = float([x for x in balance if x['asset'] == 'USDT'][0]['balance'])
    return usdt_balance

def get_current_price(symbol):
    price = client.futures_symbol_ticker(symbol=symbol)
    return float(price['price'])

def get_historical_data(symbol, limit=100):
    """Get historical kline data for technical analysis"""
    klines = client.futures_klines(symbol=symbol, interval=INTERVAL, limit=limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert to numeric and set index
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def validate_1m_data(df, symbol):
    """Validate that we have sufficient 1-minute data for analysis"""
    if len(df) < RSI_PERIOD:
        print(f"⚠️  {symbol}: Insufficient data for RSI analysis (need {RSI_PERIOD}, got {len(df)})")
        return False
    
    # Check if data is actually 1-minute intervals
    if len(df) > 1:
        time_diff = df.index[-1] - df.index[-2]
        if time_diff.total_seconds() != 60:
            print(f"⚠️  {symbol}: Data intervals are not 1-minute ({time_diff.total_seconds()}s)")
            return False
    
    return True

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band, sma

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def analyze_technical_indicators(symbol):
    """Calculate technical indicators and return trading signal"""
    try:
        # Get historical 1-minute data
        df = get_historical_data(symbol, limit=100)
        
        if len(df) < 50:  # Not enough data
            return 'HOLD', 0.0, {}
            
        # Calculate indicators
        # 1. MACD
        macd_line, macd_signal, macd_histogram = calculate_macd(df['close'])
        macd_line_current = macd_line.iloc[-1]
        macd_signal_current = macd_signal.iloc[-1]
        macd_histogram_current = macd_histogram.iloc[-1]
        
        # 2. RSI
        rsi = calculate_rsi(df['close'], RSI_PERIOD)
        rsi_current = rsi.iloc[-1]
        
        # 3. Bollinger Bands
        bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(df['close'])
        bb_upper_current = bb_upper.iloc[-1]
        bb_lower_current = bb_lower.iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # 4. VWAP
        vwap = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
        vwap_current = vwap.iloc[-1]
        
        # 5. SMA (Simple Moving Average)
        sma_20 = calculate_sma(df['close'], 20)
        sma_50 = calculate_sma(df['close'], 50)
        sma_20_current = sma_20.iloc[-1]
        sma_50_current = sma_50.iloc[-1]
        
        # 6. EMA (Exponential Moving Average)
        ema_12 = calculate_ema(df['close'], 12)
        ema_26 = calculate_ema(df['close'], 26)
        ema_12_current = ema_12.iloc[-1]
        ema_26_current = ema_26.iloc[-1]
        
        # Calculate signal strength and direction - ALL indicators must be evaluated
        signals = []
        confidence = 0.0
        indicator_status = {}  # Track which indicators are active
        
        # MACD Signals (20% weight) - ALWAYS EVALUATED
        if macd_line_current > macd_signal_current and macd_histogram_current > 0:
            signals.append('BUY')
            confidence += 0.20
            indicator_status['MACD'] = 'BUY'
        elif macd_line_current < macd_signal_current and macd_histogram_current < 0:
            signals.append('SELL')
            confidence += 0.20
            indicator_status['MACD'] = 'SELL'
        else:
            indicator_status['MACD'] = 'NEUTRAL'
            
        # RSI Signals (20% weight) - ALWAYS EVALUATED
        if rsi_current < RSI_OVERSOLD:  # Oversold
            signals.append('BUY')
            confidence += 0.20
            indicator_status['RSI'] = 'BUY'
        elif rsi_current > RSI_OVERBOUGHT:  # Overbought
            signals.append('SELL')
            confidence += 0.20
            indicator_status['RSI'] = 'SELL'
        else:
            indicator_status['RSI'] = 'NEUTRAL'
            confidence += 0.05  # Small bonus for neutral RSI
            
        # Bollinger Bands Signals (15% weight) - ALWAYS EVALUATED
        if current_price <= bb_lower_current:  # Touch lower band
            signals.append('BUY')
            confidence += 0.15
            indicator_status['BB'] = 'BUY'
        elif current_price >= bb_upper_current:  # Touch upper band
            signals.append('SELL')
            confidence += 0.15
            indicator_status['BB'] = 'SELL'
        else:
            indicator_status['BB'] = 'NEUTRAL'
            
        # VWAP Signals (15% weight) - ALWAYS EVALUATED
        if current_price > vwap_current:  # Above VWAP
            signals.append('BUY')
            confidence += 0.15
            indicator_status['VWAP'] = 'BUY'
        elif current_price < vwap_current:  # Below VWAP
            signals.append('SELL')
            confidence += 0.15
            indicator_status['VWAP'] = 'SELL'
        else:
            indicator_status['VWAP'] = 'NEUTRAL'
            
        # SMA Signals (15% weight) - ALWAYS EVALUATED
        if sma_20_current > sma_50_current and current_price > sma_20_current:  # SMA crossover + price above
            signals.append('BUY')
            confidence += 0.15
            indicator_status['SMA'] = 'BUY'
        elif sma_20_current < sma_50_current and current_price < sma_20_current:  # SMA crossover + price below
            signals.append('SELL')
            confidence += 0.15
            indicator_status['SMA'] = 'SELL'
        else:
            indicator_status['SMA'] = 'NEUTRAL'
            
        # EMA Signals (15% weight) - ALWAYS EVALUATED
        if ema_12_current > ema_26_current and current_price > ema_12_current:  # EMA crossover + price above
            signals.append('BUY')
            confidence += 0.15
            indicator_status['EMA'] = 'BUY'
        elif ema_12_current < ema_26_current and current_price < ema_12_current:  # EMA crossover + price below
            signals.append('SELL')
            confidence += 0.15
            indicator_status['EMA'] = 'SELL'
        else:
            indicator_status['EMA'] = 'NEUTRAL'
        
        # Collect indicator values for detailed logging
        indicator_values = {
            'rsi': rsi_current,
            'macd_line': macd_line_current,
            'macd_signal': macd_signal_current,
            'macd_histogram': macd_histogram_current,
            'bb_upper': bb_upper_current,
            'bb_lower': bb_lower_current,
            'bb_mid': bb_mid.iloc[-1],
            'vwap': vwap_current,
            'sma_20': sma_20_current,
            'sma_50': sma_50_current,
            'ema_12': ema_12_current,
            'ema_26': ema_26_current,
            'current_price': current_price,
            'indicator_status': indicator_status
        }
        
        # Enhanced decision logic - require stronger consensus
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        total_active_indicators = buy_signals + sell_signals
        
        # Require at least 3 indicators to agree AND minimum confidence
        min_indicators_required = 3
        
        if (buy_signals >= min_indicators_required and 
            buy_signals > sell_signals and 
            confidence >= MIN_CONFIDENCE and
            total_active_indicators >= 4):  # At least 4 indicators must be active
            return 'BUY', confidence, indicator_values
        elif (sell_signals >= min_indicators_required and 
              sell_signals > buy_signals and 
              confidence >= MIN_CONFIDENCE and
              total_active_indicators >= 4):  # At least 4 indicators must be active
            return 'SELL', confidence, indicator_values
        else:
            return 'HOLD', confidence, indicator_values
            
    except Exception as e:
        print(f"⚠️  {symbol}: Error in technical analysis - {e}")
        return 'HOLD', 0.0, {}

def analyze_rsi_strategy(symbol):
    """
    RSI-focused trading strategy for 1-minute timeframe:
    - RSI > 70: Overbought → SHORT position
    - RSI < 30: Oversold → LONG position
    - RSI 30-70: HOLD
    """
    try:
        # Get historical 1-minute data
        df = get_historical_data(symbol, limit=100)
        
        # Validate 1-minute data quality
        if not validate_1m_data(df, symbol):
            return 'HOLD', 0.0, 0.0
            
        # Calculate RSI with 1-minute data
        rsi = calculate_rsi(df['close'], RSI_PERIOD)
        rsi_current = rsi.iloc[-1]
        
        # Get the latest 1-minute candle timestamp for logging
        latest_candle_time = df.index[-1].strftime('%H:%M:%S')
        
        # RSI-based decision making
        if rsi_current > RSI_OVERBOUGHT:  # Overbought - Execute SHORT
            confidence = min(0.9, (rsi_current - RSI_OVERBOUGHT) / 30 + 0.7)  # Higher confidence for more extreme RSI
            return 'SELL', confidence, rsi_current
        elif rsi_current < RSI_OVERSOLD:  # Oversold - Execute LONG
            confidence = min(0.9, (RSI_OVERSOLD - rsi_current) / 30 + 0.7)  # Higher confidence for more extreme RSI
            return 'BUY', confidence, rsi_current
        else:  # RSI between oversold and overbought: Hold
            return 'HOLD', 0.0, rsi_current
            
    except Exception as e:
        print(f"⚠️  {symbol}: Error in 1-minute RSI analysis - {e}")
        return 'HOLD', 0.0, 0.0

def get_price_precision(symbol):
    """Get price precision for each symbol"""
    precisions = {
        'BTCUSDT': 1,
        'ETHUSDT': 2,
        'ADAUSDT': 4,
        'SOLUSDT': 3,
        'SUIUSDT': 4,
        'AVAXUSDT': 3,
        'TRXUSDT': 5,
        'TONUSDT': 4,
        'LTCUSDT': 2,
        'NEARUSDT': 4,
        'XRPUSDT': 4,
        'DOGEUSDT': 5,
        'DOTUSDT': 3,
        'LINKUSDT': 3,
        'XLMUSDT': 5,
        'XMRUSDT': 2,
        'AVAAIUSDT': 4
    }
    return precisions.get(symbol, 2)

def get_tick_size(symbol):
    """Get tick size for each symbol (minimum price increment)"""
    # Cache for tick sizes to avoid repeated API calls
    if not hasattr(get_tick_size, 'cache'):
        get_tick_size.cache = {}
    
    if symbol in get_tick_size.cache:
        return get_tick_size.cache[symbol]
    
    try:
        # Get exchange info to find the exact tick size
        exchange_info = client.futures_exchange_info()
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'PRICE_FILTER':
                        tick_size = float(filter_info['tickSize'])
                        get_tick_size.cache[symbol] = tick_size
                        print(f"📊 {symbol}: Detected tick size = {tick_size}")
                        return tick_size
        
        # Fallback to static values if not found
        print(f"⚠️ {symbol}: Using fallback tick size")
        
    except Exception as e:
        print(f"⚠️ {symbol}: Error getting tick size from exchange: {e}")
    
    # Static fallback tick sizes (updated with more accurate values)
    tick_sizes = {
        'BTCUSDT': 0.10,
        'ETHUSDT': 0.01,
        'ADAUSDT': 0.00001,
        'SOLUSDT': 0.001,
        'SUIUSDT': 0.0001,
        'AVAXUSDT': 0.001,
        'TRXUSDT': 0.000001,
        'TONUSDT': 0.0001,
        'LTCUSDT': 0.01,
        'NEARUSDT': 0.0001,
        'XRPUSDT': 0.0001,
        'DOGEUSDT': 0.00001,
        'DOTUSDT': 0.001,
        'LINKUSDT': 0.001,
        'XLMUSDT': 0.00001,
        'XMRUSDT': 0.01,
        'AVAAIUSDT': 0.0001
    }
    
    fallback_tick_size = tick_sizes.get(symbol, 0.01)
    get_tick_size.cache[symbol] = fallback_tick_size
    return fallback_tick_size

def round_to_tick_size(price, tick_size):
    """Round price to the nearest valid tick size"""
    if tick_size == 0:
        return price
    
    try:
        # Use decimal arithmetic for precise rounding with small tick sizes
        from decimal import Decimal, ROUND_HALF_UP
        
        # Convert to Decimal for precise arithmetic
        price_decimal = Decimal(str(price))
        tick_size_decimal = Decimal(str(tick_size))
        
        # Calculate how many tick sizes fit into the price
        multiplier = (price_decimal / tick_size_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        
        # Round to the nearest tick size
        rounded_price = multiplier * tick_size_decimal
        
        # Convert back to float with proper precision
        result = float(rounded_price)
        
        return result
        
    except Exception as e:
        print(f"⚠️ Tick size rounding error: {e}")
        # Simple fallback rounding
        return round(price / tick_size) * tick_size

def calculate_dynamic_tp_sl(indicators, confidence, signal):
    """
    Calculate dynamic Take Profit and Stop Loss based on technical indicators
    
    Factors considered:
    - Market volatility (Bollinger Bands width, RSI extremes)
    - Trend strength (MACD momentum, EMA/SMA divergence)
    - Confidence level (higher confidence = wider TP, tighter SL)
    - Price position relative to key levels
    """
    try:
        # Base values
        dynamic_tp = BASE_TAKE_PROFIT_PCT
        dynamic_sl = BASE_STOP_LOSS_PCT
        
        # 1. Volatility Adjustment (Bollinger Bands)
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        bb_mid = indicators.get('bb_mid', 0)
        current_price = indicators.get('current_price', 0)
        
        if bb_upper > 0 and bb_lower > 0 and bb_mid > 0:
            # Calculate Bollinger Band width as % of middle band
            bb_width_pct = ((bb_upper - bb_lower) / bb_mid) * 100
            
            # High volatility (wide bands) = wider SL, moderate TP
            if bb_width_pct > 2.0:  # Very volatile
                dynamic_sl = min(dynamic_sl - 0.15, MAX_STOP_LOSS_PCT)  # Wider SL
                dynamic_tp = min(dynamic_tp + 0.1, MAX_TAKE_PROFIT_PCT)  # Slightly wider TP
            elif bb_width_pct < 0.8:  # Low volatility
                dynamic_sl = max(dynamic_sl + 0.1, MIN_STOP_LOSS_PCT)  # Tighter SL
                dynamic_tp = max(dynamic_tp - 0.1, MIN_TAKE_PROFIT_PCT)  # Tighter TP
        
        # 2. RSI Extreme Adjustment
        rsi = indicators.get('rsi', 50)
        
        if signal == 'BUY':
            # For BUY signals, if RSI is extremely oversold, use wider TP
            if rsi < 20:  # Extremely oversold
                dynamic_tp = min(dynamic_tp + 0.2, MAX_TAKE_PROFIT_PCT)
                dynamic_sl = max(dynamic_sl + 0.05, MIN_STOP_LOSS_PCT)  # Slightly tighter SL
            elif rsi < 30:  # Oversold
                dynamic_tp = min(dynamic_tp + 0.1, MAX_TAKE_PROFIT_PCT)
                
        elif signal == 'SELL':
            # For SELL signals, if RSI is extremely overbought, use wider TP
            if rsi > 80:  # Extremely overbought
                dynamic_tp = min(dynamic_tp + 0.2, MAX_TAKE_PROFIT_PCT)
                dynamic_sl = max(dynamic_sl + 0.05, MIN_STOP_LOSS_PCT)  # Slightly tighter SL
            elif rsi > 70:  # Overbought
                dynamic_tp = min(dynamic_tp + 0.1, MAX_TAKE_PROFIT_PCT)
        
        # 3. MACD Momentum Adjustment
        macd_histogram = indicators.get('macd_histogram', 0)
        
        # Strong momentum = wider TP, tighter SL
        if abs(macd_histogram) > 0.005:  # Strong momentum
            dynamic_tp = min(dynamic_tp + 0.15, MAX_TAKE_PROFIT_PCT)
            dynamic_sl = max(dynamic_sl + 0.05, MIN_STOP_LOSS_PCT)
        elif abs(macd_histogram) < 0.001:  # Weak momentum
            dynamic_tp = max(dynamic_tp - 0.1, MIN_TAKE_PROFIT_PCT)
            dynamic_sl = min(dynamic_sl - 0.05, MAX_STOP_LOSS_PCT)
        
        # 4. Confidence Level Adjustment
        if confidence >= 0.8:  # Very high confidence
            dynamic_tp = min(dynamic_tp + 0.15, MAX_TAKE_PROFIT_PCT)  # Let profits run
            dynamic_sl = max(dynamic_sl + 0.1, MIN_STOP_LOSS_PCT)   # Tighter SL
        elif confidence >= 0.7:  # High confidence
            dynamic_tp = min(dynamic_tp + 0.1, MAX_TAKE_PROFIT_PCT)
            dynamic_sl = max(dynamic_sl + 0.05, MIN_STOP_LOSS_PCT)
        elif confidence < 0.65:  # Lower confidence
            dynamic_tp = max(dynamic_tp - 0.1, MIN_TAKE_PROFIT_PCT)  # Quick profits
            dynamic_sl = min(dynamic_sl - 0.05, MAX_STOP_LOSS_PCT)   # Wider SL
        
        # 5. VWAP Position Adjustment
        vwap = indicators.get('vwap', 0)
        
        if vwap > 0 and current_price > 0:
            vwap_distance_pct = abs(current_price - vwap) / vwap * 100
            
            # If price is far from VWAP, be more conservative
            if vwap_distance_pct > 1.5:  # Far from VWAP
                dynamic_tp = max(dynamic_tp - 0.1, MIN_TAKE_PROFIT_PCT)
                dynamic_sl = min(dynamic_sl - 0.05, MAX_STOP_LOSS_PCT)
        
        # 6. EMA/SMA Trend Strength
        ema_12 = indicators.get('ema_12', 0)
        ema_26 = indicators.get('ema_26', 0)
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        
        if ema_12 > 0 and ema_26 > 0 and sma_20 > 0 and sma_50 > 0:
            # Calculate trend strength
            ema_divergence = abs(ema_12 - ema_26) / ema_26 * 100
            sma_divergence = abs(sma_20 - sma_50) / sma_50 * 100
            
            # Strong trend = wider TP
            if ema_divergence > 0.5 and sma_divergence > 0.3:
                dynamic_tp = min(dynamic_tp + 0.1, MAX_TAKE_PROFIT_PCT)
                dynamic_sl = max(dynamic_sl + 0.05, MIN_STOP_LOSS_PCT)
        
        # Ensure values are within bounds
        dynamic_tp = max(MIN_TAKE_PROFIT_PCT, min(dynamic_tp, MAX_TAKE_PROFIT_PCT))
        dynamic_sl = max(MIN_STOP_LOSS_PCT, min(dynamic_sl, MAX_STOP_LOSS_PCT))
        
        # Create adjustment summary
        tp_adjustment = dynamic_tp - BASE_TAKE_PROFIT_PCT
        sl_adjustment = dynamic_sl - BASE_STOP_LOSS_PCT
        
        adjustment_summary = {
            'dynamic_tp': dynamic_tp,
            'dynamic_sl': dynamic_sl,
            'tp_adjustment': tp_adjustment,
            'sl_adjustment': sl_adjustment,
            'volatility_factor': bb_width_pct if 'bb_width_pct' in locals() else 0,
            'momentum_factor': abs(macd_histogram),
            'confidence_factor': confidence
        }
        
        return dynamic_tp, dynamic_sl, adjustment_summary
        
    except Exception as e:
        print(f"   ⚠️ Dynamic TP/SL calculation error: {e}")
        return BASE_TAKE_PROFIT_PCT, BASE_STOP_LOSS_PCT, {}

def place_order(symbol, position_side, quantity, entry_price, indicators, confidence):
    # Calculate dynamic TP/SL based on technical indicators
    dynamic_tp, dynamic_sl, adjustment_summary = calculate_dynamic_tp_sl(indicators, confidence, position_side)
    
    # Place market entry order with rate limiting
    if position_side == 'BUY':
        # For BUY orders: SL below entry, TP above entry
        sl_price = entry_price * (1 + dynamic_sl / 100)  # Stop loss below entry
        tp_price = entry_price * (1 + dynamic_tp / 100)  # Take profit above entry
        
        # Market entry order
        market_order = safe_place_order(
            lambda: client.futures_create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity
            ),
            f"Market BUY order for {symbol}"
        )
        
    else:
        # For SELL orders: SL above entry, TP below entry
        sl_price = entry_price * (1 - dynamic_sl / 100)  # Stop loss above entry
        tp_price = entry_price * (1 - dynamic_tp / 100)  # Take profit below entry
        
        # Market entry order
        market_order = safe_place_order(
            lambda: client.futures_create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            ),
            f"Market SELL order for {symbol}"
        )

    print(f"🎯 {symbol} - Entry: {position_side}, Qty: {quantity}")
    print(f"   📊 Dynamic TP/SL Applied:")
    print(f"   🎯 TP: {dynamic_tp:.2f}% (Base: {BASE_TAKE_PROFIT_PCT:.2f}%, Adj: {adjustment_summary.get('tp_adjustment', 0):.2f}%)")
    print(f"   🛡️  SL: {dynamic_sl:.2f}% (Base: {BASE_STOP_LOSS_PCT:.2f}%, Adj: {adjustment_summary.get('sl_adjustment', 0):.2f}%)")
    print(f"   📈 Prices: TP ${tp_price:.3f}, SL ${sl_price:.3f}")
    print(f"   📊 Factors: Vol={adjustment_summary.get('volatility_factor', 0):.2f}%, Mom={adjustment_summary.get('momentum_factor', 0):.4f}, Conf={adjustment_summary.get('confidence_factor', 0):.2f}")
    print(f"   🎯 Indicator-Based Adaptive Risk Management")

    # Get proper tick size for this symbol
    tick_size = get_tick_size(symbol)
    print(f"   🔧 Using tick size: {tick_size} for {symbol}")
    
    # Round prices to valid tick sizes with error handling
    try:
        tp_price_rounded = round_to_tick_size(tp_price, tick_size)
        sl_price_rounded = round_to_tick_size(sl_price, tick_size)
        
        # Validate that TP and SL prices are different from entry price
        if tp_price_rounded == entry_price:
            print(f"   ⚠️ TP price equals entry price, adjusting...")
            if position_side == 'BUY':
                tp_price_rounded = entry_price + tick_size
            else:
                tp_price_rounded = entry_price - tick_size
        
        if sl_price_rounded == entry_price:
            print(f"   ⚠️ SL price equals entry price, adjusting...")
            if position_side == 'BUY':
                sl_price_rounded = entry_price - tick_size
            else:
                sl_price_rounded = entry_price + tick_size
        
        # Validate that the rounded prices are actually valid
        tp_remainder = tp_price_rounded % tick_size
        sl_remainder = sl_price_rounded % tick_size
        
        if abs(tp_remainder) > 1e-8:  # Check for floating point precision issues
            print(f"   ⚠️ TP price {tp_price_rounded} not aligned with tick size {tick_size} (remainder: {tp_remainder})")
        if abs(sl_remainder) > 1e-8:
            print(f"   ⚠️ SL price {sl_price_rounded} not aligned with tick size {tick_size} (remainder: {sl_remainder})")
            
    except Exception as e:
        print(f"   ⚠️ Tick size rounding failed: {e}")
        # Fallback to decimal precision rounding
        precision = get_price_precision(symbol)
        tp_price_rounded = round(tp_price, precision)
        sl_price_rounded = round(sl_price, precision)
        print(f"   🔄 Using fallback precision rounding: {precision} decimals")
    
    print(f"   🎯 Price adjustments - TP: ${tp_price:.6f} → ${tp_price_rounded} | SL: ${sl_price:.6f} → ${sl_price_rounded}")
    print(f"   📏 Tick size: {tick_size}")
    
    # Place TP order with rate limiting
    print(f"   🔄 Placing TP order: {symbol} {position_side} -> {'SELL' if position_side == 'BUY' else 'BUY'} LIMIT {quantity} @ {tp_price_rounded}")
    
    tp_order = safe_place_order(
        lambda: client.futures_create_order(
            symbol=symbol,
            side='SELL' if position_side == 'BUY' else 'BUY',
            type='LIMIT',
            quantity=quantity,
            price=tp_price_rounded,
            timeInForce='GTC'
        ),
        f"Take Profit order for {symbol} @ ${tp_price_rounded}"
    )
    
    # If TP order failed, try alternative approach
    if tp_order is None:
        print(f"   🔄 Attempting alternative TP price alignment...")
        alt_tp_price = round(tp_price_rounded, 8)  # More precision
        safe_place_order(
            lambda: client.futures_create_order(
                symbol=symbol,
                side='SELL' if position_side == 'BUY' else 'BUY',
                type='LIMIT',
                quantity=quantity,
                price=alt_tp_price,
                timeInForce='GTC'
            ),
            f"Alternative Take Profit order for {symbol} @ ${alt_tp_price}"
        )

    # Place SL order with rate limiting
    print(f"   🔄 Placing SL order: {symbol} {position_side} -> {'SELL' if position_side == 'BUY' else 'BUY'} STOP_MARKET @ {sl_price_rounded}")
    
    safe_place_order(
        lambda: client.futures_create_order(
            symbol=symbol,
            side='SELL' if position_side == 'BUY' else 'BUY',
            type='STOP_MARKET',
            stopPrice=sl_price_rounded,
            closePosition=True
        ),
        f"Stop Loss order for {symbol} @ ${sl_price_rounded}"
    )
        
    print(f"   📈 Entry complete for {symbol}")
    print("-" * 50)

def get_min_quantity(symbol):
    """Get minimum order quantity for each symbol"""
    min_quantities = {
        'BTCUSDT': 0.001,
        'ETHUSDT': 0.001,
        'ADAUSDT': 1.0,
        'SOLUSDT': 0.01,
        'SUIUSDT': 0.1,
        'AVAXUSDT': 0.01,
        'TRXUSDT': 1.0,
        'TONUSDT': 0.01,
        'LTCUSDT': 0.01,
        'NEARUSDT': 0.01,
        'XRPUSDT': 0.1,
        'DOGEUSDT': 1.0,
        'DOTUSDT': 0.01,
        'LINKUSDT': 0.01,
        'XLMUSDT': 1.0,
        'XMRUSDT': 0.001,
        'AVAAIUSDT': 0.1
    }
    return min_quantities.get(symbol, 0.001)

def get_quantity_precision(symbol):
    """Get quantity precision for each symbol"""
    precisions = {
        'BTCUSDT': 3,
        'ETHUSDT': 3,
        'ADAUSDT': 0,
        'SOLUSDT': 2,
        'SUIUSDT': 1,
        'AVAXUSDT': 2,
        'TRXUSDT': 0,
        'TONUSDT': 2,
        'LTCUSDT': 2,
        'NEARUSDT': 2,
        'XRPUSDT': 1,
        'DOGEUSDT': 0,
        'DOTUSDT': 2,
        'LINKUSDT': 2,
        'XLMUSDT': 0,
        'XMRUSDT': 3,
        'AVAAIUSDT': 1
    }
    return precisions.get(symbol, 3)

def validate_indicators():
    """Test all indicators are working properly at startup"""
    print("🔍 Validating all 6 indicators...")
    
    test_symbols = ['BTCUSDT', 'ETHUSDT']  # Test with 2 symbols
    
    for symbol in test_symbols:
        try:
            print(f"   Testing {symbol}...")
            signal, confidence, indicators = analyze_technical_indicators(symbol)
            
            if not indicators:
                print(f"   ❌ {symbol}: No indicator data returned")
                continue
                
            # Check if all indicators are present
            required_indicators = ['rsi', 'macd_histogram', 'bb_upper', 'bb_lower', 'vwap', 'sma_20', 'sma_50', 'ema_12', 'ema_26']
            missing_indicators = [ind for ind in required_indicators if ind not in indicators]
            
            if missing_indicators:
                print(f"   ❌ {symbol}: Missing indicators: {missing_indicators}")
                continue
                
            # Check indicator status
            indicator_status = indicators.get('indicator_status', {})
            expected_status = ['MACD', 'RSI', 'BB', 'VWAP', 'SMA', 'EMA']
            
            if len(indicator_status) != 6:
                print(f"   ❌ {symbol}: Expected 6 indicator statuses, got {len(indicator_status)}")
                continue
                
            active_count = sum(1 for status in indicator_status.values() if status in ['BUY', 'SELL'])
            
            print(f"   ✅ {symbol}: All indicators working - Signal: {signal}, Confidence: {confidence:.2f}")
            print(f"      Status: {indicator_status}")
            print(f"      Active indicators: {active_count}/6")
            
        except Exception as e:
            print(f"   ❌ {symbol}: Validation error - {e}")
    
    print("✅ Indicator validation complete")
    print()

def validate_dynamic_tp_sl():
    """Test the dynamic TP/SL system with sample data"""
    print("🎯 Validating Dynamic TP/SL System...")
    
    # Test scenarios
    scenarios = [
        {
            'name': 'High Volatility + High Confidence',
            'indicators': {
                'bb_upper': 52000, 'bb_lower': 48000, 'bb_mid': 50000,
                'rsi': 25, 'macd_histogram': 0.008, 'vwap': 49900,
                'current_price': 50000, 'ema_12': 50100, 'ema_26': 49900,
                'sma_20': 50050, 'sma_50': 49950
            },
            'confidence': 0.85,
            'signal': 'BUY'
        },
        {
            'name': 'Low Volatility + Low Confidence',
            'indicators': {
                'bb_upper': 50200, 'bb_lower': 49800, 'bb_mid': 50000,
                'rsi': 55, 'macd_histogram': 0.001, 'vwap': 50010,
                'current_price': 50000, 'ema_12': 50020, 'ema_26': 49980,
                'sma_20': 50010, 'sma_50': 49990
            },
            'confidence': 0.62,
            'signal': 'SELL'
        }
    ]
    
    for scenario in scenarios:
        print(f"   Testing: {scenario['name']}")
        dynamic_tp, dynamic_sl, summary = calculate_dynamic_tp_sl(
            scenario['indicators'], scenario['confidence'], scenario['signal']
        )
        
        print(f"      TP: {dynamic_tp:.2f}% (Adj: {summary.get('tp_adjustment', 0):.2f}%)")
        print(f"      SL: {dynamic_sl:.2f}% (Adj: {summary.get('sl_adjustment', 0):.2f}%)")
        print(f"      Volatility: {summary.get('volatility_factor', 0):.2f}%")
        print(f"      Momentum: {summary.get('momentum_factor', 0):.4f}")
        print(f"      Confidence: {summary.get('confidence_factor', 0):.2f}")
        print()
    
    print("✅ Dynamic TP/SL system validation complete")
    print()

def initialize_tick_sizes():
    """Initialize tick sizes for all symbols at startup"""
    print("🔧 Initializing tick sizes for all symbols...")
    for symbol in SYMBOLS:
        try:
            tick_size = get_tick_size(symbol)
            print(f"   {symbol}: {tick_size}")
        except Exception as e:
            print(f"   ❌ {symbol}: Error getting tick size - {e}")
    print("✅ Tick size initialization complete")
    print()

def show_configuration():
    """Display bot configuration for 1-minute timeframe trading"""
    print("🚀 1-Minute Multi-Indicator Scalping Bot Configuration")
    print("="*60)
    print(f"📊 Timeframe: {INTERVAL} (1-minute candles)")
    print(f"📈 Technical Indicators:")
    print(f"   • RSI Period: {RSI_PERIOD} | Overbought: {RSI_OVERBOUGHT} | Oversold: {RSI_OVERSOLD}")
    print(f"   • MACD: 12,26,9 | Bollinger Bands: 20,2")
    print(f"   • SMA: 20,50 | EMA: 12,26 | VWAP")
    print(f"⚡ Leverage: {LEVERAGE}x")
    print(f"🎯 Dynamic TP/SL System:")
    print(f"   • Base TP: {BASE_TAKE_PROFIT_PCT:.1f}% | Range: {MIN_TAKE_PROFIT_PCT:.1f}% to {MAX_TAKE_PROFIT_PCT:.1f}%")
    print(f"   • Base SL: {BASE_STOP_LOSS_PCT:.1f}% | Range: {MIN_STOP_LOSS_PCT:.1f}% to {MAX_STOP_LOSS_PCT:.1f}%")
    print(f"   • Adjustment factors: Volatility, RSI extremes, MACD momentum, Confidence, VWAP distance, Trend strength")
    print(f"💰 Capital per symbol: {MARGIN_RATIO*100}% of balance")
    print(f"⏱️  Rate Limiting Protection:")
    print(f"   • Max orders per 10s: {MAX_ORDERS_PER_10_SECONDS}")
    print(f"   • Batch size: {BATCH_SIZE} symbols")
    print(f"   • Symbol delay: {SYMBOL_PROCESSING_DELAY}s")
    print(f"   • Batch delay: 2s")
    print(f"🏆 Trading Requirements:")
    print(f"   • Minimum Confidence: {MIN_CONFIDENCE*100}%")
    print(f"   • Minimum 3 indicators must agree for same direction")
    print(f"   • Minimum 4 indicators must be active (not neutral)")
    print(f"   • Majority consensus required (more BUY than SELL or vice versa)")
    print(f"📋 Symbols: {len(SYMBOLS)} trading pairs")
    print(f"⚖️ Signal Weighting: MACD(20%), RSI(20%), VWAP(15%), BB(15%), SMA(15%), EMA(15%)")
    print("="*60)
    print()

def show_decision_requirements():
    """Display clear decision requirements"""
    print("🎯 TRADING DECISION REQUIREMENTS")
    print("="*50)
    print("✅ To OPEN a position, ALL of the following must be met:")
    print("   1. Minimum 3 indicators must agree (BUY or SELL)")
    print("   2. Minimum 4 indicators must be active (not neutral)")
    print("   3. Clear majority (more BUY than SELL, or vice versa)")
    print("   4. Minimum 60% confidence score")
    print("   5. All 6 indicators must be evaluated:")
    print("      • MACD (20% weight)")
    print("      • RSI (20% weight)")
    print("      • Bollinger Bands (15% weight)")
    print("      • VWAP (15% weight)")
    print("      • SMA (15% weight)")
    print("      • EMA (15% weight)")
    print("="*50)
    print("🎯 DYNAMIC TP/SL SYSTEM:")
    print("   • Take Profit adjusts based on trend strength and volatility")
    print("   • Stop Loss adjusts based on confidence and market conditions")
    print("   • Higher confidence = wider TP, tighter SL")
    print("   • High volatility = wider SL, moderate TP")
    print("   • Strong momentum = wider TP, tighter SL")
    print("="*50)
    print("🛡️  If ANY requirement is not met → HOLD (no trade)")
    print("🚀 This ensures only high-probability trades with optimal risk management")
    print("="*50)
    print()

def trade():
    usdt_balance = get_balance()
    # Divide capital among all symbols
    capital_per_symbol = (usdt_balance * MARGIN_RATIO) / len(SYMBOLS)
    
    # Get current leverage (since we can't change it)
    current_leverage = 125  # Default fallback (user's actual leverage)
    
    print(f"💰 Balance: ${usdt_balance:.2f} | Capital per symbol: ${capital_per_symbol:.2f}")
    print(f"📊 Analyzing 1-minute data with 6 indicators (RSI, MACD, BB, VWAP, SMA, EMA) for {len(SYMBOLS)} symbols")
    print(f"⏱️  Rate Limiting: Processing {BATCH_SIZE} symbols per batch with {SYMBOL_PROCESSING_DELAY}s delays")
    print("="*60)
    
    # Process symbols in batches to prevent rate limiting
    for i in range(0, len(SYMBOLS), BATCH_SIZE):
        batch = SYMBOLS[i:i+BATCH_SIZE]
        batch_number = i // BATCH_SIZE + 1
        total_batches = (len(SYMBOLS) - 1) // BATCH_SIZE + 1
        
        print(f"🔄 Processing batch {batch_number}/{total_batches}: {batch}")
        
        # Trade each symbol in the current batch
        for symbol_index, symbol in enumerate(batch):
            try:
                current_price = get_current_price(symbol)
                
                # Get trading signal from comprehensive technical analysis
                signal, confidence, indicators = analyze_technical_indicators(symbol)
                
                # Calculate quantity with proper precision for this symbol
                precision = get_quantity_precision(symbol)
                quantity = round(capital_per_symbol * current_leverage / current_price, precision)
                
                # Ensure minimum order size for this symbol
                min_quantity = get_min_quantity(symbol)
                if quantity < min_quantity:
                    print(f"⚠️  {symbol}: Calculated quantity {quantity} too small, using minimum {min_quantity}")
                    quantity = min_quantity

                # Execute trades based on comprehensive technical analysis
                if signal == 'BUY':
                    rsi_val = indicators.get('rsi', 0)
                    indicator_status = indicators.get('indicator_status', {})
                    
                    # Count agreeing indicators
                    buy_count = sum(1 for status in indicator_status.values() if status == 'BUY')
                    sell_count = sum(1 for status in indicator_status.values() if status == 'SELL')
                    neutral_count = sum(1 for status in indicator_status.values() if status == 'NEUTRAL')
                    
                    # Show agreeing indicators
                    buy_indicators = [name for name, status in indicator_status.items() if status == 'BUY']
                    
                    print(f"📈 {symbol}: CONSENSUS LONG - {buy_count} indicators agree (Confidence: {confidence:.2f})")
                    print(f"   🟢 BUY signals: {', '.join(buy_indicators)}")
                    print(f"   📊 RSI: {rsi_val:.1f} | MACD: {indicators.get('macd_histogram', 0):.4f} | Price vs BB: {indicators.get('current_price', 0):.2f}/{indicators.get('bb_lower', 0):.2f}")
                    print(f"   📈 Vote: {buy_count} BUY, {sell_count} SELL, {neutral_count} NEUTRAL")
                    
                    place_order(symbol, 'BUY', quantity, current_price, indicators, confidence)
                    
                elif signal == 'SELL':
                    rsi_val = indicators.get('rsi', 0)
                    indicator_status = indicators.get('indicator_status', {})
                    
                    # Count agreeing indicators
                    buy_count = sum(1 for status in indicator_status.values() if status == 'BUY')
                    sell_count = sum(1 for status in indicator_status.values() if status == 'SELL')
                    neutral_count = sum(1 for status in indicator_status.values() if status == 'NEUTRAL')
                    
                    # Show agreeing indicators
                    sell_indicators = [name for name, status in indicator_status.items() if status == 'SELL']
                    
                    print(f"📉 {symbol}: CONSENSUS SHORT - {sell_count} indicators agree (Confidence: {confidence:.2f})")
                    print(f"   🔴 SELL signals: {', '.join(sell_indicators)}")
                    print(f"   📊 RSI: {rsi_val:.1f} | MACD: {indicators.get('macd_histogram', 0):.4f} | Price vs BB: {indicators.get('current_price', 0):.2f}/{indicators.get('bb_upper', 0):.2f}")
                    print(f"   📉 Vote: {buy_count} BUY, {sell_count} SELL, {neutral_count} NEUTRAL")
                    
                    place_order(symbol, 'SELL', quantity, current_price, indicators, confidence)
                    
                else:
                    rsi_val = indicators.get('rsi', 0)
                    indicator_status = indicators.get('indicator_status', {})
                    
                    # Count all indicators
                    buy_count = sum(1 for status in indicator_status.values() if status == 'BUY')
                    sell_count = sum(1 for status in indicator_status.values() if status == 'SELL')
                    neutral_count = sum(1 for status in indicator_status.values() if status == 'NEUTRAL')
                    
                    # Show why we're holding
                    active_indicators = buy_count + sell_count
                    
                    print(f"😐 {symbol}: HOLD - Insufficient consensus (Confidence: {confidence:.2f})")
                    print(f"   ⚖️  Vote: {buy_count} BUY, {sell_count} SELL, {neutral_count} NEUTRAL")
                    print(f"   ⚠️  Need: ≥3 same signals + ≥4 active indicators (current: {active_indicators})")
                    if indicators:
                        print(f"   📊 RSI: {rsi_val:.1f} | SMA: {indicators.get('sma_20', 0):.2f}/{indicators.get('sma_50', 0):.2f}")
                
                # Add delay between symbols to prevent rate limiting
                if symbol_index < len(batch) - 1:  # Don't delay after the last symbol in batch
                    print(f"   ⏳ Waiting {SYMBOL_PROCESSING_DELAY}s before next symbol...")
                    time.sleep(SYMBOL_PROCESSING_DELAY)
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        # Add extra delay between batches
        if i + BATCH_SIZE < len(SYMBOLS):
            print(f"⏸️  Batch {batch_number} complete. Waiting 2s before next batch...")
            time.sleep(2)
    
    print("="*60)

def validate_rate_limiting():
    """Test the rate limiting system"""
    print("⏱️  Validating Rate Limiting System...")
    
    # Test rate limit tracking
    print(f"   • Max orders per 10s: {MAX_ORDERS_PER_10_SECONDS}")
    print(f"   • Current orders in queue: {len(ORDER_TIMESTAMPS)}")
    
    # Simulate some order timestamps
    test_start = time.time()
    for i in range(5):
        record_order()
        print(f"   • Simulated order {i+1} - Queue size: {len(ORDER_TIMESTAMPS)}")
        time.sleep(0.1)
    
    # Test rate limit check
    can_place = check_rate_limit()
    print(f"   • Can place more orders: {can_place}")
    
    # Clear test data
    ORDER_TIMESTAMPS.clear()
    
    # Test batch processing
    total_batches = (len(SYMBOLS) - 1) // BATCH_SIZE + 1
    estimated_time = (total_batches * BATCH_SIZE * SYMBOL_PROCESSING_DELAY) + ((total_batches - 1) * 2)
    
    print(f"   • Symbols: {len(SYMBOLS)} in {total_batches} batches")
    print(f"   • Estimated processing time: {estimated_time:.1f}s per cycle")
    
    print("✅ Rate limiting system validation complete")
    print()

# === LOOPING UTAMA ===
print("🤖 Starting 1-Minute Multi-Indicator Scalping Bot...")
initialize_tick_sizes()
show_configuration()
validate_indicators()
validate_dynamic_tp_sl()
validate_rate_limiting()
show_decision_requirements()

while True:
    try:
        trade()
        time.sleep(60)  # wait for next minute candle
    except Exception as e:
        print(f"❌ Main loop error: {e}")
        print("⏳ Retrying in 10 seconds...")
        time.sleep(10)
