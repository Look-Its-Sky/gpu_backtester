import pandas as pd
import numpy as np
import cudf
import cupy 

''' ATR '''
def atr(
    high: cudf.Series | pd.Series, 
    low: cudf.Series | pd.Series, 
    close: cudf.Series | pd.Series, 
    period: int = 14,
) -> cudf.Series | pd.Series:
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr_df = cudf.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})

    true_range = tr_df.max(axis=1)
    atr_series = true_range.ewm(alpha=1/period, adjust=False).mean()

    return atr_series

''' ADX '''
def adx(
    high: cudf.Series | pd.Series, 
    low: cudf.Series | pd.Series, 
    close: cudf.Series | pd.Series, 
    period: int = 14,
) -> cudf.Series | pd.Series:

    # Messy but works for now
    cudf = cudf if isinstance(high, cudf.Series) else pd
    cupy = cupy if isinstance(high, cudf.Series) else np

    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr_df = cudf.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    true_range = tr_df.max(axis=1)

    correct_index = true_range.index

    high_diff = high - high.shift(1)
    low_diff = low.shift(1) - low
    
    plus_dm_condition = ((high_diff > low_diff) & (high_diff > 0)).fillna(False)
    plus_dm = cupy.where(plus_dm_condition, high_diff.fillna(0), 0)

    minus_dm_condition = ((low_diff > high_diff) & (low_diff > 0)).fillna(False)
    minus_dm = cupy.where(minus_dm_condition, low_diff.fillna(0), 0)

    smooth_plus_dm = cudf.Series(plus_dm, index=correct_index).ewm(alpha=1/period, adjust=False).mean()
    smooth_minus_dm = cudf.Series(minus_dm, index=correct_index).ewm(alpha=1/period, adjust=False).mean()
    smooth_tr = true_range.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (smooth_plus_dm / smooth_tr)
    minus_di = 100 * (smooth_minus_dm / smooth_tr)
    
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)

    dx_denominator = plus_di + minus_di
    dx = cupy.where(dx_denominator > 0, 100 * (abs(plus_di - minus_di) / dx_denominator), 0)

    adx = cudf.Series(dx, index=correct_index).ewm(alpha=1/period, adjust=False).mean()
    
    return adx

''' Aroon '''
def aroon(
    high: cudf.Series | pd.Series,
    low: cudf.Series | pd.Series,
    period: int,
) -> cudf.DataFrame | pd.DataFrame:
    periods_since_high = (period - 1) - high.rolling(period).apply(np.argmax, raw=True)
    periods_since_low = (period - 1) - low.rolling(period).apply(np.argmin, raw=True)

    upper = 100 * (period - periods_since_high) / period
    lower = 100 * (period - periods_since_low) / period

    return cudf.DataFrame({'upper': upper, 'lower': lower}) if isinstance(high, cudf.Series) else pd.DataFrame({'upper': upper, 'lower': lower})

''' Aroon Oscillator '''
def aroonosc(
    high: cudf.Series,
    low: cudf.Series,
    period: int,
):
    aroon_df = aroon(high, low, period)
    return aroon_df['upper'] - aroon_df['lower']

''' Balance of Power '''
def bop(
    high: cudf.Series | pd.Series,
    low: cudf.Series | pd.Series,
    close: cudf.Series | pd.Series,
) -> cudf.Series | pd.Series:
    price_range = high - low
    return ((close - open) / price_range).where(price_range > 0, 0)

''' Moving Average Convergence Divergence (MACD) '''
def macd(
    p: cudf.Series | pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> cudf.DataFrame | pd.DataFrame:
    fast_ema = p.ewm(span=fast_period, adjust=False).mean()
    slow_ema = p.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    histogram = macd_line - signal_line

    if isinstance(macd_line, cudf.Series):
        return cudf.DataFrame({
            'macd': macd_line, 
            'signal': signal_line,
            'histogram': histogram
        })

    else:
        return pd.DataFrame({
            'macd': macd_line, 
            'signal': signal_line,
            'histogram': histogram
        })
   
''' Money Flow Index (MFI) '''
def mfi(
    high: cudf.Series | pd.Series,
    low: cudf.Series | pd.Series,
    close: cudf.Series | pd.Series,
    volume: cudf.Series | pd.Series,
    period: int,
) -> cudf.Series | pd.Series:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))

    return mfi.fillna(0)

''' Stochastic Oscillator (STOCH) '''
def stoch(
    high: cudf.Series | pd.Series,
    low: cudf.Series | pd.Series,
    close: cudf.Series | pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> cudf.DataFrame | pd.DataFrame:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    price_range = highest_high - lowest_low
    percent_k = 100 * ((close - lowest_low) / price_range)
    
    percent_k = percent_k.where(price_range > 0, 0).fillna(0)
    percent_d = percent_k.rolling(window=d_period).mean()
    
    if isinstance(close, cudf.Series):
        return cudf.DataFrame({'k': percent_k, 'd': percent_d})
    else:
        return pd.DataFrame({'k': percent_k, 'd': percent_d})

''' Relative Strength Index (RSI) '''
def rsi(
    close: cudf.Series | pd.Series,
    period: int = 14
) -> cudf.Series | pd.Series:
    delta = close.diff(1)
    
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    
    return rsi_series.fillna(0)