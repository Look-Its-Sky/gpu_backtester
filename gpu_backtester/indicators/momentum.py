import pandas as pd
import numpy as np
import cudf
import cupy 

''' Calculates ATR on GPU '''
def atr(
    high: cudf.Series, 
    low: cudf.Series, 
    close: cudf.Series, 
    window: int = 14
) -> cudf.Series:
    window = int(window)
    
    if not isinstance(high, cudf.Series): high = cudf.Series(high)
    if not isinstance(low, cudf.Series): low = cudf.Series(low)
    if not isinstance(close, cudf.Series): close = cudf.Series(close)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr_df = cudf.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})

    true_range = tr_df.max(axis=1)
    atr_series = true_range.ewm(alpha=1/window, adjust=False).mean()

    return atr_series

''' Calculates ADX on GPU '''
def adx(
    high: cudf.Series, 
    low: cudf.Series, 
    close: cudf.Series, 
    window: int = 14
) -> cudf.Series:
    window = int(window)

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

    smooth_plus_dm = cudf.Series(plus_dm, index=correct_index).ewm(alpha=1/window, adjust=False).mean()
    smooth_minus_dm = cudf.Series(minus_dm, index=correct_index).ewm(alpha=1/window, adjust=False).mean()
    smooth_tr = true_range.ewm(alpha=1/window, adjust=False).mean()

    plus_di = 100 * (smooth_plus_dm / smooth_tr)
    minus_di = 100 * (smooth_minus_dm / smooth_tr)
    
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)

    dx_denominator = plus_di + minus_di
    dx = cupy.where(dx_denominator > 0, 100 * (abs(plus_di - minus_di) / dx_denominator), 0)

    adx = cudf.Series(dx, index=correct_index).ewm(alpha=1/window, adjust=False).mean()
    
    return adx
