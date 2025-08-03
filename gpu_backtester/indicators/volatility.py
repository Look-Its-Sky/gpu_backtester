import pandas as pd 
import numpy as np
import cudf, cupy

''' Average True Range (ATR) '''
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

''' True Range (TR)'''
def tr(
    high: cudf.Series | pd.Series, 
    low: cudf.Series | pd.Series, 
    close: cudf.Series | pd.Series
) -> cudf.Series | pd.Series:
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr_df = cudf.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    tr_series = tr_df.max(axis=1)

    return tr_series 